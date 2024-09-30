"""
Training scripts

License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

# %% Imports

import copy
import json
import logging
import pickle
import traceback
import uuid
from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from factorization.config import DEVICE, SAVE_DIR
from factorization.data.modular import DataloaderConfig, SMADataloader
from factorization.models.softmax_model import Model

logger = logging.getLogger(__name__)


# %% Utils


def copy_weights(model: Model) -> dict[str, torch.Tensor]:
    """
    Return a copy of the model's state_dict

    Parameters
    ----------
    model:
        The model whose state_dict will be copied.

    Returns
    -------
    A copy of the model's state_dict.
    """
    if model.output.weight.device == torch.device("cpu"):
        return {k: copy.deepcopy(v) for k, v in model.state_dict().items()}
    else:
        return {k: v.cpu().detach() for k, v in model.state_dict().items()}


@dataclass
class ExperimentConfig:
    # data
    vocab_size: int = 2
    seq_length: int = 12
    sparsity_index: int = 5
    nb_data: int = 2048

    # optimization
    batch_size: int = 32
    nb_epochs: int = 1_000
    lr: float = 3e-3
    mlp_lr_discount: float = None

    # model
    emb_dim: int = 2
    nb_emb: int = None
    ffn_dim: int = None
    ffn_bias: bool = True
    ffn_dropout: float = 0
    activation: float = "gelu"

    # randomness
    seed: int = None
    MAX_FFN_DIM: int = 2048

    # saving
    save_ext: str = None
    save_weights: bool = False
    interactive: bool = True
    id: str = None

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        if self.batch_size is None:
            self.batch_size = self.nb_data
        if self.nb_emb is None:
            self.nb_emb = self.vocab_size
        if self.ffn_dim is None:
            self.ffn_dim = 4 * self.emb_dim
        if self.id is None:
            self.id = uuid.uuid4().hex


# %% Main function


def run_from_config(config: ExperimentConfig):
    """
    Run the training loop using the provided configuration.

    Parameters
    ----------
    config:
        The configuration object containing the experiment parameters.
    """
    if config.seed is not None:
        RNG = np.random.default_rng(config.seed)
        np.random.seed(seed=config.seed)
        torch.manual_seed(seed=config.seed)
    else:
        RNG = np.random.default_rng()

    if config.save_ext is None:
        save_dir = SAVE_DIR / config.id
    else:
        save_dir = SAVE_DIR / config.save_ext / config.id

    save_dir.mkdir(exist_ok=True, parents=True)
    with open(save_dir / "config.json", "w") as f:
        json.dump(asdict(config), f)

    # Data

    data_config = DataloaderConfig(
        vocab_size=config.vocab_size,
        seq_length=config.seq_length,
        sparsity_index=config.sparsity_index,
        nb_data=config.nb_data,
        batch_size=config.batch_size,
        mode="train",
    )
    train_loader = SMADataloader(data_config, rng=RNG, device=DEVICE)
    data_config.mode = "test"
    test_loader = SMADataloader(data_config, rng=RNG, device=DEVICE)

    logger.info(f"Training set: {train_loader.dataset}")
    logger.info(f"Testing set: {test_loader.dataset}")

    # Consistent random initialization when varying ffn_dim
    nb_params = 2 * config.emb_dim + 2
    with torch.no_grad():
        params = torch.rand(config.MAX_FFN_DIM, nb_params)
        params = params[: config.ffn_dim]
        params *= 2
        params -= 1

    # Model

    tmp = config.vocab_size
    config.vocab_size = config.nb_emb
    model = Model(config)
    config.vocab_size = tmp
    model.mlp.set_parameters(params)
    model.to(device=DEVICE)
    logger.info(f"Model with {sum(p.numel() for p in model.parameters())} parameters, running on {DEVICE}")

    # Adapative updates

    if config.mlp_lr_discount is not None:
        mlp_params = [f"mlp.{n}" for n, p in model.mlp.named_parameters()]
        parameters = [
            {"params": [p for n, p in model.named_parameters() if n not in mlp_params]},
            {"params": model.mlp.parameters(), "lr": config.lr / config.mlp_lr_discount},
        ]
    else:
        parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=config.lr)

    # Training Loop

    losses = torch.zeros(config.nb_epochs)
    test_losses = torch.zeros(config.nb_epochs)
    accs = torch.zeros(config.nb_epochs)
    test_accs = torch.zeros(config.nb_epochs)

    if config.save_weights:
        weights = [copy_weights(model)]

    X_train = train_loader.dataset.data
    Y_train = train_loader.dataset.targets
    X_test = test_loader.dataset.data
    Y_test = test_loader.dataset.targets

    for epoch in (bar := tqdm(range(config.nb_epochs), disable=not config.interactive)):

        for X, Y in train_loader:
            # X = X.to(device=DEVICE)
            # Y = Y.to(device=DEVICE)

            optimizer.zero_grad()

            # forward
            score = model(X, verbose=False)
            loss = F.cross_entropy(score, Y)

            # backward
            loss.backward()
            optimizer.step()

        # record statistics
        with torch.no_grad():
            score = model(X_train, verbose=False)
            loss = F.cross_entropy(score, Y_train)
            losses[epoch] = loss.item()
            accs[epoch] = (score.argmax(-1) == Y_train).float().mean()
            score_test = model(X_test)
            test_losses[epoch] = F.cross_entropy(score_test, Y_test)
            test_accs[epoch] = (score_test.argmax(-1) == Y_test).float().mean()
            if config.save_weights:
                weights.append(copy_weights(model))

        if config.interactive:
            bar.set_postfix(loss=losses[epoch].item(), acc=accs[epoch].item(), test_acc=test_accs[epoch].item())
        else:
            logger.info(
                f"Epoch {epoch}/{config.nb_epochs}: "
                f"loss={losses[epoch].item()}, "
                f"acc={accs[epoch].item()}, test_acc={test_accs[epoch].item()}"
            )

    # Saving results

    logger.info(f"Saving results in {save_dir}.")
    save_dir.mkdir(exist_ok=True, parents=True)
    pickle.dump(losses, open(save_dir / "losses.pkl", "wb"))
    pickle.dump(test_losses, open(save_dir / "test_losses.pkl", "wb"))
    pickle.dump(accs, open(save_dir / "accs.pkl", "wb"))
    pickle.dump(test_accs, open(save_dir / "test_accs.pkl", "wb"))
    if config.save_weights:
        pickle.dump(weights, open(save_dir / "weights.pkl", "wb"))


# %% CLI Wrapper


def run_experiments(
    vocab_size: int = 2,
    seq_length: int = 12,
    sparsity_index: int = 5,
    nb_data: int = 2048,
    batch_size: int = 32,
    nb_epochs: int = 1_000,
    lr: float = 3e-3,
    mlp_lr_discount: float = None,
    emb_dim: int = 2,
    nb_emb: int = None,
    ffn_dim: int = 32,
    ffn_bias: bool = True,
    ffn_dropout: float = 0,
    activation: float = "gelu",
    seed: int = None,
    save_ext: str = None,
    save_weights: bool = False,
) -> None:
    """
    Run experiments with the given configuration.

    Parameters
    ----------
    vocab_size:
        The size of the vocabulary.
    seq_length:
        The length of the sequence.
    sparsity_index:
        The sparsity index.
    nb_data:
        The number of data.
    batch_size:
        The batch size.
    nb_epochs:
        The number of epochs.
    lr:
        The learning rate.
    mlp_lr_discount:
        Discount factor for the MLP learning rate.
    emb_dim:
        The embedding dimension.
    nb_emb:
        The number of embeddings.
    ffn_dim:
        The dimension of the feed-forward network.
    ffn_bias:
        Whether to include bias in the feed-forward network.
    ffn_dropout:
        The dropout rate for the feed-forward network.
    activation:
        The activation function.
    seed:
        The random seed.
    save_ext:
        Experiments saving folder identifier.
    save_weights:
        Whether to save the weights.
    """

    config = ExperimentConfig(
        vocab_size=vocab_size,
        seq_length=seq_length,
        sparsity_index=sparsity_index,
        nb_data=nb_data,
        batch_size=batch_size,
        nb_epochs=nb_epochs,
        lr=lr,
        mlp_lr_discount=mlp_lr_discount,
        emb_dim=emb_dim,
        nb_emb=nb_emb,
        ffn_dim=ffn_dim,
        ffn_bias=ffn_bias,
        ffn_dropout=ffn_dropout,
        activation=activation,
        seed=seed,
        save_ext=save_ext,
        save_weights=save_weights,
    )

    logger.info(f"Running experiment with {config=}")
    run_from_config(config)


def run_from_jsonl(file: str, **kwargs: dict[str, any]) -> None:
    """
    Run experiments from a JSONL file.

    Parameters
    ----------
    file:
        The path to the JSONL file.
    kwargs:
        Additional arguments to override the configuration.
    """
    with open(file, "r") as f:
        for line in f.readlines():
            try:
                config = ExperimentConfig(**json.loads(line))
                for k, v in kwargs.items():
                    if v is not None:
                        setattr(config, k, v)
                logger.info(f"Running experiment with {config=}")
                run_from_config(config)
            except Exception as e:
                logger.warning(f"Error when loading: {line}")
                logger.warning(traceback.format_exc())
                logger.warning(e)


# %% Grid run


def run_grid(
    num_tasks: int = 1,
    task_id: int = 1,
    ablation: str = None,
    save_weight: bool = False,
    nb_seeds: int = 1,
) -> None:
    """
    Run a grid of configurations for training.

    Parameters
    ----------
    num_tasks:
        The total number of tasks to run concurrently.
    task_id:
        The ID of the current task.
    ablation:
        Type of ablation study to perform.
    save_weight:
        Whether to save the weights.
    nb_seeds:
        The number of seeds to run.
    """
    grid = {
        "vocab_size": [2],
        "seq_length": [12],
        "sparsity_index": [5],
        "nb_data": [2048],
        "batch_size": [32],
        "nb_epochs": [1_000],
        "lr": [3e-3],
        "mlp_lr_discount": [None],
        "emb_dim": [2],
        "nb_emb": [3],
        "ffn_dim": [32],
        "ffn_bias": [True],
        "ffn_dropout": [0],
        "activation": ["gelu"],
        "seed": range(nb_seeds),
        "save_weights": [save_weight],
    }

    if ablation == "batch_size":
        grid["batch_size"] = np.logspace(0, 11, num=12, base=2).astype(int).tolist()
    elif ablation == "lr":
        grid["lr"] = np.logspace(0, -4, num=20).tolist()
    elif ablation == "mlp_lr":
        grid["mlp_lr_discount"] = np.logspace(-2, 2, num=20).tolist()
    elif ablation == "ffn_dim":
        grid["ffn_dim"] = np.logspace(1, 4, 20).astype(int).tolist()
        grid["MAX_FFN_DIM"] = [10_000]
    elif ablation == "ffn_bias":
        grid["ffn_bias"] = [True, False]
    elif ablation == "ffn_dropout":
        grid["ffn_dropout"] = np.linspace(0, 0.9, 20).tolist()
    elif ablation == "seed":
        grid["seed"] = range(100)

    nb_configs = sum(1 for _ in product(*grid.values()))
    logger.info(f"Running {nb_configs} configurations with {num_tasks} tasks.")
    logger.info(f"Ablation mode is {ablation}.")

    for i, values in enumerate(product(*grid.values())):
        # Handling the grid concurrently with many tasks
        if i % num_tasks != (task_id - 1):
            continue

        # setup configuration
        kwargs = dict(zip(grid.keys(), values))
        kwargs["interactive"] = False
        kwargs["save_ext"] = ablation
        config = ExperimentConfig(**kwargs)

        logger.info(f"{config=}")

        try:
            run_from_config(config)
        except Exception as e:
            logger.warning(f"Error for configuration: {config}.")
            logger.warning(traceback.format_exc())
            logger.warning(e)
            continue


# %% CLI

if __name__ == "__main__":
    import fire

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    fire.Fire(
        {
            "run": run_experiments,
            "config": run_from_jsonl,
            "grid": run_grid,
        }
    )

# %%
