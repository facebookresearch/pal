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

from factorization.config import SAVE_DIR
from factorization.data.modular import DataloaderConfig, SMADataloader
from factorization.models.softmax_model import Model

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()]
)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
CONFIG_FILE = SAVE_DIR / "config.jsonl"


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
    batch_size: int = None
    nb_epochs: int = 4_000
    lr: float = 1e-2

    # adaptive updates
    adapt_method: str = None
    mlp_lr: float = None

    # model
    emb_dim: int = 2
    nb_emb: int = None
    ffn_dim: int = None
    ffn_bias: bool = True
    ffn_dropout: float = 0
    activation: float = "gelu"

    # randomness
    seed: int = None

    # saving
    save_weights: bool = False
    interactive: bool = True

    def __post_init__(self):
        if self.batch_size is None:
            self.batch_size = self.nb_data
        if self.nb_emb is None:
            self.nb_emb = self.vocab_size
        if self.ffn_dim is None:
            self.ffn_dim = 4 * self.emb_dim

        if self.adapt_method not in [None, "init", "lr"]:
            raise ValueError(f"adapt_method should be 'init', 'lr' or None, got {self.adapt_method}")
        if self.adapt_method == "lr" and self.mlp_lr is None:
            raise ValueError("'mlp_lr' is not specified")


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

    unique_id = uuid.uuid4().hex
    save_dir = SAVE_DIR / unique_id

    save_dir.mkdir(exist_ok=True, parents=True)
    with open(save_dir / "config.json", "w") as f:
        json.dump(asdict(config) | {"id": unique_id}, f)

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

    # Model

    tmp = config.vocab_size
    config.vocab_size = config.nb_emb
    model = Model(config)
    config.vocab_size = tmp
    model.to(device=DEVICE)
    logger.info(f"Model with {sum(p.numel() for p in model.parameters())} parameters, running on {DEVICE}")

    # Adapative computation
    if config.adapt_method == "init":
        model.mlp.mup_init()
    if config.adapt_method == "lr":
        mlp_params = [f"mlp.{n}" for n, p in model.mlp.named_parameters()]
        optimizer = torch.optim.Adam(
            [
                {"params": [p for n, p in model.named_parameters() if n not in mlp_params]},
                {"params": model.mlp.parameters(), "lr": config.mlp_lr},
            ],
            lr=config.lr,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

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
    batch_size: int = None,
    nb_epochs: int = 1_000,
    lr: float = 1e-2,
    adapt_method: str = None,
    mlp_lr: float = None,
    emb_dim: int = 2,
    nb_emb: int = None,
    ffn_dim: int = 10,
    ffn_bias: bool = True,
    ffn_dropout: float = 0,
    activation: float = "gelu",
    seed: int = None,
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
    adapt_method:
        The adaptation method.
    mlp_lr:
        The MLP learning rate.
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
        adapt_method=adapt_method,
        mlp_lr=mlp_lr,
        emb_dim=emb_dim,
        nb_emb=nb_emb,
        ffn_dim=ffn_dim,
        ffn_bias=ffn_bias,
        ffn_dropout=ffn_dropout,
        activation=activation,
        seed=seed,
        save_weights=save_weights,
    )

    run_from_config(config)


# %% Grid run


def run_grid(
    num_tasks=1,
    task_id=1,
) -> None:
    """
    Run a grid of configurations for training.

    Parameters
    ----------
    num_tasks:
        The total number of tasks to run concurrently.
    task_id:
        The ID of the current task.
    """
    grid = {
        "vocab_size": [2],
        "seq_length": [12],
        "sparsity_index": [5],
        "nb_data": [2048],
        "batch_size": [None, 32],
        "nb_epochs": [1_000],
        "lr": [1e-2],
        "mlp_adpat": [None, "init", "lr"],
        "mlp_lr": [1e-3],
        "emb_dim": [2],
        "nb_emb": [3],
        "ffn_dim": [8, 16, 32, 128],
        "ffn_bias": [True],
        "ffn_dropout": [0],
        "activation": ["gelu"],
        "seed": range(100),
        "save_weights": [True],
    }
    all_configs = product(*grid.values())

    logger.info(f"Running {len(list(all_configs))} configurations with {num_tasks} tasks.")

    for i, values in enumerate(product(*grid.values())):
        # Handling the grid concurrently with many tasks
        if i % num_tasks != (task_id - 1):
            continue

        # setup configuration
        kwargs = {"interactive": False}
        for k, v in zip(grid.keys(), values):
            kwargs[k] = v
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

    fire.Fire(
        {
            "run": run_experiments,
            "grid": run_grid,
        }
    )

# %%
