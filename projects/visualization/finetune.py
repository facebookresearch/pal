"""
Finetuning

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

from factorization.config import CONFIG_DIR, DEVICE, SAVE_DIR
from factorization.data.modular import DataloaderConfig, SMADataloader
from factorization.models.softmax_model import Model, ModelConfig

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
class FinetuneConfig:
    # back-end model
    save_ext: str = None
    unique_id: str = None

    # fine-tuning data
    vocab_size: int = 3
    sparsity_index: int = 5
    nb_data: int = 2048

    # optimization
    batch_size: int = 32
    nb_epochs: int = 1_000
    lr: float = 3e-3
    mlp_lr_discount: float = None

    # randomness
    seed: int = None

    # saving
    save_weights: bool = False
    interactive: bool = True


# %% Main function


def run_from_config(config: FinetuneConfig):
    """
    Run the fine-tuning process using the provided configuration.

    Parameters
    ----------
    config
        Configuration for the fine-tuning process.
    """
    if config.seed is not None:
        RNG = np.random.default_rng(config.seed)
        np.random.seed(seed=config.seed)
        torch.manual_seed(seed=config.seed)
    else:
        RNG = np.random.default_rng()

    # Load directory

    unique_id = uuid.uuid4().hex
    if config.save_ext is None:
        load_dir = SAVE_DIR / config.unique_id
        save_dir = SAVE_DIR / "finetune" / unique_id
    else:
        load_dir = SAVE_DIR / config.save_ext / config.unique_id
        save_dir = SAVE_DIR / "finetune" / config.save_ext / unique_id

    # Load pretrained config, and save config

    cfg = json.load(open(load_dir / "config.json", "r"))
    save_dir.mkdir(exist_ok=True, parents=True)
    with open(save_dir / "config.json", "w") as f:
        json.dump(cfg | asdict(config) | {"id": unique_id}, f)

    # Pretrained model

    cfg["vocab_size"] = cfg["nb_emb"]
    cfg = ModelConfig(**cfg)
    logger.info(f"Instanciating model with config: {cfg}.")
    model = Model(cfg)

    logger.info(f"Loading weights from {load_dir}.")
    weights = pickle.load(open(load_dir / "weights.pkl", "rb"))[-1]
    model.load_state_dict(weights)
    model.to(DEVICE)
    model.train()

    # Data

    data_config = DataloaderConfig(
        vocab_size=config.vocab_size,
        seq_length=cfg.seq_length,
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
    save_ext: str = None,
    unique_id: str = None,
    vocab_size: int = 3,
    sparsity_index: int = 5,
    nb_data: int = 2048,
    batch_size: int = 32,
    nb_epochs: int = 1_000,
    lr: float = 3e-3,
    mlp_lr_discount: float = None,
    seed: int = None,
    save_weights: bool = False,
) -> None:
    """
    Run experiments with the given configuration.

    Parameters
    ----------
    save_ext:
        Experiments saving folder identifier.
    unique_id:
        Unique identifier for the experiment.
    vocab_size:
        The size of the vocabulary.
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
    seed:
        The random seed.
    save_weights:
        Whether to save the weights.
    """

    config = FinetuneConfig(
        save_ext=save_ext,
        unique_id=unique_id,
        vocab_size=vocab_size,
        sparsity_index=sparsity_index,
        nb_data=nb_data,
        batch_size=batch_size,
        nb_epochs=nb_epochs,
        lr=lr,
        mlp_lr_discount=mlp_lr_discount,
        seed=seed,
        save_weights=save_weights,
    )

    logger.info(f"Running experiment with {config=}")
    run_from_config(config)


# %% Grid run


def run_grid(
    num_tasks: int = 1,
    task_id: int = 1,
    save_ext: str = None,
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
    """
    # raise NotImplementedError("This function is not implemented yet.")
    grid = {
        "save_ext": [save_ext],
        "vocab_size": [3],
        "sparsity_index": [None],
        "nb_data": [None],
        "batch_size": [None],
        "nb_epochs": [None],
        "lr": [None],
        "mlp_lr_discount": [None],
        "seed": range(nb_seeds),
        "save_weights": [save_weight],
    }

    inherited_keys = ["sparsity_index", "nb_data", "batch_size", "nb_epochs", "lr", "mlp_lr_discount"]

    # Recover pretrained configs
    config_file = CONFIG_DIR / f"{save_ext}.jsonl"
    with open(config_file, "r") as f:
        lines = f.readlines()
    pretrained_configs = []
    for line in lines:
        try:
            config = json.loads(line)
            pretrained_configs.append(config)
        except json.JSONDecodeError:
            continue

    nb1 = sum(1 for _ in product(*grid.values()))
    nb2 = len(pretrained_configs)
    logger.info(f"Running {nb1}x{nb2} configurations with {num_tasks} tasks.")

    # iterate over configurations and grid values.
    ind = 0
    for values in product(*grid.values()):
        for config in pretrained_configs:
            ind += 1
            if ind % num_tasks != task_id - 1:
                continue

            # setup configuration
            kwargs = dict(zip(grid.keys(), values))
            kwargs["unique_id"] = config["id"]
            kwargs["interactive"] = False
            for key in inherited_keys:
                if kwargs[key] is None:
                    kwargs[key] = config[key]
            config = FinetuneConfig(**kwargs)

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
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()]
    )

    fire.Fire(
        {
            "run": run_experiments,
            "grid": run_grid,
        }
    )
