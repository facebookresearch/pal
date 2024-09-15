# %% Imports

import copy
import fcntl
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
from factorization.models.softmax_model import Model, torch_init_mlp

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()]
)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
CONFIG_FILE = SAVE_DIR / "config.jsonl"


# %% Utils


def copy_weights(model):
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
    lr_mlp: float = None

    # model
    emb_dim: int = 2
    nb_emb: int = None
    ffn_dim: int = None
    ffn_bias: bool = True
    ffn_dropout: float = 0
    activation: float = "gelu"
    mlp_speed: str = "default"

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


# %% Main function


def run_from_config(config: ExperimentConfig):
    if config.seed is not None:
        RNG = np.random.default_rng(config.seed)
        np.random.seed(seed=config.seed)
        torch.manual_seed(seed=config.seed)
    else:
        RNG = np.random.default_rng()

    unique_id = uuid.uuid4().hex
    save_dir = SAVE_DIR / unique_id

    SAVE_DIR.mkdir(exist_ok=True, parents=True)
    with open(CONFIG_FILE, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        json.dump(asdict(config) | {"id": unique_id}, f)
        f.write("\n")
        fcntl.flock(f, fcntl.LOCK_UN)

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

    # MLP initialization
    assert config.mlp_speed in [
        "default",
        "weight_init",
        "lrs",
    ], "Value should be in [default, weight_init, lrs]."
    if config.mlp_speed == "weight_init":
        torch_init_mlp(model)

    config.vocab_size = tmp
    model.to(device=DEVICE)
    logger.info(f"Model with {sum(p.numel() for p in model.parameters())} parameters, running on {DEVICE}")

    # Training Loop
    if (config.mlp_speed == "lrs") and (config.lr_mlp is not None):
        my_list = ["mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias"]
        params = [p for n, p in model.named_parameters() if n in my_list]
        base_params = [p for n, p in model.named_parameters() if n not in my_list]
        optimizer = torch.optim.Adam(
            [
                {"params": base_params},
                {"params": params, "lr": config.lr_mlp},
            ],
            lr=config.lr,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

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
    lr_mlp: float = None,
    emb_dim: int = 2,
    nb_emb: int = None,
    ffn_dim: int = 10,
    ffn_bias: bool = True,
    ffn_dropout: float = 0,
    activation: float = "gelu",
    mlp_speed: str = "default",
    seed: int = None,
    save_weights: bool = False,
):
    config = ExperimentConfig(
        vocab_size=vocab_size,
        seq_length=seq_length,
        sparsity_index=sparsity_index,
        nb_data=nb_data,
        batch_size=batch_size,
        nb_epochs=nb_epochs,
        lr=lr,
        lr_mlp=lr_mlp,
        emb_dim=emb_dim,
        nb_emb=nb_emb,
        ffn_dim=ffn_dim,
        ffn_bias=ffn_bias,
        ffn_dropout=ffn_dropout,
        activation=activation,
        mlp_speed=mlp_speed,
        seed=seed,
        save_weights=save_weights,
    )

    run_from_config(config)


# %% Grid run


def run_grid(
    num_tasks=1,
    task_id=1,
):
    grid = {
        "vocab_size": [2],
        "seq_length": [12],
        "sparsity_index": [5],
        "nb_data": [2048],
        "batch_size": [None],  # [None, 32],
        "nb_epochs": [10],  # [1_000],
        "lr": [1e-2],  # [1e-2, 1e-3, 1e-4],
        "lr_mlp": [1e-3],  # [None, 1e-3, 1e-4],
        "emb_dim": [2],  # [3],
        "nb_emb": [2],  # [3],
        "ffn_dim": [10],  # [8, 16, 32, 128],
        "ffn_bias": [False],
        "ffn_dropout": [0],
        "activation": ["gelu"],
        "mlp_speed": ["default", "weight_init", "lrs"],
        "seed": range(5),
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
