"""
Training scripts for the compression experiment.

License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

import json
import logging
import traceback
import uuid
from dataclasses import asdict, dataclass
from itertools import product
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from factorization.config import DEVICE, SAVE_DIR
from factorization.data.factorized import DataConfig, FactorizedDataset
from factorization.models.mlp import Model, ModelConfig

torch.random.manual_seed(0)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------


@dataclass
class CompressionConfig:
    # data config
    input_factors: list[int]
    output_factors: list[int]
    data_emb_dim: int
    alphas: Union[list[float], float] = 1e-3

    # model config
    emb_dim: int = 32
    ffn_dim: int = 64
    nb_layers: int = 1

    # optimization config
    nb_epochs: int = 1000
    learning_rate: float = 1e-3

    # randomness
    seed: int = None

    # saving options
    save_ext: str = "compression"
    save_weights: bool = False
    interactive: bool = True
    id: str = None

    def __post_init__(self):
        data_config = DataConfig(
            input_factors=self.input_factors,
            output_factors=self.output_factors,
            emb_dim=self.data_emb_dim,
            alphas=self.alphas,
        )
        model_config = ModelConfig(
            input_size=self.input_size,
            output_size=self.output_size,
            emb_dim=self.emb_dim,
            ffn_dim=self.ffn_dim,
            nb_layers=self.nb_layers,
        )
        self.input_size = data_config.nb_data

        # unique identifier
        if self.id is None:
            self.id = uuid.uuid4().hex

        # dictionary representation
        self.dict_repr = asdict(self)

        self.data_config = data_config
        self.model_config = model_config
        self.device = DEVICE
        if self.seed is not None:
            torch.manual_seed(seed=self.seed)


def run_from_config(config: CompressionConfig):
    """
    Run the experiment from a configuration object.

    Parameters
    ----------
    config
        Configuration object.
    """
    logger.info(f"Running experiment with config {config}.")

    # save config
    save_dir = SAVE_DIR / config.save_ext / config.id
    save_dir.mkdir(exist_ok=True, parents=True)
    with open(save_dir / "config.json", "w") as f:
        json.dump(config.dict_repr, f)

    dataset = FactorizedDataset(config.data_config).to(config.device)
    model = Model(config.model_config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # placeholders
    nb_epochs = config.nb_epochs
    losses = torch.empty(nb_epochs, device=DEVICE)

    # compute minimum loss
    all_loss = F.cross_entropy(torch.log(targets), targets, reduction="none")
    min_loss = all_loss.mean().item()
    if np.isnan(min_loss):
        logger.warning("Minimum loss is NaN.")
    else:
        losses -= min_loss

    # training loop
    model.train()
    inputs = dataset.data
    targets = dataset.probas
    for epoch in (bar := tqdm(range(nb_epochs), disable=not config.interactive)):
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            losses[epoch] += loss
        if config.interactive:
            bar.set_postfix(loss=losses[epoch].item())
        else:
            logger.info(f"Epoch {epoch}/{config.nb_epochs}: loss={losses[epoch, 0].item()}.")

    # Savings
    logger.info(f"Saving results in {save_dir}.")
    save_dir.mkdir(exist_ok=True, parents=True)
    np.save(save_dir / "losses.npy", losses.cpu().numpy())
    if config.save_weights:
        torch.save(model.state_dict(), save_dir / "model.pth")


# -----------------------------------------------------------------------------
# Grid runs
# -----------------------------------------------------------------------------


DEFAULT_GRID = {
    "input_divisors": [
        [[2, 2, 3, 5]],
        [[3, 4, 5]],
        [[2, 5, 6]],
        [[2, 3, 10]],
        [[2, 2, 15]],
        [[5, 12]],
        [[3, 20]],
        [[2, 30]],
        [[60]],
    ],
    "output_divisors": [None],
    "compression_rate": [0.5],
    "input_size": [60],
    "output_size": [30],
    "epsilon": [1e-7],
    "emb_dim": [32],
    "ffn_dim": [64],
    "nb_layers": [1],
    "nb_epochs": [1000],
    "learning_rate": [1e-1],
    "zipf_coef": [2],
}


def run_grid(
    grid: dict[str, list[any]] = DEFAULT_GRID,
    num_tasks: int = 1,
    task_id: int = 1,
    save_weight: bool = False,
    nb_seeds: int = 1,
    **kwargs: dict[str, any],
) -> None:
    """
    Run a grid of configurations for training.

    Parameters
    ----------
    num_tasks:
        The total number of tasks to run concurrently.
    task_id:
        The ID of the current task.
    save_weight:
        Whether to save the weights.
    nb_seeds:
        The number of seeds to run.
    """
    logger.info(f"Running task {task_id}/{num_tasks}.")

    grid = (
        DEFAULT_GRID
        | grid
        | {
            "seed": range(nb_seeds),
            "save_weights": [save_weight],
        }
    )

    nb_configs = sum(1 for _ in product(*grid.values()))
    logger.info(f"Running {nb_configs} configurations with {num_tasks} tasks.")

    for i, values in enumerate(product(*grid.values())):
        # Handling the grid concurrently with many tasks
        if i % num_tasks != (task_id - 1):
            continue

        # setup configuration
        config_dict = dict(zip(grid.keys(), values)) | kwargs
        config_dict["interactive"] = False
        config = CompressionConfig(**config_dict)

        try:
            run_from_config(config)
        except Exception as e:
            logger.warning(f"Error for configuration: {config}.")
            logger.warning(traceback.format_exc())
            logger.warning(e)
            continue


# -----------------------------------------------------------------------------
# JSON interface
# -----------------------------------------------------------------------------


def run_json(file: str, num_tasks: int = 1, task_id: int = 1, **kwargs: dict[str, any]) -> None:
    """
    Run experiments from a JSON file.

    Parameters
    ----------
    num_tasks:
        The total number of tasks to run concurrently.
    task_id:
        The ID of the current task.
    file:
        The path to the JSONL file.
    kwargs:
        Additional arguments to override the configuration.
    """
    with open(file, "r") as f:
        all_configs = json.load(f)
    for i, config_dict in enumerate(all_configs):
        # Handling the grid concurrently with many tasks
        if i % num_tasks != (task_id - 1):
            continue
        try:
            config_dict |= kwargs
            config = CompressionConfig(**config_dict)
            run_from_config(config)
        except Exception as e:
            logger.warning(f"Error when loading: {config_dict}")
            logger.warning(traceback.format_exc())
            logger.warning(e)


def run_grid_json(file: str, **kwargs: dict[str, any]) -> None:
    """
    Run grid experiments from a JSON file.

    Parameters
    ----------
    num_tasks:
        The total number of tasks to run concurrently.
    kwargs:
        Additional arguments to pass to `run_grid`.
    """
    with open(file, "r") as f:
        all_grids = json.load(f)
    for grid in all_grids:
        try:
            run_grid(grid=grid, **kwargs)
        except Exception as e:
            logger.warning(f"Error when loading: {grid}")
            logger.warning(traceback.format_exc())
            logger.warning(e)


# -----------------------------------------------------------------------------
# Cli interface
# -----------------------------------------------------------------------------


def run_experiments(
    input_factors: list[int],
    output_factors: list[int],
    data_emb_dim: int,
    alphas: Union[list[float], float] = 1e-3,
    emb_dim: int = 32,
    ffn_dim: int = 64,
    nb_layers: int = 1,
    nb_epochs: int = 1000,
    learning_rate: float = 1e-3,
    seed: int = None,
    save_ext: str = "compression",
    save_weights: bool = False,
):
    """
    Run experiments with the given configurations

    Parameters
    ----------
    input_factors
        List of cardinality of the input factors.
    output_factors
        List of cardinality of the output factors.
    data_emb_dim
        Embedding dimension used for the factors generation.
    emb_dim
        Model embedding dimension.
    ffn_dim
        Hidden dimension of the feedforward layers.
    nb_layers
        Number of feedforward layers in the model.
    nb_epochs
        Number of epochs to train the model.
    learning_rate
        Learning rate for the optimizer.
    seed
        Random seed for reproducibility.
    save_ext
        Extension for the save directory.
    save_weights
        If True, save the model weights.
    """
    config = CompressionConfig(
        input_divisors=input_divisors,
        output_divisors=output_divisors,
        compression_rate=compression_rate,
        weights=weights,
        input_size=input_size,
        output_size=output_size,
        epsilon=epsilon,
        random=random,
        concentration=concentration,
        emb_dim=emb_dim,
        ffn_dim=ffn_dim,
        nb_layers=nb_layers,
        nb_epochs=nb_epochs,
        learning_rate=learning_rate,
        zipf_coef=zipf_coef,
        device=device,
        seed=seed,
        save_weights=save_weights,
        interactive=interactive,
    )
    run_from_config(config)


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
            "json": run_json,
            "grid": run_grid,
            "json-grid": run_grid_json,
        }
    )
