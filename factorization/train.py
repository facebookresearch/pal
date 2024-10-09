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
from functools import reduce
from itertools import product
from operator import mul
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from factorization.config import DEVICE, SAVE_DIR
from factorization.data.factorized import DataConfig, FactorizedDataset
from factorization.models.mlp import Model, ModelConfig

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------


@dataclass
class ExperimentalConfig:
    # data config
    input_factors: list[int]
    output_factors: list[int]
    parents: list[list[int]] = None
    bernouilli: Union[list[float], float] = 0.5
    alphas: Union[list[list[float]], list[float], float] = 1e-2
    data_split: float = 0.9

    # model config
    emb_dim: int = 32
    ratio_dim: int = 2
    ffn_dim: int = None
    nb_layers: int = 1

    # optimization config
    nb_epochs: int = 10_000
    learning_rate: float = 1e-3
    batch_size: int = None

    # experimental mode
    mode: str = "generalization"

    # randomness
    seed: int = None
    bernouilli_seed: int = None

    # saving options
    save_ext: str = None
    save_weights: bool = False
    interactive: bool = True
    unique_id: str = None

    def __post_init__(self):
        if self.bernouilli_seed is None:
            self.bernouilli_seed = self.seed

        # useful to ensure graph filtration
        if self.bernouilli_seed is not None:
            torch.manual_seed(seed=self.bernouilli_seed)

        self.mode = self.mode.lower()
        if self.mode not in ["compression", "generalization"]:
            raise ValueError(f"Invalid mode: {self.mode}.")

        if self.parents is None:
            logger.info("Parents not specified. Drawing edges from random Bernouilli.")
            self.parents = [
                [j for j in range(len(self.input_factors)) if torch.rand(1) < self.bernouilli]
                for _ in range(len(self.output_factors))
            ]

        data_config = DataConfig(
            input_factors=self.input_factors,
            output_factors=self.output_factors,
            parents=self.parents,
            emb_dim=self.emb_dim,
            alphas=self.alphas,
        )

        # some statistics
        self.input_size = data_config.nb_data
        self.output_size = data_config.nb_classes
        self.data_complexity = 0
        self.output_complexity = 0
        self.input_complexity = 0
        for i, out_factor in enumerate(self.output_factors):
            if len(self.parents[i]):
                in_factor = reduce(mul, [self.input_factors[p] for p in self.parents[i]])
                self.data_complexity += in_factor * out_factor
                self.output_complexity += out_factor
                self.input_complexity += in_factor
        logger.info(f"Data complexity: {self.data_complexity}.")

        if self.ffn_dim is None:
            logger.info("Setting `ffn_dim` to `ratio_dim * emb_dim`.")
            self.ffn_dim = int(self.emb_dim * self.ratio_dim)

        model_config = ModelConfig(
            input_size=self.input_size,
            output_size=self.output_size,
            emb_dim=self.emb_dim,
            ffn_dim=self.ffn_dim,
            nb_layers=self.nb_layers,
        )

        # saving identifier
        if self.unique_id is None:
            self.unique_id = uuid.uuid4().hex
        if self.save_ext is None:
            self.save_ext = self.mode

        # dictionary representation
        self.dict_repr = asdict(self) | {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "data_complexity": self.data_complexity,
            "input_complexity": self.input_complexity,
            "output_complexity": self.output_complexity,
        }

        self.data_config = data_config
        self.model_config = model_config
        self.device = DEVICE

        if self.seed is not None:
            torch.manual_seed(seed=self.seed)


def run_from_config(config: ExperimentalConfig):
    """
    Run the experiment from a configuration object.

    Parameters
    ----------
    config
        Configuration object.
    """
    if config.mode == "generalization":
        generalization_run_from_config(config)
    elif config.mode == "compression":
        compression_run_from_config(config)
    else:
        raise ValueError(f"Invalid mode: {config.mode}.")


def generalization_run_from_config(config: ExperimentalConfig):
    """
    Run the experiment in generalization mode from a configuration object.

    Parameters
    ----------
    config
        Configuration object.
    """
    logger.info(f"Running experiment with config {config}.")

    # save config
    save_dir = SAVE_DIR / config.save_ext / config.unique_id
    save_dir.mkdir(exist_ok=True, parents=True)
    with open(save_dir / "config.json", "w") as f:
        json.dump(config.dict_repr, f)

    dataset = FactorizedDataset(config.data_config).to(config.device)
    model = Model(config.model_config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # shared embeddings
    model.embeddings.weight.data = dataset.emb
    model.embeddings.weight.requires_grad = False

    inputs = dataset.data
    targets = dataset.probas
    random_indices = torch.randperm(len(inputs), device=DEVICE)
    n_train = int(config.input_size * config.data_split)
    train_indices = random_indices[:n_train]
    test_indices = random_indices[n_train:]

    # placeholders
    nb_epochs = config.nb_epochs
    losses = torch.zeros([nb_epochs + 1, 2], device=DEVICE)

    # compute minimum loss
    min_loss = Categorical(probs=targets[train_indices]).entropy().mean().item()
    if np.isnan(min_loss):
        logger.warning("Minimum loss is NaN.")
        losses[-1, 0] -= min_loss
    else:
        logger.warning(f"Minimum train loss is {min_loss}.")
        losses[:, 0] -= min_loss
    min_loss = Categorical(probs=targets[test_indices]).entropy().mean().item()
    if np.isnan(min_loss):
        logger.warning("Minimum loss is NaN.")
        losses[-1, 1] -= min_loss
    else:
        logger.warning(f"Minimum test loss is {min_loss}.")
        losses[:, 1] -= min_loss

    # define dataloader
    if config.batch_size is None:
        config.batch_size = config.input_size
    train_batch_size = min(len(train_indices), config.batch_size)
    test_batch_size = min(len(test_indices), config.batch_size)
    trainloader = DataLoader(
        TensorDataset(inputs[train_indices], targets[train_indices]),
        batch_size=train_batch_size,
        shuffle=True,
    )
    testloader = DataLoader(
        TensorDataset(inputs[test_indices], targets[test_indices]),
        batch_size=test_batch_size,
        shuffle=False,
    )

    # training loop
    model.train()
    for epoch in (bar := tqdm(range(nb_epochs), disable=not config.interactive)):
        running_loss = 0.0
        for inputs, targets in trainloader:
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                running_loss += loss.item()

        with torch.no_grad():
            running_loss /= len(trainloader)
            losses[epoch, 0] += running_loss

            running_loss = 0
            for inputs, targets in testloader:
                logits = model(inputs)
                loss = F.cross_entropy(logits, targets)
                running_loss += loss.item()
            running_loss /= len(testloader)
            losses[epoch, 1] += loss
        if config.interactive:
            bar.set_postfix(train_loss=losses[epoch, 0].item(), test_loss=losses[epoch, 1].item())
        else:
            logger.info(
                f"Epoch {epoch}/{config.nb_epochs}: losses={losses[epoch, 0].item()}, {losses[epoch, 1].item()}."
            )

    # Savings
    logger.info(f"Saving results in {save_dir}.")
    save_dir.mkdir(exist_ok=True, parents=True)
    np.save(save_dir / "losses.npy", losses.cpu().numpy())
    if config.save_weights:
        torch.save(model.state_dict(), save_dir / "model.pth")


def compression_run_from_config(config: ExperimentalConfig):
    """
    Run the experiment in compression mode from a configuration object.

    Parameters
    ----------
    config
        Configuration object.
    """
    logger.info(f"Running experiment with config {config}.")

    # save config
    save_dir = SAVE_DIR / config.save_ext / config.unique_id
    save_dir.mkdir(exist_ok=True, parents=True)
    with open(save_dir / "config.json", "w") as f:
        json.dump(config.dict_repr, f)

    dataset = FactorizedDataset(config.data_config).to(config.device)
    model = Model(config.model_config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    inputs = dataset.data
    targets = dataset.probas

    # placeholders
    nb_epochs = config.nb_epochs
    losses = torch.zeros(nb_epochs + 1, device=DEVICE)

    # compute minimum loss
    min_loss = Categorical(probs=targets).entropy().mean().item()
    if np.isnan(min_loss):
        logger.warning("Minimum loss is NaN.")
    else:
        losses -= min_loss

    # training loop
    model.train()
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
            logger.info(f"Epoch {epoch}/{config.nb_epochs}: loss={losses[epoch].item()}.")

    # Savings
    logger.info(f"Saving results in {save_dir}.")
    save_dir.mkdir(exist_ok=True, parents=True)
    np.save(save_dir / "losses.npy", losses.cpu().numpy())
    if config.save_weights:
        torch.save(model.state_dict(), save_dir / "model.pth")


# -----------------------------------------------------------------------------
# Grid runs
# -----------------------------------------------------------------------------


def run_grid(
    grid: dict[str, list[any]],
    num_tasks: int = 1,
    task_id: int = 1,
    save_weight: bool = False,
    nb_seeds: int = 1,
    nb_bernouilli_seeds: int = None,
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

    grid |= {
        "seed": range(nb_seeds),
        "bernouilli_seed": range(nb_bernouilli_seeds) if nb_bernouilli_seeds is not None else [None],
        "save_weights": [save_weight],
    }

    nb_configs = sum(1 for _ in product(*grid.values()))
    logger.info(f"Running {nb_configs} configurations with {num_tasks} tasks.")

    for i, values in enumerate(product(*grid.values())):
        # Handling the grid concurrently with many tasks
        if i % num_tasks != (task_id - 1):
            continue

        # setup configuration
        config_dict = dict(zip(grid.keys(), values)) | kwargs
        config_dict["interactive"] = False
        config = ExperimentalConfig(**config_dict)

        save_dir = SAVE_DIR / config.save_ext / config.unique_id
        save_dir.mkdir(exist_ok=True, parents=True)
        with open(save_dir / "task_id", "w") as f:
            f.write(str(task_id))

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
            config = ExperimentalConfig(**config_dict)
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
    output_factors: list[int] = None,
    parents: list[list[int]] = None,
    bernouilli: Union[list[float], float] = 1.0,
    alphas: Union[list[list[float]], list[float], float] = 1e-3,
    data_split: float = 0.8,
    emb_dim: int = 32,
    ratio_dim: int = 4,
    ffn_dim: int = None,
    nb_layers: int = 1,
    nb_epochs: int = 1000,
    learning_rate: float = 1e-3,
    batch_size: int = None,
    mode: str = "generalization",
    seed: int = None,
    bernouilli_seed: int = None,
    save_ext: str = "interactive",
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
    parents
        List of parents for each output factor.
    bernouilli
        If `parents` is not specified, it will be defined randomly.
        This parameter speicficies the probability of edges between input and output factors.
    alphas
        Concentration coefficient for the conditional distribution `p(y_i | x_i)`.
        The conditional are drawn from a Dirichlet distribution with concentration `alpha`.
    data_split
        Proportion of the data used for training.
    emb_dim
        Model embedding dimension.
    ratio_dim
        Ratio between the embedding dimension and the MLP hidden dimension.
    ffn_dim
        Hidden dimension of the feedforward layers.
    nb_layers
        Number of feedforward layers in the model.
    nb_epochs
        Number of epochs to train the model.
    learning_rate
        Learning rate for the optimizer.
    batch_size
        Batch size for training and testing.
    mode
        Experimental mode: generalization or compression.
    seed
        Random seed for reproducibility.
    bernouilli_seed
        Random seed for the graph filtration.
    save_ext
        Extension for the save directory.
    save_weights
        If True, save the model weights.
    """
    config = ExperimentalConfig(
        input_factors=input_factors,
        output_factors=output_factors,
        parents=parents,
        bernouilli=bernouilli,
        alphas=alphas,
        data_split=data_split,
        emb_dim=emb_dim,
        ratio_dim=ratio_dim,
        ffn_dim=ffn_dim,
        nb_layers=nb_layers,
        nb_epochs=nb_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        mode=mode,
        seed=seed,
        bernouilli_seed=bernouilli_seed,
        save_ext=save_ext,
        save_weights=save_weights,
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
