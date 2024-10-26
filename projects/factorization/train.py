"""
Factorization Experiments Scripts.

License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

import json
import logging
import math
import uuid
from dataclasses import asdict, dataclass
from functools import partial, reduce
from operator import mul
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from factorization.config import DEVICE, SAVE_DIR
from factorization.data.factorized import DataConfig, FactorizedDataset
from factorization.io.launchers import run_grid, run_grid_json, run_json
from factorization.models.mlp import Model, ModelConfig

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Scheduler
# -----------------------------------------------------------------------------


class MyScheduler(LRScheduler):
    def __init__(self, optimizer, T=1e6, eta_min=3e-4, last_epoch=-1):
        self.per = math.log(T)
        self.end = math.log(eta_min)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        x = (1 + math.cos(math.pi * math.log(self.last_epoch + 1) / self.per)) / 2
        inis = [math.log(base_lr) for base_lr in self.base_lrs]
        return [math.exp(x * ini + (1 - x) * self.end) for ini in inis]


# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------


@dataclass
class ExperimentalConfig:
    # data config
    input_factors: list[int]
    output_factors: list[int]
    nb_parents: int = 2
    beta: float = None
    alphas: Union[list[list[float]], list[float], float] = 1e-1
    data_split: float = 0.9

    # model config
    emb_dim: int = 64
    ratio_dim: int = 2
    ffn_dim: int = None
    nb_layers: int = 1

    # optimization config
    nb_epochs: int = 1_000
    learning_rate: float = 3e-2
    batch_size: int = None
    scheduler: str = "cosine"

    # experimental mode
    mode: str = "generalization"

    # randomness
    seed: int = None
    graph_seed: int = None

    # saving options
    save_ext: str = None
    save_weights: bool = False
    interactive: bool = True
    unique_id: str = None

    def __post_init__(self):
        if self.graph_seed is None:
            self.graph_seed = self.seed

        # useful to ensure graph filtration
        if self.graph_seed is not None:
            torch.manual_seed(seed=self.graph_seed)

        self.mode = self.mode.lower()
        if self.mode not in ["iid", "compression", "generalization"]:
            raise ValueError(f"Invalid mode: {self.mode}.")

        if self.beta is not None:
            logger.info("Drawing edges from random Bernoulli.")
            self.parents = [
                [j for j in range(len(self.input_factors)) if torch.rand(1) < self.beta]
                for _ in range(len(self.output_factors))
            ]
        else:
            logger.info(f"Drawing {self.nb_parents} edges for each output factor.")
            self.parents = [
                torch.randperm(len(self.input_factors))[: self.nb_parents].tolist()
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
        self.statistical_complexity = 0
        self.compression_complexity = 0
        for i, out_factor in enumerate(self.output_factors):
            if len(self.parents[i]):
                in_factor = reduce(mul, [self.input_factors[p] for p in self.parents[i]])

                self.statistical_complexity += in_factor * out_factor
                self.compression_complexity += min(in_factor, out_factor)

        logger.info(f"Complexity: {self.statistical_complexity}, {self.compression_complexity}.")

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
            "parents": self.parents,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "statistical_complexity": self.statistical_complexity,
            "compression_complexity": self.compression_complexity,
        }

        self.data_config = data_config
        self.model_config = model_config
        self.device = DEVICE

        if self.seed is not None:
            torch.manual_seed(seed=self.seed)

        self.save_dir = SAVE_DIR / self.save_ext / self.unique_id
        self.save_dir.mkdir(exist_ok=True, parents=True)


def run_from_config(config: ExperimentalConfig):
    """
    Run the experiment from a configuration object.

    Parameters
    ----------
    config
        Configuration object.
    """
    if config.mode == "iid":
        iid_run_from_config(config)
    elif config.mode == "compression":
        compression_run_from_config(config)
    elif config.mode == "generalization":
        generalization_run_from_config(config)
    else:
        raise ValueError(f"Invalid mode: {config.mode}.")


def iid_run_from_config(config: ExperimentalConfig):
    """
    Run the experiment in iid mode from a configuration object.

    Parameters
    ----------
    config
        Configuration object.
    """
    logger.info(f"Running experiment with config {config}.")

    # save config
    with open(config.save_dir / "config.json", "w") as f:
        json.dump(config.dict_repr, f)

    dataset = FactorizedDataset(config.data_config).to(config.device)
    model = Model(config.model_config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, config.nb_epochs)
    else:
        logger.info("Using custom scheduler.")
        scheduler = MyScheduler(optimizer)

    all_inputs = dataset.data
    probas = dataset.probas

    # placeholders
    nb_epochs = config.nb_epochs
    losses = torch.zeros((nb_epochs + 1, 2), device=DEVICE)

    # compute minimum loss
    min_loss = Categorical(probs=dataset.probas).entropy().mean().item()
    if np.isnan(min_loss):
        logger.warning("Minimum loss is NaN.")
    else:
        losses -= min_loss

    # training loop
    model.train()
    for epoch in (bar := tqdm(range(nb_epochs), disable=not config.interactive)):
        inputs = torch.randint(0, config.input_size, (config.batch_size,), device=DEVICE)
        targets = torch.multinomial(dataset.probas[inputs], 1).squeeze()
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            losses[epoch, 0] += loss
            logits = model(all_inputs)
            losses[epoch, 1] += F.cross_entropy(logits, probas)

        if config.interactive:
            bar.set_postfix(train_loss=losses[epoch, 0].item(), test_loss=losses[epoch, 1].item())
        else:
            logger.info(
                f"Epoch {epoch}/{config.nb_epochs}: losses={losses[epoch, 0].item()}, {losses[epoch, 1].item()}."
            )

    # Savings
    logger.info(f"Saving results in {config.save_dir}.")
    np.save(config.save_dir / "losses.npy", losses.cpu().numpy())
    if config.save_weights:
        torch.save(model.state_dict(), config.save_dir / "model.pth")


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
    with open(config.save_dir / "config.json", "w") as f:
        json.dump(config.dict_repr, f)

    dataset = FactorizedDataset(config.data_config).to(config.device)
    model = Model(config.model_config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, config.nb_epochs)
    else:
        logger.info("Using custom scheduler.")
        scheduler = MyScheduler(optimizer)

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
        scheduler.step()

        with torch.no_grad():
            losses[epoch] += loss
        if config.interactive:
            bar.set_postfix(loss=losses[epoch].item())
        else:
            logger.info(f"Epoch {epoch}/{config.nb_epochs}: loss={losses[epoch].item()}.")

    # Savings
    logger.info(f"Saving results in {config.save_dir}.")
    np.save(config.save_dir / "losses.npy", losses.cpu().numpy())
    if config.save_weights:
        torch.save(model.state_dict(), config.save_dir / "model.pth")


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
    with open(config.save_dir / "config.json", "w") as f:
        json.dump(config.dict_repr, f)

    dataset = FactorizedDataset(config.data_config).to(config.device)
    model = Model(config.model_config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, config.nb_epochs)
    else:
        logger.info("Using custom scheduler.")
        scheduler = MyScheduler(optimizer)

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
        logger.info(f"Minimum train loss is {min_loss}.")
        losses[:, 0] -= min_loss
    min_loss = Categorical(probs=targets[test_indices]).entropy().mean().item()
    if np.isnan(min_loss):
        logger.warning("Minimum loss is NaN.")
        losses[-1, 1] -= min_loss
    else:
        logger.info(f"Minimum test loss is {min_loss}.")
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
            scheduler.step()

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
    logger.info(f"Saving results in {config.save_dir}.")
    np.save(config.save_dir / "losses.npy", losses.cpu().numpy())
    if config.save_weights:
        torch.save(model.state_dict(), config.save_dir / "model.pth")


# -----------------------------------------------------------------------------
# Cli interface
# -----------------------------------------------------------------------------


def run_experiments(
    input_factors: list[int],
    output_factors: list[int] = None,
    parents: list[list[int]] = None,
    beta: float = 1.0,
    alphas: Union[list[list[float]], list[float], float] = 1e-3,
    data_split: float = 0.8,
    emb_dim: int = 32,
    ratio_dim: int = 4,
    ffn_dim: int = None,
    nb_layers: int = 1,
    nb_epochs: int = 1000,
    learning_rate: float = 1e-3,
    batch_size: int = None,
    mode: str = "iid",
    seed: int = None,
    graph_seed: int = None,
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
    beta
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
        Experimental mode: iid, compression, or generalization.
    seed
        Random seed for reproducibility.
    graph_seed
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
        beta=beta,
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
        graph_seed=graph_seed,
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
            "json": partial(run_json, run_from_config, ExperimentalConfig),
            "grid": partial(run_grid, run_from_config, ExperimentalConfig),
            "json-grid": partial(run_grid_json, run_from_config, ExperimentalConfig),
        }
    )
