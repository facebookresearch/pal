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
import uuid
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from factorization.config import DEVICE, SAVE_DIR
from factorization.data.synthetic import Sampler, SamplerConfig
from factorization.models.mlp import Model, ModelConfig

torch.random.manual_seed(0)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------


@dataclass
class ExperimentalConfig:
    # data config
    input_divisors: list[list[int]] = None
    output_divisors: list[list[int]] = None
    compression_rate: float = None
    weights: list[float] = None
    input_size: int = None
    output_size: int = None
    epsilon: float = 0
    random: bool = False
    concentration: float = 1

    # model config
    emb_dim: int = 32
    ffn_dim: int = 64
    nb_layers: int = 1

    # optimization config
    nb_epochs: int = 1000
    learning_rate: float = 1e-3
    zipf_coef: float = 2
    device: str = None

    # randomness
    seed: int = None

    # saving options
    save_weights: bool = False
    interactive: bool = True
    id: str = None

    def __post_init__(self):
        if self.id is None:
            self.id = uuid.uuid4().hex

        # dictionary representation
        self.dict_repr = asdict(self)

        if self.device is None:
            self.device = DEVICE

        if self.seed is not None:
            torch.manual_seed(seed=self.seed)

        self.data_config = SamplerConfig(
            input_divisors=self.input_divisors,
            output_divisors=self.output_divisors,
            compression_rate=self.compression_rate,
            weights=self.weights,
            input_size=self.input_size,
            output_size=self.output_size,
            epsilon=self.epsilon,
            random=self.random,
            concentration=self.concentration,
        )

        if self.input_size is None:
            self.input_size = self.data_config.input_size
        if self.output_size is None:
            self.output_size = self.data_config.output_size

        self.model_config = ModelConfig(
            input_size=self.input_size,
            output_size=self.output_size,
            emb_dim=self.emb_dim,
            ffn_dim=self.ffn_dim,
            nb_layers=self.nb_layers,
        )


def run_from_config(config: ExperimentalConfig):
    """
    Run the experiment from a configuration object.

    Parameters
    ----------
    config
        Configuration object.
    """
    logger.info(f"Running experiment with config {config}.")

    # save config
    save_dir = SAVE_DIR / "compression" / config.id
    save_dir.mkdir(exist_ok=True, parents=True)
    with open(save_dir / "config.json", "w") as f:
        json.dump(config.dict_repr, f)

    input_size = config.input_size
    nb_epochs = config.nb_epochs

    sampler = Sampler(config.data_config).to(config.device)
    model = Model(config.model_config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # define inputs, outputs, and a distribution over inputs
    inputs = torch.arange(input_size).to(dtype=int, device=DEVICE)
    targets = sampler.probas
    input_probas = (inputs.float() + 1) ** (-config.zipf_coef)
    input_probas /= input_probas.sum()

    # placeholders
    losses = torch.empty((nb_epochs, 2), device=DEVICE)

    # compute minimum loss
    all_loss = F.cross_entropy(torch.log(targets), targets, reduction="none")
    min_loss = all_loss.mean().item()
    all_loss *= input_probas
    min_weighted_loss = all_loss.sum().item()

    # training loop
    model.train()
    for epoch in (bar := tqdm(range(nb_epochs), disable=not config.interactive)):
        logits = model(inputs)
        all_loss = F.cross_entropy(logits, targets, reduction="none")
        all_weighted_loss = all_loss * input_probas

        loss = all_weighted_loss.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            losses[epoch, 0] = all_loss.sum() - min_loss
            losses[epoch, 1] = loss - min_weighted_loss

        if config.interactive:
            bar.set_postfix(loss=losses[epoch, 0].item(), weighted_loss=losses[epoch, 1].item())
        else:
            logger.info(
                f"Epoch {epoch}/{config.nb_epochs}: "
                f"loss={losses[epoch, 0].item()}, weighted_loss={losses[epoch, 1].item()}, "
            )

    # Savings
    logger.info(f"Saving results in {save_dir}.")
    save_dir.mkdir(exist_ok=True, parents=True)
    np.save(save_dir / "losses.npy", losses.cpu().numpy())
    if config.save_weights:
        torch.save(model.state_dict(), save_dir / "model.pth")


# -----------------------------------------------------------------------------
# Cli interface
# -----------------------------------------------------------------------------


def run_experiments(
    input_divisors: list[list[int]] = None,
    output_divisors: list[list[int]] = None,
    compression_rate: float = None,
    weights: list[float] = None,
    input_size: int = None,
    output_size: int = None,
    epsilon: float = 0,
    random: bool = False,
    concentration: float = 1,
    emb_dim: int = 32,
    ffn_dim: int = 64,
    nb_layers: int = 1,
    nb_epochs: int = 1000,
    learning_rate: float = 1e-3,
    zipf_coef: float = 2,
    device: str = None,
    seed: int = None,
    save_weights: bool = False,
    interactive: bool = True,
):
    """
    Run experiments with the given configurations
    """
    config = ExperimentalConfig(
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

    fire.Fire(run_experiments)
