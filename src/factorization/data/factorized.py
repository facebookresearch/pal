"""
Generate synthetic data to study LLM behaviors in controlled settings.

License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

import logging
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Union

import torch
from torch.distributions import Dirichlet
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    # number of input and output factors
    log_input_factors: list[int]
    output_factors: list[int]

    # embedding dimension of the data
    emb_dim: int

    # concentration coefficient for p(y_i|x_i)
    alphas: Union[list[float], float] = 1e-3

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        if len(self.log_input_factors) != len(self.output_factors):
            raise ValueError("log_input_factors and output_factors must have the same length")

        if not isinstance(self.alphas, list):
            self.alphas = [float(self.alphas)] * len(self.output_factors)

        self.nb_data = 2 ** sum(self.log_input_factors)
        self.nb_classes = reduce(mul, self.output_factors)


class FactorizedDataset(Dataset):
    """
    Dataset with Factorized Structured

    Attributes
    ----------
    emb:
        Embedding of each input vector
    probas:
        Conditional probabilities of target given inputs
    """

    def __init__(self, config: DataConfig):
        """
        Parameters
        ----------
        config:
            Configuration of the dataset
        """
        self.data = torch.arange(config.nb_data)
        self.emb = torch.randn((config.nb_data, config.emb_dim))
        p_y_x = torch.ones((config.nb_data, *config.output_factors))

        view = [config.nb_data] + [
            1,
        ] * len(config.log_input_factors)

        for i in range(len(config.log_input_factors)):
            alpha = config.alphas[i]
            log_input_factor = config.log_input_factors[i]
            output_factor = config.output_factors[i]

            directions = torch.randn((config.emb_dim, log_input_factor))
            sign = (self.emb @ directions).sign()
            value = (torch.tensor([2**i for i in range(log_input_factor)]) * sign).sum(axis=1)

            alphas = torch.full((output_factor,), alpha)
            p_yi = Dirichlet(concentration=alphas).sample((2**log_input_factor,))
            p_yis_x = p_yi[value.long()]

            view[i + 1] = output_factor
            p_y_x *= p_yis_x.view(view)
            view[i + 1] = 1

        p_y_x = p_y_x.view(config.nb_data, -1)
        self.probas = p_y_x

    def to(self, device):
        self.data = self.data.to(device)
        self.emb = self.emb.to(device)
        self.probas = self.probas.to(device)
        return self

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target = torch.multinomial(self.probas[idx], 1).item()
        return self.data[idx], target

    def __repr__(self):
        return f"Dataset with {self.n} sequences among {self.p ** self.len} unique ones."
