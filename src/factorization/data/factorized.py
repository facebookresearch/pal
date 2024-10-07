"""
Generate synthetic data to study LLM behaviors in controlled settings.

License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

# %% Debug

import logging
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Union

import torch
from torch.distributions import Dirichlet
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Factors decomposition and recomposition
# -----------------------------------------------------------------------------


class Factorizer:
    """
    Factorizer of numbers into factors.

    Attributes
    ----------
    divisors
        List of `k` divisors to factorize numbers.
    """

    def __init__(self, divisors: torch.Tensor):
        self.divisors = divisors

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Get the factors of a number.

        Parameters
        ----------
        inputs
            List of `N` numbers to factorize.

        Returns
        -------
        factors
            Matrix `N x k` indicating the factors of each number.
        """
        factors = torch.empty((len(inputs), len(self.divisors)), dtype=torch.int)
        gen = inputs
        for i, modulo in enumerate(self.divisors):
            factors[:, i] = gen % modulo
            gen = gen // modulo
        return factors

    def recomposition(self, factors: torch.Tensor) -> torch.Tensor:
        """
        Get a number from its factors.

        Parameters
        ----------
        factors
            Matrix `N x k` indicating the factors of each number.

        Returns
        -------
        outputs
            List of `N` recomposed numbers.
        """
        outputs = torch.zeros(len(factors), dtype=torch.int)
        multiplier = 1
        for i, p in enumerate(self.divisors):
            outputs += factors[:, i] * multiplier
            multiplier *= p
        return outputs


# -----------------------------------------------------------------------------
# Factorized transform
# -----------------------------------------------------------------------------


@dataclass
class DataConfig:
    # number of input and output factors
    input_factors: list[int]
    output_factors: list[int]
    parents: list[list[int]]

    # embedding dimension of the data
    emb_dim: int = 32

    # concentration coefficient for p(y_i|x_i)
    alphas: Union[list[list[float]], list[float], float] = 1e-3

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        if self.parents is None:
            logger.info("No parents specified, assuming independence")
            self.parents = [[i] for i in range(len(self.output_factors))]

        if len(self.parents) != len(self.output_factors):
            raise ValueError("parents and output_factors must have the same length")

        if not isinstance(self.alphas, list):
            self.alphas = [float(self.alphas)] * len(self.output_factors)

        self.nb_data = reduce(mul, self.input_factors)
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
        x_factors = Factorizer(config.input_factors)(self.data)

        self.emb = torch.zeros((config.nb_data, config.emb_dim))
        for i in range(len(config.input_factors)):
            emb = torch.randn((config.input_factors[i], config.emb_dim))
            self.emb += emb[x_factors[:, i]]

        p_y_x = torch.ones((config.nb_data, *config.output_factors))
        view = [config.nb_data] + [1] * len(config.output_factors)

        for i in range(len(config.output_factors)):
            parent = config.parents[i]
            local_factors = [config.input_factors[j] for j in parent]
            x_local = Factorizer(local_factors).recomposition(x_factors[:, parent])

            output_factor = config.output_factors[i]
            alpha = config.alphas[i]
            if not isinstance(alpha, list):
                alpha = torch.full((output_factor,), alpha)
            else:
                alpha = torch.tensor(alpha)

            if len(local_factors):
                nb_factors = reduce(mul, local_factors)
            else:
                nb_factors = 1
            p_yi = Dirichlet(alpha).sample((nb_factors,))
            p_yi_x = p_yi[x_local]

            view[i + 1] = output_factor
            p_y_x *= p_yi_x.view(view)
            view[i + 1] = 1

        self.probas = p_y_x.view(config.nb_data, -1)

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
