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
from typing import List
import itertools

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class FactorizedTransformFactory:
    """
    This class is a factory for transformations that take their input in a cartesian product space.
    The transform can be represented as a tensor of shape (log_input_factors, emb_dim) where log_input_factors is the size of each
    factor of the product space and emb_dim is the dimension of output vectors.
    """  # TODO should the output be vectors or discrete values in another product space?

    def __init__(self, factors: List[int], output_dim: int = 1):
        self.factors = factors
        self.output_dim = output_dim

    def generate_random_gaussian(self) -> torch.Tensor:
        """
        Generates a multi-dimensional tensor of length len(log_input_factors).
        """
        return torch.randn(self.factors + [self.output_dim])  # TODO verify this makes sense.

    def generate_transform(self):
        # TODO implement good factorized transform generator.
        """
        Generate a transform with some control on the complexity of the transform.
        """
        pass


@dataclass
class FactorizedDatasetConfig:
    """
    The configuration to build a factorized dataset.
    """

    # number of input and output factors
    log_input_factors: list[int]
    # embedding dimension of the data
    emb_dim: int


class FactorizedDataset:
    """
    This class is an iterator that links discrete product-space tuples (labels) to continuous embeddings (vectors).
    The labels are tuples from a product space. (e.g. [0, 1, 0, 2] for 2x3x2 data)
    The vectors are a continuous representation of the labels.
    They are computed by generating a random embedding for each element of each product space,
    and then summing them up across factors of the product space.

    Attributes
    ----------
    config: FactorizedDatasetConfig
    """

    def __init__(self, config: FactorizedDatasetConfig):
        """
        Parameters
        ----------
        config:
            Configuration of the dataset
        """
        # the list of tuples in the cartesian product of the input factors
        self.tuples = torch.tensor(
            list(itertools.product(*[range(log_input_factor) for log_input_factor in config.log_input_factors]))
        )
        self.embeddings = [
            torch.randn((log_input_factor, config.emb_dim)) for log_input_factor in config.log_input_factors
        ]
        self.vectors = []
        for tuple in self.tuples:
            self.vectors.append(
                sum([self.embeddings[i][tuple[i]] for i in range(len(tuple))])
            )  # TODO verify this works + this can probably be done in tensor optimized way

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        return self.tuples[idx], self.vectors[idx]


class FactorizedTransformDataset(Dataset):
    """
    This dataset returns vectors from an embedding space (data) and their output by a function defined on a product space (labels).
    """

    def __init__(
        self,
        dataset: FactorizedDataset,
        factorized_transform: torch.Tensor,  # tensor of shape (log_input_factors, emb_dim)
    ):
        self.dataset = dataset
        self.factorized_transform = factorized_transform

        self.data = self.dataset.data
        self.labels = self.dataset.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx],
            self.factorized_transform[self.labels[idx]],
        )  # TODO might have to recast tensor with .view()

    def to(self, device):
        self.data = self.data.to(device)
        self.labels = self.labels.to(device)
        return self

    def __repr__(self):
        return f"Dataset with {self.dataset} sequences."


# Implementation of the model of interest


class Model(torch.nn.Module):
    """
    The model used to learn the factorized transform.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        # TODO implement interesting model / this is just a dummy model
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
