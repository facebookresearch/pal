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
from typing import Optional, Union

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

# %% debug

input_factors = [4, 2, 3]
output_factors = [2, 2]
parents = [[0, 1], [1, 2]]
alphas = [1.0, 1.0]

nb_data = reduce(mul, input_factors)
nb_classes = reduce(mul, output_factors)

input_data = torch.arange(nb_data)
factors = Factorizer(input_factors)(input_data)

p_y_x = torch.ones((nb_data, *output_factors))
view = [nb_data] + [1] * len(output_factors)

for i in range(len(output_factors)):
    output_factor = output_factors[i]
    parent = parents[i]
    alpha = alphas[i]
    input_factor = [input_factors[j] for j in parent]
    tmp = Factorizer(input_factor)
    ids = tmp.recomposition(factors[:, parent])
    nb_options = reduce(mul, input_factor)

    if not isinstance(alpha, list):
        alpha = torch.full((output_factor,), alpha)
    else:
        alpha = torch.tensor(alpha)

    p_yi = Dirichlet(alpha).sample((nb_options,))
    # This should be changed
    p_yi_x = p_yi[ids.long()]

    view[i + 1] = output_factor
    p_y_x *= p_yi_x.view(view)
    view[i + 1] = 1

p_y_x = p_y_x.view(nb_data, -1)


# %% debug


class FactorizedProbas:
    """
    Generate a factorized transform.

    Parameters
    ----------
    input_divisors
        List of `k` divisors to factorize inputs.
    ouput_divisors
        List of `k` divisors to factorize outputs.
    input_size
        Number of inputs.
    output_size
        Number of outputs.

    Attributes
    ----------
    transforms
        List of `k` random transforms applied to each factors.
    probas
        Matrix `input_size x output_size` indicating the random transform.
    """

    def __init__(
        self,
        input_divisors: torch.Tensor,
        output_divisors: torch.Tensor,
        input_size: Optional[int] = None,
        output_size: Optional[int] = None,
    ):
        if input_size is None:
            input_size = input_divisors.prod().item()
        if output_size is None:
            output_size = output_divisors.prod().item()
        inputs = torch.arange(input_size)
        transform = self.generate_factorized_transformed(input_divisors, output_divisors, inputs)
        self.probas = self.transform_to_probas(transform, output_size)

    def generate_factorized_transformed(self, input_divisors, output_divisors, inputs):
        """
        Generate a factorized transform.
        """
        nb_factors = len(input_divisors)
        factors = Factorizer(input_divisors)(inputs)
        self.transforms = [
            self.generate_random_transform(input_divisors[i], output_divisors[i]) for i in range(nb_factors)
        ]
        transformed_factors = torch.zeros_like(factors)
        for i in range(nb_factors):
            transformed_factors[:, i] = self.transforms[i][factors[:, i]]
        transform = Factorizer(output_divisors).recomposition(transformed_factors)
        return transform

    @staticmethod
    def generate_random_transform(input_size: int, output_size: int) -> torch.Tensor:
        """
        Generate a random transform.
        """
        transform = torch.randint(0, output_size, (input_size,))
        # transform = torch.multinomial(probas, input_size, replacement=True)
        return transform

    @staticmethod
    def transform_to_probas(transform: torch.Tensor, output_size: int) -> torch.Tensor:
        """
        Generate the matrix associated with a random transform.
        """
        probas = torch.zeros(len(transform), output_size)
        probas[torch.arange(len(transform)), transform] = 1
        return probas


# -----------------------------------------------------------------------------
# Sampler classes
# -----------------------------------------------------------------------------


@dataclass
class SamplerConfig:
    # factorization
    input_divisors: list[list[int]] = None
    output_divisors: list[list[int]] = None

    # compression factor between input and output
    compression_rate: float = None
    weights: list[float] = None

    # input size
    input_size: int = None
    output_size: int = None

    # noise addition
    epsilon: float = 0

    # for random probas from Dirichlet distribution
    random: bool = False
    concentration: float = 1

    # allows calling the class with random keyword arguments
    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        if self.random:
            logger.info("Random sampler, overriding many of SamplerConfig parameters.")
            return
        print(self.input_divisors)
        self.input_divisors = [torch.tensor(p) for p in self.input_divisors]
        if self.output_divisors is None:
            self.output_divisors = [(self.compression_rate * p).ceil().int() for p in self.input_divisors]
        else:
            self.output_divisors = [torch.tensor(q) for q in self.output_divisors]
        if self.input_size is None:
            self.input_size = max(p.prod().item() for p in self.input_divisors)
        if self.output_size is None:
            self.output_size = max(q.prod().item() for q in self.output_divisors)


class Sampler:
    """
    Sampler class.

    Attributes
    ----------
    input_size
        The size of the input space.
    output_size
        The size of the output space.
    probas
        Matrix of size input_size x output_size.
        Each row is a probability distribution over the output space.
        It corresponds to the conditional probability of the output given the input.
    """

    def __init__(self, config: SamplerConfig):
        self.input_size = config.input_size
        self.output_size = config.output_size
        if config.random:
            concentration = config.concentration
            self.random_init(concentration)
        else:
            all_probas = [
                FactorizedProbas(p, q, self.input_size, self.output_size).probas
                for p, q in zip(config.input_divisors, config.output_divisors)
            ]
            weights = config.weights
            epsilon = config.epsilon
            self.sparse_init(all_probas, weights, epsilon)

    def to(self, device):
        self.probas = self.probas.to(device)
        return self

    def random_init(self, concentration: Union[float, list[float]]):
        """
        Init output distributions from Dirichlet distributions.

        Parameters
        ----------
        concentration
            The `input_size` concentration parameter of the Dirichlet distributions.
        """
        if isinstance(concentration, float) or isinstance(concentration, int):
            concentration = [concentration] * self.output_size
        if not isinstance(concentration, torch.Tensor):
            concentration = torch.tensor(concentration).float()
        generator = Dirichlet(concentration)
        self.probas = generator.sample((self.input_size,))

    def sparse_init(self, all_probas: list[torch.Tensor], weights: list[float] = None, epsilon: float = 0):
        """
        Init output distribution by aggregating factorized sampler together.

        Parameters
        ----------
        all_probas
            The list of probability matrices.
        weights
            The logits weights to aggregate probability matrices.
        epsilon
            The probability of choosing the uniform sampler.
        """
        self.probas = torch.ones((self.input_size, self.output_size))
        self.probas *= epsilon / self.output_size
        if weights is None:
            weights = torch.zeros(len(all_probas), dtype=torch.float)
        else:
            weights = torch.tensor(weights, dtype=torch.float)
        weights = torch.exp(weights)
        weights /= torch.sum(weights)
        weights *= 1 - epsilon
        for weight, probas in zip(weights, all_probas):
            self.probas += weight * probas

    def sample_in_parallel(self, n_samples: int = 1) -> torch.Tensor:
        """
        Sample outputs from each inputs in parallel.

        Parameters
        ----------
        n_samples
            Number of samples to generate.

        Returns
        -------
        samples
            Matrix of `input_size x n_samples` generated samples.
        """
        return torch.multinomial(self.probas, n_samples, replacement=True)

    def sample_at_once(self, input_id: int, n_samples: int = 1) -> torch.Tensor:
        """
        Sample output given an input.

        Parameters
        ----------
        input_id
            Input to condition the sampling on.
        n_samples
            Number of samples to generate.

        Returns
        -------
        samples
            List of `n_samples` generated samples.
        """
        return torch.multinomial(self.probas[input_id], n_samples, replacement=True)

    def __call__(self, inputs: list[int]):
        """
        Generate targets given inputs.

        Parameters
        ----------
        inputs
            List of input data.

        Returns
        -------
        outputs
            List of generated targets, sampled conditionally to the inputs.
        """
        device = inputs.device
        outputs = torch.empty(len(inputs), dtype=int, device=device)

        # generate random samples all at once
        _, counts = torch.unique(inputs, return_counts=True)
        n_samples = counts.max().item()
        samples = self.sample_in_parallel(n_samples)

        # retrieve the output in the order it was in
        indices = torch.zeros(self.input_size, dtype=int, device=device)
        for i in range(len(inputs)):
            input_id = int(inputs[i])
            outputs[i] = samples[input_id, indices[input_id]]
            indices[input_id] += 1
        return outputs
