"""
Generate synthetic data to study LLM behaviors in controlled settings.

License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

from typing import Union

import torch
from torch.distributions import Dirichlet

# -----------------------------------------------------------------------------
# Random transform and corresponding transform matrix
# -----------------------------------------------------------------------------


def generate_transform(input_size: int, output_size: int, probas: torch.Tensor = None) -> torch.Tensor:
    """
    Generate a random transform.

    Parameters
    ----------
    input_size
        The size of the input space.
    output_size
        The size of the output space.
    probas
        The probability of sampling the different outputs.

    Returns
    -------
    transform
        Tensor of `input_size` integers representing the random transform.
    """
    if probas is None:
        transform = torch.randint(0, output_size, (input_size,))
    else:
        transform = torch.multinomial(probas, input_size, replacement=True)
    return transform


def transform_to_probas(transform: torch.Tensor, output_size: int) -> torch.Tensor:
    """
    Generate the matrix associated with a random transform.

    Parameters
    ----------
    transform
        The random transform.
    output_size
        The size of the output space.

    Returns
    -------
    probas
        Matrix associated with the random transform.
        `probas[i, j] = 1` if the `i`-th input is transformed into the `j`-th output.
    """
    probas = torch.zeros(len(transform), output_size)
    probas[torch.arange(len(transform)), transform] = 1
    return probas


def generate_transform_matrix(input_size: int, output_size: int, probas: torch.Tensor = None) -> torch.Tensor:
    """
    Generate a random deterministic transformation matrix.

    Parameters
    ----------
    input_size
        The size of the input space.
    output_size
        The size of the output space.
    probas
        The probability of sampling the different outputs.

    Returns
    -------
    probas
        Matrix of size `input_size x output_size` representing the probability of sampling the different outputs.
    """
    transform = generate_transform(input_size, output_size, probas)
    probas = transform_to_probas(transform, output_size)
    return probas


# -----------------------------------------------------------------------------
# Factors decompisition and recomposition
# -----------------------------------------------------------------------------


class Factorizer:
    """
    Factorizer class.

    Attributes
    ----------
    ps
        List of numbers to use for the factorization.
    """

    def __init__(self, ps: torch.Tensor):
        self.ps = ps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the factors of a number.

        Parameters
        ----------
        x
            List of numbers to factorize.

        Returns
        -------
        factors
            The factors of the numbers.
        """
        factors = torch.empty((len(x), len(self.ps)), dtype=int)
        gen = x
        for i, p in enumerate(self.ps):
            factors[:, i] = gen % p
            gen = gen // p
        return factors

    def recomposition(self, factors: torch.Tensor) -> torch.Tensor:
        """
        Get the number from its factors.

        Parameters
        ----------
        factors
            List of list of factors.

        Returns
        -------
        x
            The recomposed number.
        """
        x = torch.zeros(len(factors), dtype=int)
        multiplier = 1
        for i, p in enumerate(self.ps):
            x += factors[:, i] * multiplier
            multiplier *= p
        return x


# -----------------------------------------------------------------------------
# Sampler classes
# -----------------------------------------------------------------------------


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

    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.probas = generate_transform_matrix(input_size, output_size)

    def sample(self, n_samples: int = 1):
        """
        Sample outputs.

        Parameters
        ----------
        n_samples
            The number of samples to generate.

        Returns
        -------
        samples
            The generated samples.
        """
        if hasattr(self, "probas"):
            return torch.multinomial(self.probas, n_samples, replacement=True)
        raise NotImplementedError

    def conditional_sample(self, input_id: int, n_samples: int = 1):
        """
        Sample output given an input.

        Parameters
        ----------
        input_id
            The input to condition the sampling on.

        Returns
        -------
        samples
            The generated samples.
        """
        if hasattr(self, "probas"):
            return torch.multinomial(self.probas[input_id], n_samples, replacement=True)
        raise NotImplementedError

    def generate_targets(self, inputs: list[int]):
        """
        Generate targets given inputs.

        Parameters
        ----------
        inputs
            The input data.

        Returns
        -------
        targets
            The generated targets, sampled conditionally to the inputs.
        """
        targets = torch.empty(len(inputs), dtype=torch.long)

        # generate random samples all at once
        _, counts = torch.unique(inputs, return_counts=True)
        n_samples = counts.max()
        samples = self.sample(n_samples)

        # retrieve the output in the order it was in
        indices = torch.zeros(self.input_size, dtype=torch.long)
        for i in range(len(inputs)):
            input_id = int(inputs[i])
            targets[i] = samples[input_id, indices[input_id]]
            indices[input_id] += 1

        return targets


class DirichletSampler(Sampler):
    """
    Sample output distributions from a Dirichlet distribution.
    """

    def __init__(self, input_size: int, output_size: int, alpha: Union[float, list[float]]):
        """
        Parameters
        ----------
        input_size
            The size of the input space.
        output_size
            The size of the output space.
        alpha
            The concentration parameter of the Dirichlet distribution.
        """
        if isinstance(alpha, float) or isinstance(alpha, int):
            alpha = [alpha] * output_size
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha).to(torch.float)
        self.input_size = input_size
        self.output_size = output_size
        generator = Dirichlet(alpha)
        self.probas = generator.sample((input_size,))


class AggregatedSampler(Sampler):
    """
    Aggregate sampler together.
    """

    def __init__(self, all_probas: list[torch.Tensor], weights: list[float] = None, epsilon: float = 0):
        """
        Parameters
        ----------
        all_probas
            The list of probability matrices.
        weights
            The logits weights to aggregate probability matrices.
        epsilon
            The probability of choosing the uniform sampler.
        """
        input_size, output_size = all_probas[0].shape
        self.input_size = input_size
        self.output_size = output_size
        self.probas = torch.ones((input_size, output_size))
        self.probas *= epsilon / output_size

        if weights is None:
            weights = torch.zeros(len(all_probas), dtype=torch.float)
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights).to(torch.float)
        weights = torch.exp(weights)
        weights /= torch.sum(weights)
        weights *= 1 - epsilon

        for weight, probas in zip(weights, all_probas):
            self.probas += weight * probas
