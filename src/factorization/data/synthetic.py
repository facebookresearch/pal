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
# Factors decompisition and recomposition
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
        factors = torch.empty((len(inputs), len(self.divisors)), dtype=int)
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
        outputs = torch.zeros(len(factors), dtype=int)
        multiplier = 1
        for i, p in enumerate(self.divisors):
            outputs += factors[:, i] * multiplier
            multiplier *= p
        return outputs


# -----------------------------------------------------------------------------
# Factorized transform
# -----------------------------------------------------------------------------


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
        input_size: int = None,
        output_size: int = None,
    ):
        if input_size is None:
            input_size = input_divisors.prod()
        if output_size is None:
            output_size = output_divisors.prod()
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


class RandomSampler(Sampler):
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
