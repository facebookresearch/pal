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


class Sampler:
    """
    Attributes
    ----------
    probas
        The probability distribution of the output space.
    input_size
        The size of the input space.
    output_size
        The size of the output space.
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


class DirichletSampler(Sampler):
    """
    Sample output distribution from a Dirichlet distribution.
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
    def __init__(self, weights: list[float], all_probas: list[torch.Tensor], epsilon: float = 0):
        """
        Parameters
        ----------
        input_size
            The size of the input space.
        output_size
            The size of the output space.
        weights
            The logits weights to choose a deterministic sampler.
        all_probas
            The list of samplers to choose from.
        epsilon
            The probability of choosing the uniform sampler.
        """
        input_size, output_size = all_probas[0].shape
        super().__init__(input_size, output_size)
        self.probas = torch.ones((input_size, output_size))
        self.probas *= epsilon / output_size

        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights).to(torch.float)
        weights = torch.exp(weights)
        weights /= torch.sum(weights)
        weights *= 1 - epsilon

        for weight, probas in zip(weights, all_probas):
            self.probas += weight * probas
