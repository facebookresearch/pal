"""
Multi-Layer Perceptron, mimicking a Transformer architecture without Attention.

License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable


@dataclass
class ModelConfig:
    input_size: int
    output_size: int
    emb_dim: int
    ffn_dim: int
    nb_layers: int

    # allows calling the class with random keyword arguments
    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)


class FeedForwardBlock(nn.Module):
    """
    Transformer FeedForward block.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.w1 = nn.Linear(config.emb_dim, config.ffn_dim, bias=False)
        self.w2 = nn.Linear(config.ffn_dim, config.emb_dim, bias=False)
        self.w3 = nn.Linear(config.emb_dim, config.ffn_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def get_inference_flops_per_token(self) -> int:
        """
        Compute the FLOPs at inference time.
        """
        flops_w1 = self.config.emb_dim * self.config.ffn_dim * 2
        flops_silu = self.config.ffn_dim * 2
        flops_w2 = self.config.ffn_dim * self.config.emb_dim * 2
        flops_w3 = self.config.emb_dim * self.config.ffn_dim * 2
        flops_mul = self.config.ffn_dim

        total_flops = flops_w1 + flops_silu + flops_w2 + flops_w3 + flops_mul
        return total_flops


class RMSNorm(torch.nn.Module):
    """
    Normalization layer.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def get_inference_flops_per_token(self, input_size: int) -> int:
        """
        Compute the number of FLOPs for a forward pass.
        """
        return 3 * input_size


class Model(nn.Module):
    """
    Model mimicking the Transformer architecture without Attention.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.input_size, config.emb_dim)
        self.layers: Iterable[FeedForwardBlock] = torch.nn.ModuleList(
            [FeedForwardBlock(config=config) for _ in range(config.nb_layers)]
        )
        self.output = nn.Linear(config.emb_dim, config.output_size, bias=False)
        self.norm = RMSNorm()

    def forward(self, x: torch.Tensor):
        out = self.embeddings(x)
        for layer in self.layers:
            out = layer(self.norm(out)) + out
        return self.output(self.norm(out))

    def get_inference_flops_per_token(self) -> int:
        """
        Compute the FLOPs at inference time.
        """
        total_flops = (
            0  # The embedding layer is a lookup operation, no FLOPs
            + sum(feed_forward_block.get_inference_flops_per_token() for feed_forward_block in self.layers)
            + 2 * self.config.emb_dim * self.config.output_size  # output layer
            + self.norm.get_inference_flops_per_token(input_size=self.config.emb_dim)  # normalization layer
        )
        return total_flops

    def get_flops_per_token(self) -> int:
        """
        Compute the FLOPs for a training step as the sum of FLOPs at forward and backward pass.
        """
        return (
            self.get_inference_flops_per_token()  # forward pass
            + 2 * self.get_inference_flops_per_token()  # approximation for backward pass
        )
