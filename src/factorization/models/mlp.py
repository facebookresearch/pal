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

        self.w1 = nn.Linear(config.emb_dim, config.ffn_dim, bias=False)
        self.w2 = nn.Linear(config.ffn_dim, config.emb_dim, bias=False)
        self.w3 = nn.Linear(config.emb_dim, config.ffn_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(torch.nn.Module):
    """
    Normalization layer.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class Model(nn.Module):
    """
    Model mimicking the Transformer architecture without Attention.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embeddings = nn.Embedding(config.input_size, config.emb_dim)
        self.layers = torch.nn.ModuleList([FeedForwardBlock(config=config) for _ in range(config.nb_layers)])
        self.output = nn.Linear(config.emb_dim, config.output_size, bias=False)
        self.norm = RMSNorm()
        self.flops = self.get_flops(config.emb_dim, config.ffn_dim, config.nb_layers, config.output_size)

    def forward(self, x: torch.Tensor):
        out = self.embeddings(x)
        for layer in self.layers:
            out = layer(self.norm(out)) + out
        return self.output(self.norm(out))

    @staticmethod
    def get_flops(emb_dim: int, ffn_dim: int, nb_layers: int, output_size: int):
        """
        Get the number of flops for the model.
        """
        # lookup table
        flop_embedding = 0

        flop_norm = 3 * emb_dim
        flop_res = emb_dim

        # 3 matrix multiplications and the silu activation
        flop_linear = 3 * 2 * emb_dim * ffn_dim + 2 * ffn_dim
        flop_layer = flop_res + flop_linear + flop_norm

        flop_output = 2 * emb_dim * output_size + flop_norm

        total_flops = flop_embedding + nb_layers * flop_layer + flop_output

        # accounting for backward pass twice bigger than foward pass
        return 3 * total_flops
