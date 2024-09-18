"""
Multi-Layer Perceptron.

License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F

from .utils import LayerNorm, RMSNorm

# --------------------------------------------------------------------------------
# Multi-Layer Perceptron
# --------------------------------------------------------------------------------


class MLP(nn.Module):
    """
    Multi-layer Perceptron.

    Parameters
    ----------
    config: configuration class with
        fan_in: int
            input dimension
        fan_out: int
            output dimension
        hidden_dim: int, or list of int
            hidden dimension(s) of the MLP
        activation: str
            activation function. Options are "relu", "gelu", or any name for F.name.
        bias: bool
            whether to use bias in the MLP
        residual: bool
            whether to use residual connection
        norm: str
            normalization layer. Options are "batch", "layer", or None.
        norm_bias: bool
            whether to use bias in the normalization layer
        norm_eps: float
            epsilon value for normalization layer
        dropout: float
            dropout rate
    """

    def __init__(self, config):
        super().__init__()

        if isinstance(config.hidden_dim, int):
            hidden_dim = [config.hidden_dim]
        else:
            hidden_dim = config.hidden_dim

        # linear layers
        fan_in = config.fan_in
        layers = []
        for fan_out in hidden_dim:
            layers.append(nn.Linear(fan_in, fan_out, bias=config.bias))
            fan_in = fan_out
        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(fan_in, config.fan_out, bias=config.bias)

        # normalization layer
        match config.norm.lower():
            case "layer":
                NormLayer = LayerNorm
            case "rms":
                NormLayer = RMSNorm
            case "":

                def NormLayer(*args, **kwargs):
                    return None

            case _:
                raise ValueError(f"Unknown normalization layer '{config.norm}'")
        self.norm = NormLayer(fan_in, bias=config.norm_bias, eps=config.norm_eps)
        self.residual = config.residual

        # activation function
        activation = config.activation.lower()
        match activation:
            case "square":
                self.activation = lambda x: x**2
            case _:
                try:
                    self.activation = getattr(F, activation)
                except AttributeError:
                    raise ValueError(f"Unknown activation function '{activation}'")

        # dropout regularization
        self.dropout = config.dropout

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            out = self.activation(out)
            if self.norm:
                out = self.norm(out)
            if self.residual:
                out = out + x
            x = F.dropout(out, p=self.dropout, training=self.training)
        out = self.output(x)
        return out


# --------------------------------------------------------------------------------
# Configuration Class
# --------------------------------------------------------------------------------


@dataclass
class MLPConfig:
    fan_in: int = None
    fan_out: int = None
    hidden_dim: int = None
    activation: str = "relu"
    bias: bool = False
    residual: bool = True
    norm: str = ""
    norm_bias: bool = False
    norm_eps: float = 1e-5
    dropout: float = 0.0
