"""
Various utilities

License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------
# Normalization Layers
# --------------------------------------------------------------------------------

if torch.__version__ < "2.1":

    class LayerNorm(nn.Module):
        """
        LayerNorm normalization layer

        Parameters
        ----------
        fan_in: int
            input dimension
        eps: float
            epsilon value for numerical stability

        Notes
        -----
        Pytorch 2.0.1 does not have LayerNorm without bias
        """

        def __init__(self, fan_in, bias=False, eps=1e-5):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(fan_in))
            self.bias = nn.Parameter(torch.zeros(fan_in)) if bias else None
            self.eps = eps

        def forward(self, x):
            out = F.layer_norm(x, normalized_shape=self.weight.shape, weight=self.weight, bias=self.bias, eps=1e-5)
            # numerically stable version if self.eps is too small compared to float precision
            # out = F.layer_norm(
            #     x.float(), normalized_shape=self.weight.shape, weight=self.weight, bias=self.bias, eps=1e-5
            # ).type_as(x)
            return out

else:
    LayerNorm = nn.LayerNorm


class RMSNorm(nn.Module):
    """
    RMSNorm normalization layer

    Parameters
    ----------
    fan_in: int
        input dimension
    eps: float
        epsilon value for numerical stability

    Notes
    -----
    This is the normalization used by Mistral
    """

    def __init__(self, fan_in, bias=False, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(fan_in))
        self.bias = nn.Parameter(torch.zeros(fan_in)) if bias else None
        self.eps = eps

    def forward(self, x):
        norm = (x**2).mean(dim=-1, keepdim=True).sqrt() + self.eps
        # numerically stable version if self.eps is too small compared to float precision
        # norm = ((x.float() ** 2).mean(dim=-1, keepdim=True).sqrt() + self.eps).type_as(x)
        out = x / norm
        out = out * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out.type_as(x)
