"""
License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import hessian


def get_embeddings(num, dim, norm=True):
    """
    Random embeddings of size (num, dim).

    Parameters
    ----------
    num: int
        Number of token to embed
    dim: int
        Embedding dimension
    norm: bool, optional
        Whether to sample for Gaussian distribution or uniform on the sphere
    """
    emb = torch.randn(num, dim)
    if norm:
        emb /= emb.norm(dim=1, keepdim=True)
    else:
        emb /= sqrt(dim)
    return emb


class LN(nn.Module):
    """
    Layer normalization.
    """

    def forward(self, x):
        return x / torch.sqrt((x**2).sum(dim=1) + 1e-6).unsqueeze(1)


class AssociativeMemory(nn.Module):
    def __init__(self, E, U, random_init=True, layer_norm=False):
        """
        Associative memory model.

        Parameters
        ----------
        E: torch.Tensor of size (num, dim)
            Input embedding matrix
        U: torch.Tensor of size (cls, dim)
            Output (un)embedding matrix
        random_init: bool, optional
            Whether to initialize the weight matrix randomly.

        Abbreviations
        -------------
        nun: number of tokens
        dim: embedding dimension
        cls: num of output tokens
        """
        super().__init__()
        self.layer_norm = layer_norm
        dim = E.shape[1]
        if random_init:
            self.W = nn.Parameter(torch.randn(dim, dim) / sqrt(dim))
        else:
            assert not layer_norm, "would lead to division by zero!"
            self.W = nn.Parameter(torch.zeros(dim, dim))
        self.ln = LN()
        self.register_buffer("E", E)
        self.register_buffer("UT", U.T)

    def forward(self, x):
        """
        Compute the score :math:`log p(y|x) = u_y^\top W e_x`.

        Parameters
        ----------
        x: torch.Tensor of integers of size (bsz,)
            Input tensor made of token indices.

        Abbreviations
        -------------
        bsz: batch size
        """
        emb = self.E[x]
        if self.layer_norm or emb.size(0) < self.UT.size(1):  # if b < m
            out = emb @ self.W  # (b, d) @ (d, d) -> (b, d)  in O(bd^2)
            if self.layer_norm:
                out = self.ln(out)
            out = out @ self.UT  # (b, d) @ (d, m) -> (b, m)  in O(bdm)
        else:
            out = self.W @ self.UT  # (d, d) @ (d, m) -> (d, m)  in O(md^2)
            out = emb @ out  # (b, d) @ (d, m) -> (b, m)  in O(bdm)
        return out

    def fit(self, x):
        """
        Compute the prediction :math:`f(x) = \arg\max_y u_y^\top W e_x`.
        """
        score = self.forward(x)
        return score.argmax(dim=1)

    @torch.no_grad()
    def hessian(self, x, weight=None):
        """
        Hessian specific computation for the associative memory model

        Parameters
        ----------
        x: torch.Tensor of size (bsz,)
            Input tensor
        weight: torch.Tensor of size (bsz,), optional
            Input weight, useful to compute the population Hessian
        """
        # useful variables
        emb = self.E[x]
        bsz, dim = emb.shape
        U = self.UT.T

        score = self.forward(x)
        prob = F.softmax(score, dim=1)

        # part of the tensor that is due to e_x
        tmp_x = emb.view(bsz, dim, 1) * emb.view(bsz, 1, dim)
        if weight is not None:
            weight /= weight.sum()
            tmp_x *= weight.view(bsz, 1, 1)
        else:
            tmp_x /= bsz

        # part of the tensor that is due to u_z u_z'
        tmp = prob @ U
        tmp_z_cross = tmp.view(bsz, dim, 1) * tmp.view(bsz, 1, dim)

        # part of the tensor that is due to u_z u_z
        tmp = U.view(U.size(0), dim, 1) * U.view(U.size(0), 1, dim)
        tmp_z_inner = torch.einsum("nm,mde->nde", prob, tmp)

        # gathering the parts
        tmp_z = tmp_z_inner - tmp_z_cross

        tmp_x = tmp_x.view(bsz, dim, 1, dim, 1)
        tmp_z = tmp_z.view(bsz, 1, dim, 1, dim)
        out = (tmp_x * tmp_z).sum(dim=0)

        # collapse the 4d tensor into a 2d one
        out = out.view(dim * dim, dim * dim)
        return out

    @torch.no_grad()
    def autograd_hessian(self, x, weight=None):
        """
        Compute Hessian with PyTorch internals.

        Useful to check correctness, but much slower than `self.hessian`.
        """
        emb = self.E[x]
        bsz, dim = emb.shape

        if weight is not None:
            weight /= weight.sum()
        else:
            weight = torch.ones(bsz) / bsz

        def loss_func(inp):
            logit = emb @ inp @ self.UT
            log_likelihood = F.log_softmax(logit, dim=1)[:, 0]
            log_likelihood = log_likelihood * weight
            return -log_likelihood.sum()

        return hessian(loss_func, self.W).view(dim * dim, dim * dim)
