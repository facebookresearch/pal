"""
License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

from math import sqrt

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


def write_numpy_file(filepath, x, overwrite=False):
    if overwrite:
        mode = "wb"
    else:
        mode = "ab"
    with open(filepath, mode) as f:
        f.write(x.tobytes())


def read_numpy_file(filepath, dtype=np.float64, shape=None, order="C"):
    with open(filepath, "rb") as f:
        tmp = f.read()
    out = np.frombuffer(tmp, dtype=dtype)
    if shape is not None:
        if order == "F":
            out = out.reshape(shape[::-1])
            out = out.T
        else:
            out = out.reshape(shape)
    return out


def get_embeddings(n, d, norm=True):
    emb = th.randn(n, d)
    if norm:
        emb /= emb.norm(dim=1, keepdim=True)
    else:
        emb /= sqrt(d)
    return emb


class LN(nn.Module):
    def forward(self, x):
        return x / th.sqrt((x**2).sum(dim=1) + 1e-6).unsqueeze(1)


class AssMem(nn.Module):
    def __init__(self, d, n, m, use_ln=False):
        """
        Association Memory Scheme.

        d: int
            Memory dimensionality
        n: int
            Number of input tokens
        m: int
            Number of classes
        """
        super().__init__()
        self.W = nn.Parameter(th.randn(d, d) / sqrt(d))
        # self.E = get_embeddings(n, d, norm=False)
        # self.U = get_embeddings(m, d, norm=True).T
        self.E = th.randn(n, d) / sqrt(d)
        self.UT = th.randn(d, m) / sqrt(d)
        self.use_ln = use_ln
        self.ln = LN()

    def forward(self, x):
        out = self.E[x] @ self.W
        if self.use_ln:
            out = self.ln(out)
        out = out @ self.UT
        return out


class AssMemExp(nn.Module):
    def __init__(self, d, n, m, use_ln=False):
        """
        Association Memory Scheme with exponential memory capacity.

        d: int
            Memory dimensionality
        n: int
            Number of input tokens
        m: int
            Number of classes
        """
        super().__init__()
        self.E = nn.Parameter(th.randn(n, d) / sqrt(d))
        self.UT = nn.Parameter(th.randn(d, m) / sqrt(d))
        self.use_ln = use_ln
        self.ln = LN()

    def forward(self, x):
        out = self.E[x]
        out = F.relu(out)
        if self.use_ln:
            out = self.ln(out)
        out = out @ self.UT
        return out


class AssMemLearnable(nn.Module):
    def __init__(self, d, n, m, use_ln=False):
        """
        Association Memory Scheme with learnable embeddings.

        d: int
            Memory dimensionality
        n: int
            Number of input tokens
        m: int
            Number of classes
        """
        super().__init__()
        # self.W = nn.Parameter(th.eye(d, d))
        self.W = nn.Parameter(th.randn(d, d) / sqrt(d))
        self.E = nn.Parameter(th.randn(n, d) / sqrt(d))
        self.UT = nn.Parameter(th.randn(d, m) / sqrt(d))
        self.use_ln = use_ln
        self.ln = LN()

    def forward(self, x):
        out = self.E[x] @ self.W
        if self.use_ln:
            out = self.ln(out)
        out = out @ self.UT
        return out
