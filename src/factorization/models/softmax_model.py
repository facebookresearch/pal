"""
Simple Transformer-like Architecture.
"""

import math
from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------
# Embedding Module
# --------------------------------------------------------------------------------


class Embedding(nn.Module):
    """
    Embedding layer.

    Parameters
    ----------
    config: configuration class with
        vocab_size: int
            token vocabulary size
        emb_dim: int
            embedding dimension of the tokens
        pos_emb: bool
            whether to use positional embedding
        seq_len: int
            maximum sequence length (required if pos_emb is True)
        emb_dropout: float
            dropout probability for the embeddings layer
    """

    def __init__(self, config):
        super().__init__()

        # token embedding
        self.token_emb = nn.Embedding(config.vocab_size, config.emb_dim)

        # position embedding
        if config.pos_emb:
            self.L = config.seq_len
            self.pos_dim = config.pos_dim
            self.pos_emb = nn.Embedding(self.L, self.pos_dim).requires_grad_(False if config.freeze_pos else True)
        else:
            self.pos_emb = None

        # dropout regularization
        self.dropout = config.emb_dropout

    def forward(self, x):
        out = self.token_emb(x)
        if self.pos_emb is not None:
            L = x.size(1)
            assert L <= self.L, f"Input sequence length {L} is longer than the maximum sequence length {self.L}"
            out[..., : self.pos_dim] = out[..., : self.pos_dim] + self.pos_emb.weight[:L]
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


# --------------------------------------------------------------------------------
# Transformer-like Module
# --------------------------------------------------------------------------------


class SoftmaxLayer(nn.Module):
    def __init__(self, emb_dim):
        super(SoftmaxLayer, self).__init__()
        self.emb_dim = emb_dim
        self.query = nn.Linear(emb_dim, 1, bias=False)
        self.value = nn.Linear(emb_dim, emb_dim, bias=False)

    def forward(self, x, verbose=False):
        # x: [bsz, seq_len, emb_dim]
        query = self.query.weight
        key = x
        value = self.value(x)

        attn = query @ key.transpose(-1, -2) / math.sqrt(self.emb_dim)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ value).squeeze(1)
        if verbose:
            return out, attn.squeeze(1)
        return out


# --------------------------------------------------------------------------------
# Feedfoward Module
# --------------------------------------------------------------------------------


class TransformerFeedForward(nn.Module):
    """
    Feed-forward network in transformer architecture.

    Parameters
    ----------
    config: configuration class with
        emb_dim: int
            embedding dimension of the inputs
        ffn_dim: int
            hidden dimension of the MLP
        activation: str
            activation function. Options are "relu", "gelu".
        ffn_bias: bool
            whether to use bias in the MLP
        ffn_dropout: float
            dropout probability
    """

    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.emb_dim, config.ffn_dim, bias=config.ffn_bias)
        self.fc2 = nn.Linear(config.ffn_dim, config.emb_dim, bias=config.ffn_bias)
        self.dropout = config.ffn_dropout

        # Parsing the activation function
        activation = config.activation.lower()
        self.activation = getattr(F, activation, None)
        if self.activation is None:
            raise ValueError(f"Unknown activation function '{config.activation}'")

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


# -------------------------------------------------------------------------------
# Normalization Module
# -------------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """
    RMSNorm normalization layer
    """

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        norm = (x**2).mean(dim=-1, keepdim=True).sqrt() + self.eps
        out = x / norm
        return out.type_as(x)


# -------------------------------------------------------------------------------
# Complete Architecture
# -------------------------------------------------------------------------------


@dataclass
class ModelConfig:
    activation: float = "gelu"
    emb_dim: bool = None
    ffn_dim: bool = None
    ffn_bias: bool = False
    ffn_dropout: float = 0

    def __post_init__(self):
        if self.ffn_dim is None:
            self.ffn_dim = 4 * self.emb_dim


class Model(nn.Module):
    def __init__(self, emb_dim, vocab_size, length, ffn_dim=None):
        if ffn_dim is None:
            ffn_dim = 4 * emb_dim
        super(Model, self).__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(length, emb_dim)

        self.softmax = SoftmaxLayer(emb_dim)
        config = ModelConfig(
            emb_dim=emb_dim,
            ffn_dim=ffn_dim,
            ffn_bias=True,
            activation="gelu",
        )
        self.mlp = TransformerFeedForward(config)

        self.output = nn.Linear(emb_dim, vocab_size, bias=False)
        self.output.weight = self.token_emb.weight

        self.norm1 = RMSNorm()
        self.norm2 = RMSNorm()

    def forward(self, x, verbose=False):
        out = self.token_emb(x) + self.pos_emb.weight
        out = self.softmax(self.norm1(out), verbose=verbose)
        if verbose:
            out, attn = out
        out = out + self.mlp(self.norm2(out))
        out = self.output(out)
        if verbose:
            return out, attn
        return out