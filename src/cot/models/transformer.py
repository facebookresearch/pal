"""
Transformer model.

Notes
-----
Comments abbreviations:
    N: batch size
    L: decoding sequence length
    S: encoding sequence length
    E: embedding dimension
    H: number of heads
    D: downsampling factor in attention

License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import LayerNorm, RMSNorm

# -------------------------------------------------------------------------------
# Attention Layers
# -------------------------------------------------------------------------------


class SelfAttention(nn.Module):
    """
    Self-attention layer.

    Parameters
    ----------
    config: configuration class with
        emb_dim: int
            embedding dimensionality of the input
        n_head: int
            number of attention heads (should divide emb_dim)
        attn_bias: bool
            whether to use bias in attention
        attn_dropout: float
            dropout probability
        attn_downsampling: int
            downsampling factor for key and value matrices
        flash: bool
            whether to use flash attention (could mess up half precision computation for mistral)

    TODO
    ----
    - Implement caching behavior when processing a sentence token by token at inference time (only feed next token).
    """

    def __init__(self, config):
        super().__init__()

        assert config.emb_dim % config.n_head == 0, "embedding dimension must be divisible by number of heads"

        self.H = config.n_head
        E = config.emb_dim

        # matrices
        bias = config.attn_bias
        self.qkv_mat = nn.Linear(E, 3 * E, bias=bias)
        self.output = nn.Linear(E, E, bias=bias)

        # flash attention implementation and attention mask
        self.flash = config.flash

        # needed to checked attention matrices
        L = config.seq_len
        mask = torch.ones(L, L)
        mask = torch.tril(mask, diagonal=0)
        self.register_buffer("mask", mask.view(1, 1, L, L) == 0)

        # drop-out regularization
        self.dropout = config.attn_dropout

    def forward(self, x, verbose=False):
        """
        Self attention

        Parameters
        ----------
        x: torch.Tensor (N, L, E)
            input sequence

        See Also
        --------
        `Flash Attention <https://arxiv.org/abs/2205.14135>`_.
        """
        # Query, key, value: (N, L, E) @ (E, 3 * E) -> (N, L, 3 * E) -> (N, L, E) * 3
        q, k, v = self.qkv_mat(x).chunk(3, dim=-1)

        # attention layer: (N, L, E)
        N, L, E = q.size()
        H, dim = self.H, E // self.H

        # reformating: (N, L, E) -> (N, L, H, E / H) -> (N, H, L, E / H)
        q = q.view(N, L, H, dim).transpose(1, 2)
        k = k.view(N, L, H, dim).transpose(1, 2)
        v = v.view(N, L, H, dim).transpose(1, 2)

        if not self.flash or verbose:
            # classical implementation
            # (N, H, L, E / H) @ (N, H, E / H, L) -> (N, H, L, L)
            attn = q @ k.transpose(-1, -2) / math.sqrt(E // H)
            attn = attn.masked_fill(self.mask[..., :L, :L], float("-inf"))
            attn = F.softmax(attn, dim=-1)
            # (N, H, L, L) @ (N, H, L, E / H) -> (N, H, L, E / H)
            z = attn @ v
        else:
            # Fast implementation based on fused kernel
            z = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0, is_causal=True)

        # reformating: (N, H, L, E / H) -> (N, L, H, E / H) -> (N, L, E)
        z = z.transpose(1, 2).contiguous().view(N, L, E)

        # output layer: (N, L, E) @ (E, E) -> (N, L, E)
        z = F.dropout(self.output(z), p=self.dropout, training=self.training)
        if verbose:
            return z, attn
        return z


# --------------------------------------------------------------------------------
# Feed-forward Layers
# --------------------------------------------------------------------------------


class FeedForward(nn.Module):
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


# --------------------------------------------------------------------------------
# Transformer Block
# --------------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """
    Transformer block.

    Parameters
    ----------
    config: configuration class with
        emb_dim: int
            embedding dimension of the input
        pre_norm: bool
            whether to apply layer normalization before attention and feedforward layer
        norm: str
            type of normalization layer. Options are "layer", "rms".
        norm_bias: bool
            whether to use bias in the layer normalization
        norm_eps: float
            epsilon parameter for layer normalization
        and the parameter to initialize Attention and FeedForward layers

    See Also
    --------
    Attention
    FeedForward
    """

    def __init__(self, config):
        super().__init__()
        match config.norm.lower():
            case "layer":
                NormLayer = LayerNorm
            case "rms":
                NormLayer = RMSNorm
            case _:
                raise ValueError(f"Unknown normalization layer '{config.norm}'")

        self.norm_1 = NormLayer(config.emb_dim, bias=config.norm_bias, eps=config.norm_eps)
        self.attn = SelfAttention(config)
        self.norm_2 = NormLayer(config.emb_dim, bias=config.norm_bias, eps=config.norm_eps)
        self.ffn = FeedForward(config)
        self.pre_norm = config.pre_norm

    def forward(self, x, verbose=False):
        if self.pre_norm:
            x = self.norm_1(x)
            z = self.attn(x, verbose=verbose)
            if verbose:
                z, att = z
            out = x + z
            out = out + self.ffn(self.norm_2(out))
        else:
            z = self.attn(x, verbose=verbose)
            if verbose:
                z, att = z
            z = self.norm_1(z)
            out = x + z
            out = out + self.norm_2(self.ffn(out))
        if verbose:
            return out, att
        return out


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
# Transformer Architecture
# --------------------------------------------------------------------------------


class Transformer(nn.Module):
    """
    Decoder only transformer.

    Parameters
    ----------
    config: configuration class with
        n_layer: int
            number of transformer blocks
        weight_tying: bool
            whether to use weight tying between the token embedding and the output layer
        output_dropout: float
            dropout probability for the embeddings layer
        norm: str
            type of normalization layer. Options are "layer", "rms"
        norm_bias: bool
            whether to use bias in the normalization layer
        and the parameter to initialize TransformerBlock and Embedding

    See Also
    --------
    Embedding
    TransformerBlock
    """

    def __init__(self, config):
        super().__init__()

        self.embeddings = Embedding(config)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        match config.norm.lower():
            case "layer":
                NormLayer = LayerNorm
            case "rms":
                NormLayer = RMSNorm
            case _:
                raise ValueError(f"Unknown normalization layer '{config.norm}'")
        self.output_norm = NormLayer(config.emb_dim, bias=config.norm_bias, eps=config.norm_eps)

        self.output = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

        if config.weight_tying:
            # Tying token embedding and un-embedding
            self.output.weight = self.embeddings.token_emb.weight

        self.dropout = config.output_dropout

    def forward(self, x, verbose=False):
        out = self.embeddings(x)
        attentions = []
        for block in self.blocks:
            out = block(out, verbose=verbose)
            if verbose:
                out, att = out
                attentions.append(att)
        out = self.output_norm(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.output(out)
        if verbose:
            attentions = torch.stack(attentions)
            return out, attentions
        return out


# --------------------------------------------------------------------------------
# Configuration Class
# --------------------------------------------------------------------------------


@dataclass
class TransformerConfig:
    # Embedding parameters
    vocab_size: int = -1
    emb_dim: int = -1
    pos_emb: bool = True
    pos_dim: int = None
    freeze_pos: bool = False
    seq_len: int = -1
    emb_dropout: float = None

    # Attention parameters
    n_head: int = -1
    attn_bias: bool = False
    attn_dropout: float = None
    attn_downsampling: int = 1

    # Feed-forward parameters
    activation: float = "gelu"
    ffn_dim: bool = None
    ffn_bias: bool = False
    ffn_dropout: float = None

    # Transformer block parameter
    norm: str = "layer"
    norm_bias: bool = False
    norm_eps: float = 1e-5
    pre_norm: bool = True

    # Transformer parameters
    n_layer: int = -1
    flash: bool = None
    weight_tying: bool = False
    output_dropout: float = None
    dropout: float = 0.0

    def __post_init__(self):
        # position embedding dimension
        if self.pos_dim is None:
            self.pos_dim = self.emb_dim

        # hidden feed-forward dimension
        if self.ffn_dim is None:
            self.ffn_dim = 4 * self.emb_dim

        # flash attention in PyTorch
        if self.flash is None:
            self.flash = torch.__version__ >= "2"

        # single dropout parameter
        if self.emb_dropout is None:
            self.emb_dropout = self.dropout
        if self.attn_dropout is None:
            self.attn_dropout = self.dropout
        if self.ffn_dropout is None:
            self.ffn_dropout = self.dropout
        if self.output_dropout is None:
            self.output_dropout = self.dropout
