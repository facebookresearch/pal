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

from .embeddings import Embedding
from .mlp import TransformerFeedForward
from .normalization import RMSNorm

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
        flash: bool
            whether to use flash attention (could mess up half precision computation for mistral)
    """

    def __init__(self, config):
        super().__init__()

        assert (
            config.emb_dim % config.n_head == 0
        ), "embedding dimension must be divisible by number of heads"

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

        # rotational positional encoding
        self.rope = config.rope
        if self.rope:
            self.L = L
            self.theta = config.rope_theta
            self.register_buffer(
                "rope_angles", self.get_rope_freqs(self.L, E // self.H, self.theta)
            )

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

        if self.rope:
            q = self.rope_view(q)
            k = self.rope_view(k)

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
            z = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0, is_causal=True
            )

        # reformating: (N, H, L, E / H) -> (N, L, H, E / H) -> (N, L, E)
        z = z.transpose(1, 2).contiguous().view(N, L, E)

        # output layer: (N, L, E) @ (E, E) -> (N, L, E)
        z = F.dropout(self.output(z), p=self.dropout, training=self.training)
        if verbose:
            return z, attn
        return z

    @staticmethod
    def get_rope_freqs(seq_len, fan_out, theta):
        """
        Returns the frequencies for the positional encoding.

        Parameters
        ----------
        seq_len: int
            sequence  length of the sequence
        fan_out: int
            output dimension for each token in the sequence
        theta: float
            rope angle parameter
        """
        freqs = 1.0 / (theta ** (torch.arange(0, fan_out - 1, 2) / fan_out))
        t = torch.arange(seq_len)
        out = t.unsqueeze(-1) * freqs.unsqueeze(0)
        out = torch.polar(torch.ones_like(out), out)
        return out

    def rope_view(self, qk):
        """
        Recast tensor to complex numbers and apply rotational position filter.
        """
        N, H, L, dim = qk.size()
        assert L <= self.rope_angles.size(
            0
        ), "sequence length is too long for rope attention"

        # Handling typing bad behavior (complex.half() -> complex.real().half())
        if self.rope_angles.dtype in [torch.float16, torch.float32]:
            self.rope_angles = self.get_rope_freqs(self.L, dim, self.theta).to(
                device=self.rope_angles.device
            )

        # need fixed type for torch.view_as_complex to work properly
        qk_complex = torch.view_as_complex(qk.float().reshape(N, H, L, dim // 2, 2))
        qk_rot = torch.view_as_real(qk_complex * self.rope_angles[:L]).flatten(-2)
        qk = qk_rot.type_as(qk)
        return qk


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
        and the parameter to initialize Attention and TransformerFeedForward layers

    See Also
    --------
    Attention
    TransformerFeedForward
    """

    def __init__(self, config):
        super().__init__()

        if config.norm_layer:
            NormLayer = RMSNorm
        else:
            NormLayer = nn.Identity

        self.norm_1 = NormLayer(eps=config.norm_eps)
        self.attn = SelfAttention(config)
        self.norm_2 = NormLayer(eps=config.norm_eps)
        self.ffn = TransformerFeedForward(config)
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

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )

        if config.norm_layer:
            NormLayer = RMSNorm
        else:
            NormLayer = nn.Identity
        self.output_norm = NormLayer(eps=config.norm_eps)

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
    rope: bool = False
    rope_theta: int = 10_000

    # Feed-forward parameters
    activation: float = "gelu"
    ffn_dim: bool = None
    ffn_bias: bool = False
    ffn_dropout: float = None

    # Transformer block parameter
    norm_layer: bool = True
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
