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
This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.

@ 2023, Vivien Cabannes
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import LayerNorm, RMSNorm

# --------------------------------------------------------------------------------
# Attention Layers
# --------------------------------------------------------------------------------


class Attention(nn.Module):
    """
    Attention layer.

    Parameters
    ----------
    attention_type: str
        type of attention, either "self" or "cross"
    config: configuration class with
        emb_dim: int
            embedding dimensionality of the input
        n_head: int
            number of attention heads (should divide emb_dim)
        causal: bool
            whether to apply a causal mask to attention
        attn_bias: bool
            whether to use bias in attention
        attn_dropout: float
            dropout probability
        attn_downsampling: int
            downsampling factor for key and value matrices
        sliding_window: int
            size of the sliding window for attention mask
        rope: bool
            whether to use relative positional encoding, in which case the following parameters are required
            seq_len: int
                maximum sequence length
            rope_theta: float
                angle parameter
        flash: bool
            whether to use flash attention (could mess up half precision computation for mistral)

    TODO
    ----
    - Implement caching behavior when processing a sentence token by token at inference time (only feed next token).
    - Check sliding window implementation.
    """

    def __init__(self, attention_type, config):
        super().__init__()

        assert (
            config.emb_dim % config.n_head == 0
        ), "embedding dimension must be divisible by number of heads"

        self.H = config.n_head
        self.D = config.attn_downsampling
        E = config.emb_dim

        # matrices
        bias = config.attn_bias
        self.query = nn.Linear(E, E, bias=bias)
        self.key = nn.Linear(E, E // self.D, bias=bias)
        self.value = nn.Linear(E, E // self.D, bias=bias)
        self.output = nn.Linear(E, E, bias=bias)

        # attention type: causal, self or cross
        self.causal = config.causal
        assert attention_type.lower() in [
            "self",
            "cross",
        ], f"attention type must be either 'self' or 'cross', not {attention_type}"
        setattr(self, "forward", getattr(self, f"{attention_type.lower()}_attention"))

        # flash attention implementation and attention mask
        self.flash = config.flash
        self.sliding_window = config.sliding_window > 0
        if self.causal and (not self.flash or self.sliding_window):
            L = config.seq_len
            mask = torch.ones(L, L)
            mask = torch.tril(mask, diagonal=0)
            if config.sliding_window:
                mask = torch.triu(mask, diagonal=-config.sliding_window + 1)
            self.register_buffer("mask", mask.view(1, 1, L, L) == 0)
        else:
            # note that one might want to use masking with cross attention
            self.mask = None

        # drop-out regularization
        self.dropout = config.attn_dropout

        # Mistral related specificities
        # rotational positional encoding
        self.rope = config.rope
        if self.rope:
            self.L = config.seq_len
            self.theta = config.rope_theta
            self.register_buffer(
                "rope_angles", self.get_rope_freqs(self.L, E // self.H, self.theta)
            )

    def attention(self, q, k, v):
        """
        Attention mechanism.

        Parameters
        ----------
        q: torch.Tensor of size (N, L, E)
            query tensor
        k: torch.Tensor of size (N, S, E)
            key tensor
        H: int
            number of attention heads
        rope: bool
            whether to use rope attention
        rope_angle: torch.Tensor
            rope frequencies tensor (in polar form)

        See Also
        --------
        `Flash Attention <https://arxiv.org/abs/2205.14135>`_.
        """
        N, L, E = q.size()
        S = k.size(1)
        H, D, dim = self.H, self.D, E // self.H

        # reformating: (N, L, E)     -> (N, L, H, E / H)     -> (N, H, L, E / H)
        q = q.view(N, L, H, dim).transpose(1, 2)
        #              (N, S, E / D) -> (N, S, H / D, E / H) -> (N, H / D, S, E / H)
        k = k.view(N, S, H // D, dim).transpose(1, 2)
        v = v.view(N, S, H // D, dim).transpose(1, 2)

        if self.rope:
            q = self.rope_view(q)
            k = self.rope_view(k)

        # (N, H / D, S, E / H) -> (N, H, S, E / H)
        k = torch.repeat_interleave(k, D, dim=1)
        v = torch.repeat_interleave(v, D, dim=1)

        if not self.flash:
            # classical implementation
            # (N, H, L, E / H) @ (N, H, E / H, L) -> (N, H, L, L)
            attn = q @ k.transpose(-1, -2) / math.sqrt(E // H)
            if self.causal:
                attn = attn.masked_fill(self.mask[..., :L, :L], float("-inf"))
            attn = F.softmax(attn, dim=-1)
            # (N, H, L, S) @ (N, H, S, E / H) -> (N, H, L, E / H)
            z = attn @ v
        else:
            # Fast implementation based on fused kernel
            z = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=self.mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=self.causal,
            )

        # reformating: (N, H, L, E / H) -> (N, L, H, E / H) -> (N, L, E)
        z = z.transpose(1, 2).contiguous().view(N, L, E)
        return z

    def cross_attention(self, x, y):
        """
        Causal attention between an encoding `y` and a decoding `x`

        Parameters
        ----------
        x: torch.Tensor (N, L, E)
            working decoding sequence
        y: torch.Tensor (N, S, E)
            fixed encoding sequence
        """
        # Query, key, value: (N, L, E) @ (E, dim) -> (N, L, dim)
        q = self.query(x)
        k = self.key(y)
        v = self.value(y)

        # attention layer: (N, L, E)
        z = self.attention(q, k, v)

        # output layer: (N, L, E) @ (E, E) -> (N, L, E)
        z = F.dropout(self.output(z), p=self.dropout, training=self.training)
        return z

    def self_attention(self, x):
        """
        Self attention

        Parameters
        ----------
        x: torch.Tensor (N, L, E)
            input sequence
        """
        return self.cross_attention(x, x)

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
        N, H, LS, dim = qk.size()
        assert LS <= self.rope_angles.size(
            0
        ), "sequence length is too long for rope attention"

        # Handling typing bad behavior (complex.half() -> complex.real().half())
        if self.rope_angles.dtype in [torch.float16, torch.float32]:
            self.rope_angles = self.get_rope_freqs(self.L, dim, self.theta).to(
                device=self.rope_angles.device
            )

        # need fixed type for torch.view_as_complex to work properly
        qk_complex = torch.view_as_complex(qk.float().reshape(N, H, LS, dim // 2, 2))
        qk_rot = torch.view_as_real(qk_complex * self.rope_angles[:LS]).flatten(-2)
        qk = qk_rot.type_as(qk)
        return qk


class SelfAttention(Attention):
    """
    Self-Attention Layer.

    Notes
    -----
    See Attention layer for detailed docstring.
    """

    def __init__(self, config):
        Attention.__init__(self, "self", config)


class CrossAttention(Attention):
    """
    Cross-Attention Layer.

    Notes
    -----
    See Attention layer for detailed docstring.
    """

    def __init__(self, config):
        Attention.__init__(self, "cross", config)


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
            activation function. Options are "relu", "gelu", "swiglu".
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
        if activation == "swiglu":
            self.swiglu = True
            self.swiglu_mat = nn.Linear(
                config.emb_dim, config.ffn_dim, bias=config.ffn_bias
            )
        else:
            self.swiglu = False
            self.activation = getattr(F, activation, None)
            if self.activation is None:
                raise ValueError(f"Unknown activation function '{config.activation}'")

    def forward(self, x):
        if self.swiglu:
            out = self.fc2(F.silu(self.fc1(x)) * self.swiglu_mat(x))
        else:
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

        self.norm_1 = NormLayer(
            config.emb_dim, bias=config.norm_bias, eps=config.norm_eps
        )
        self.attn = SelfAttention(config)
        self.norm_2 = NormLayer(
            config.emb_dim, bias=config.norm_bias, eps=config.norm_eps
        )
        self.ffn = FeedForward(config)
        self.pre_norm = config.pre_norm

    def forward(self, x):
        if self.pre_norm:
            out = x + self.attn(self.norm_1(x))
            out = out + self.ffn(self.norm_2(out))
        else:
            out = x + self.norm_1(self.attn(x))
            out = out + self.norm_2(self.ffn(out))
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
            self.pos_emb = nn.Embedding(self.L, config.emb_dim)
            self.register_buffer("position", torch.arange(self.L))
        else:
            self.pos_emb = None

        # dropout regularization
        self.dropout = config.emb_dropout

    def forward(self, x):
        out = self.token_emb(x)
        if self.pos_emb is not None:
            L = x.size(1)
            assert (
                L <= self.L
            ), f"Input sequence length {L} is longer than the maximum sequence length {self.L}"
            out = out + self.pos_emb(self.position[:L])
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


# --------------------------------------------------------------------------------
# Transformer Architecture
# --------------------------------------------------------------------------------


class CausalTransformer(nn.Module):
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
        match config.norm.lower():
            case "layer":
                NormLayer = LayerNorm
            case "rms":
                NormLayer = RMSNorm
            case _:
                raise ValueError(f"Unknown normalization layer '{config.norm}'")
        self.output_norm = NormLayer(
            config.emb_dim, bias=config.norm_bias, eps=config.norm_eps
        )

        self.output = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

        if config.weight_tying:
            # Tying token embedding and un-embedding
            self.output.weight = self.embeddings.token_emb.weight

        self.dropout = config.output_dropout

    def forward(self, x):
        out = self.embeddings(x)
        for block in self.blocks:
            out = block(out)
        out = self.output_norm(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.output(out)
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
    seq_len: int = -1
    emb_dropout: float = None

    # Attention parameters
    n_head: int = -1
    causal: bool = True
    attn_bias: bool = False
    attn_dropout: float = None
    attn_downsampling: int = 1
    rope: bool = False
    rope_theta: int = 10_000
    sliding_window: int = 0

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
