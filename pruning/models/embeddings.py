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
            self.pos_emb = nn.Embedding(self.L, self.pos_dim).requires_grad_(
                False if config.freeze_pos else True
            )
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
            out[..., : self.pos_dim] = (
                out[..., : self.pos_dim] + self.pos_emb.weight[:L]
            )
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out
