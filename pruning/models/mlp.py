import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------
# Feed-forward Layers
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
