import torch.nn as nn

# -------------------------------------------------------------------------------
# Normalization Layers
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
