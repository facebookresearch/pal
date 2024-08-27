import torch


def attn1_score(attn1):
    """
    Measure whether the first attention is focused on the sub-diagonal.

    If the induction head mechanism occurs, attn1_score should be high.

    Parameters
    ----------
    att1: torch.Tensor of size (bsz, N, N)
        First attention matrix

    """
    return torch.diagonal(attn1, offset=-1, dim1=1, dim2=2).mean()


def attn2_score(attn2, X):
    """
    Measure if second attention metric is focused on the token that
    follows the token we are processing.

    If the induction head mechanism occurs, attn2_score should be high.

    Parameters
    ----------
    att2: torch.Tensor of size (bsz, N, N)
        Second attention matrix
    X: torch.Tensor of size (bsz, N)
        Input data
    """
    mean = 0
    seq_len = X.size(1)
    for t in range(seq_len - 1):
        key = X[:, t + 1]
        mean += attn2[:, t][X == key[:, None]].mean()
    mean /= seq_len - 1
    return mean


def attn2_score_bis(attn2, X):
    keys = X[:, 1:]
    mask = X[:, None, :] == keys[:, :, None]
    return attn2[:, :-1][mask].mean()
