"""
Sparse Modular Addition Dataset and Dataloader

License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetConfig:
    vocab_size: int = 2
    seq_length: int = 12
    sparsity_index: int = 5
    nb_data: int = 2048


@dataclass
class DataloaderConfig:
    vocab_size: int = 2
    seq_length: int = 12
    sparsity_index: int = 5
    nb_data: int = 2048
    batch_size: int = 32
    mode: str = "train"

    def __post_init__(self):
        if self.mode not in ["train", "test"]:
            raise ValueError(f"mode should be either 'train' or 'test', not {self.mode}")


class SMADataset(Dataset):
    def __init__(self, config: DatasetConfig, rng: np.random.Generator = None, device: torch.device = "cpu"):
        self.len = config.seq_length
        self.p = config.vocab_size
        self.k = config.sparsity_index
        self.n = config.nb_data

        if rng is None:
            rng = np.random.default_rng()
        data = rng.integers(0, self.p, (self.n, self.len))
        targets = np.sum(data[:, : self.k], axis=1) % self.p

        self.data = torch.from_numpy(data).to(dtype=torch.long, device=device)
        self.targets = torch.from_numpy(targets).to(dtype=torch.long, device=device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def __repr__(self):
        return f"Dataset with {self.n} sequences among {self.p ** self.len} unique ones."


def SMADataloader(config: DataloaderConfig, rng: np.random.Generator = None, device: torch.device = "cpu"):
    dataset = SMADataset(config, rng=rng, device=device)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True if config.mode == "train" else False
    )
    return loader
