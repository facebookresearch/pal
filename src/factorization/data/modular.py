"""
Generate synthetic data for modular addition problem.

License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class DataArgs:
    k: int = 0
    seq_length: int = 12
    vocab_size: int = 2
    sparsity_index: int = 5
    bsz: int = 2048
    test_bsz: int = 128


class Dataset:
    def __init__(self, args: DataArgs):

        self.seq_length = args.seq_length
        self.vocab_size = args.vocab_size
        self.sparsity_index = args.sparsity_index
        self.bsz = args.bsz
        self.test_bsz = args.test_bsz

    def gen_seqs(self, seed: int):
        np.random.seed(seed=seed)
        data = np.random.rand(self.bsz, self.seq_length) // (1 / self.vocab_size)
        targets = data[:, : self.sparsity_index].sum(axis=1) % self.vocab_size
        test_data = np.random.rand(self.test_bsz, self.seq_length) // (1 / self.vocab_size)
        test_targets = data[:, : self.sparsity_index].sum(axis=1) % self.vocab_size

        return data, targets, test_data, test_targets

    def get_info(self):
        print(f"Total number of unique sequences {self.vocab_size ** self.seq_length}")
