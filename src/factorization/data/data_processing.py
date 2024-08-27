"""
Generate synthetic data to study LLM behaviors in controlled settings.

License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

import logging

import numpy as np
import torch
from torch.utils.data import Dataset

from cot.config import DATA_DIR, RNG, TOKEN_DICT

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Generic class
# -----------------------------------------------------------------------------


class SequenceDataset(Dataset):
    """
    Attributes
    ----------
    data: tensor of size (n_data, seq_len)
        Tensor with data ordered by sequence length.
    indices: tensor of int of size (len)
        Indices to delimitate difference sequence lengths.

    Parameters
    ----------
    save_dir: str
        Path of the directory where to save the data.
    """

    prefix = None

    def __init__(self, save_dir=None, cot=True):
        self.cot = cot
        if not self.cot:
            assert self.prefix is not None
            self.prefix = self.prefix + "-no-cot"

        if save_dir is None:
            save_dir = DATA_DIR
            if self.prefix is not None:
                save_dir = DATA_DIR / self.prefix
        self.save_dir = save_dir

        tag = {True: " ", False: "not "}[self.cot]
        logger.info(f"Problem will {tag}use CoT.")

    @classmethod
    def get_len(cls, seq_len):
        """Full sequence length."""
        return 2 * seq_len + 2

    @classmethod
    def generate_fixed_len_data(cls, seq_len, n_data, rng=None):
        """Generate sequence with fixed sequence length."""
        raise NotImplementedError

    def generate_datafiles(self, n_data_per_len, split_probas_by_len, rng=None):
        """
        Test/train split.

        Parameters
        ----------
        n_data_per_len : int, or list of int
            Maximum number of data points to generate for each sequence length.
        split_probas_by_len : float, or list of float
            Proportion of data to put in the training set for each sequence length.
        rng : numpy.random.Generator, optional
            Random number generator. If None, use the default generator.
        """

        logger.info(f"Generating data. Saving in {self.save_dir}")

        if rng is None:
            rng = np.random.default_rng()

        assert isinstance(n_data_per_len, list) or isinstance(split_probas_by_len, list)

        if not isinstance(n_data_per_len, list):
            n_data_per_len = [n_data_per_len for _ in split_probas_by_len]

        if not isinstance(split_probas_by_len, list):
            split_probas_by_len = [split_probas_by_len for _ in n_data_per_len]

        self.save_dir.mkdir(parents=True, exist_ok=True)
        for seq_len, (n_data, split_proba) in enumerate(zip(n_data_per_len, split_probas_by_len)):
            if n_data == 0:
                continue

            seq_len += 1
            data = self.generate_fixed_len_data(seq_len=seq_len, n_data=n_data, rng=rng)

            # without chain of thoughtsa
            if not self.cot:
                data[:, seq_len + 2] = data[:, -1]
                data[:, seq_len + 3 :] = TOKEN_DICT["EoS"]

            # random ordering and splitting
            rng.shuffle(data)
            n_train = int(split_proba * len(data))
            np.save(self.save_dir / f"train_{seq_len}.npy", data[:n_train])
            np.save(self.save_dir / f"test_{seq_len}.npy", data[n_train:])
            logger.debug(f"Sequences of length {seq_len} done. Saved in {self.save_dir} ({n_train}/{len(data)} split).")

    def load_data(self, lengths, data_type=None):
        """
        Get data (load from data directory).

        Parameters
        ----------
        lengths : list of int
            List of sequence lengths.
        data_type : str, optional
            Type of data to load. Whether 'train' or 'test'.

        Returns
        -------
        data : numpy.ndarray
            Data containing sequence of tokens specified by TOKEN_DICT.
        indices : numpy.ndarray
            Indices to split the data by sequence length.

        Notes
        -----
        Should be called after `generate_datafiles`.
        """
        assert isinstance(lengths, list), "`lenghts` must be an a list of int."
        assert data_type in ["train", "test"], "`data_type` must be 'train' or 'test'."

        logging.info(f"Loading data from {self.save_dir}.")

        # memory preallocation
        # ... compute the data size by lenghts
        n_data_by_lens = np.empty(len(lengths))
        for i, seq_len in enumerate(lengths):
            filename = self.save_dir / f"{data_type}_{seq_len}.npy"
            with open(filename, "rb") as f:
                version = np.lib.format.read_magic(f)
                header = np.lib.format._read_array_header(f, version)
            n_data_by_lens[i] = header[0][0]

        # ... deduce memory allocation
        indices = np.cumsum(n_data_by_lens, dtype=int)
        indices = np.insert(indices, 0, 0)
        data = np.full((indices[-1], self.get_len(max(lengths)) + 1), TOKEN_DICT["EoS"], dtype=np.int32)

        # load the data in the allocated memory
        for i, seq_len in enumerate(lengths):
            data[indices[i] : indices[i + 1], : self.get_len(seq_len)] = np.load(
                self.save_dir / f"{data_type}_{seq_len}.npy"
            )

        return data

    def set_data(self, lengths, data_type):
        """
        Load training data as a class attribute.

        Endows `self` with attributes `data` and `indices`.

        Parameters
        ----------
        lengths : list of int
            List of sequence lengths.
        data_type : str
            Type of data to load. Whether 'train' or 'test'.

        Notes
        -----
        Should be called after `generate_datafiles`.
        """
        data = self.load_data(lengths, data_type=data_type)
        self.data = torch.from_numpy(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# -----------------------------------------------------------------------------
# Copy problem
# -----------------------------------------------------------------------------


class BinaryCopy(SequenceDataset):
    prefix = "binary_copy"

    def __init__(self, save_dir=None, cot=True):
        super().__init__(save_dir=save_dir, cot=cot)

    @classmethod
    def generate_fixed_len_data(cls, seq_len, n_data, rng=None):
        """
        Generate copy data with fixed sequence length.

        Parameters
        ----------
        seq_len : int
            Length of the sequence.
        n_data : int
            Number of data points to generate.
            Will be reduced to 2**seq_len if greater.
        rng : numpy.random.Generator, optional
            Random number generator. If None, use the default generator.
            Used if n_data is too small compared to all the potential sequences.

        Returns
        -------
        data: numpy.ndarray
            Generated data containing sequence of tokens specified by TOKEN_DICT.
        """
        if rng is None:
            rng = np.random.default_rng()

        # allocate memory
        length = cls.get_len(seq_len)
        data = np.empty((n_data, length), dtype=np.int32)

        # input data
        data[:, 1 : seq_len + 1] = (rng.random((n_data, seq_len)) > 0.5).astype(np.int32)

        # add spectial token at begining of sentence
        if cls.prefix in TOKEN_DICT:
            data[:, 0] = TOKEN_DICT[cls.prefix]
        else:
            logger.info(f"Prefix {cls.prefix} not in TOKEN_DICT, falling back to generic BoS.")
            data[:, 0] = TOKEN_DICT["BoS"]

        # end of input
        data[:, seq_len + 1] = TOKEN_DICT["EoI"]

        # copying the data
        data[:, seq_len + 2 :] = data[:, 1 : seq_len + 1]
        return data


# -----------------------------------------------------------------------------
# Parity problem
# -----------------------------------------------------------------------------


class Parity(SequenceDataset):
    prefix = "parity"

    def __init__(self, save_dir=None, cot=True):
        super().__init__(save_dir=save_dir, cot=cot)

    @classmethod
    def generate_fixed_len_data(cls, seq_len, n_data, rng=None):
        """
        Generate parity data with fixed sequence length.

        Parameters
        ----------
        seq_len : int
            Length of the sequence.
        n_data : int
            Number of data points to generate.
            Will be reduced to 2**seq_len if greater.
        rng : numpy.random.Generator, optional
            Random number generator. If None, use the default generator.
            Used if n_data is too small compared to all the potential sequences.

        Returns
        -------
        data: numpy.ndarray
            Generated data containing sequence of tokens with
                0: begining of sentence,
                1: end of input,
                2: end of sentence,
                3: negative bit,
                4: positive bit.
        """
        if rng is None:
            rng = np.random.default_rng()

        # allocate memory
        # if 2**seq_len < n_data:
        #     n_data = 2**seq_len
        length = cls.get_len(seq_len)
        data = np.empty((n_data, length), dtype=np.int32)

        # input data
        # # ... exhaustive case
        # powers_of_two = 2 ** np.arange(seq_len)[::-1]
        # data[:, 1 : seq_len + 1] = (np.arange(n_data).reshape(-1, 1) & powers_of_two != 0).astype(np.int32)
        # ... non-exhaustive case
        data[:, 1 : seq_len + 1] = (rng.random((n_data, seq_len)) > 0.5).astype(np.int32)

        # CoT data
        data[:, seq_len + 2 :] = np.cumsum(data[:, 1 : seq_len + 1], axis=1) % 2

        # add spectial token at begining of sentence
        if cls.prefix in TOKEN_DICT:
            data[:, 0] = TOKEN_DICT[cls.prefix]
        else:
            logger.info(f"Prefix {cls.prefix} not in TOKEN_DICT, falling back to generic BoS.")
            data[:, 0] = TOKEN_DICT["BoS"]

        # end of input
        data[:, seq_len + 1] = TOKEN_DICT["EoI"]

        return data


# -----------------------------------------------------------------------------
# Polynomial Evaluation
# -----------------------------------------------------------------------------


class Polynomial(SequenceDataset):
    prefix = "polynomial"

    def __init__(self, mod=11, func=None, save_dir=None, cot=True, **kwargs):
        self.mod = mod
        if func is None:

            # polynomial iteration based on this function is not permutation invariant
            def func(x, y):
                return x * y + 1

        self.func = func
        super().__init__(save_dir=save_dir, cot=cot)

    def generate_fixed_len_data(self, seq_len, n_data, rng=None):
        """
        Generate copy data with fixed sequence length.

        Parameters
        ----------
        seq_len : int
            Length of the sequence.
        n_data : int
            Number of data points to generate.
            Will be reduced to 2**seq_len if greater.
        rng : numpy.random.Generator, optional
            Random number generator. If None, use the default generator.
            Used if n_data is too small compared to all the potential sequences.

        Returns
        -------
        data: numpy.ndarray
            Generated data containing sequence of tokens specified by TOKEN_DICT.
        """
        if rng is None:
            rng = np.random.default_rng()

        # allocate memory
        length = self.get_len(seq_len)
        data = np.empty((n_data, length), dtype=np.int32)

        # input data
        # ... non-exhaustive case
        data[:, 1 : seq_len + 1] = rng.integers(1, self.mod, size=(n_data, seq_len), dtype=np.int32)

        # add spectial token at begining of sentence
        if self.prefix in TOKEN_DICT:
            data[:, 0] = TOKEN_DICT[self.prefix]
        else:
            logger.info(f"Prefix {self.prefix} not in TOKEN_DICT, falling back to generic BoS.")
            data[:, 0] = TOKEN_DICT["BoS"]

        # end of input
        data[:, seq_len + 1] = TOKEN_DICT["EoI"]

        # compute the sum
        data[:, seq_len + 2 :] = data[:, 1 : seq_len + 1]
        for t in range(1, seq_len):
            data[:, seq_len + 2 + t] = self.func(data[:, seq_len + 2 + t - 1], data[:, seq_len + 2 + t])
            data[:, seq_len + 2 + t] %= self.mod

        return data


# -----------------------------------------------------------------------------
# Mixed data
# -----------------------------------------------------------------------------


class MixedDataset(SequenceDataset):
    prefix = "mix"

    def __init__(self, data_mix=0.5, save_dir=None, cot=True, **kwargs):
        self.data_mix = data_mix
        self.binary = BinaryCopy(cot=cot)
        self.parity = Parity(cot=cot)
        super().__init__(save_dir=save_dir)

    def generate_fixed_len_data(self, seq_len, n_data, rng=None):
        """
        Generate data with fixed sequence length.

        Parameters
        ----------
        seq_len : int
            Length of the sequence.
        n_data : int
            Number of data points to generate.
        rng : numpy.random.Generator, optional
            Random number generator.

        Returns
        -------
        data: numpy.ndarray
            Generated data containing sequence of tokens specified by TOKEN_DICT.
        """
        n_binary = int(n_data * self.data_mix)
        n_parity = n_data - n_binary

        data_binary = self.binary.generate_fixed_len_data(seq_len, n_binary, rng=rng)
        data_parity = self.parity.generate_fixed_len_data(seq_len, n_parity, rng=rng)

        data = np.vstack((data_binary, data_parity))
        return data


# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------


def data_processing(
    problem="binary-copy",
    n_len=32,
    split_probas=0.5,
    n_datas=2048,
    save_dir=None,
    cot=True,
    **kwargs,
):
    """
    Training a Transformer model on a specified problem.

    Paramters
    ---------
    problem: str
        Problem to be solved. Currently supported are "binary-copy", "parity", and "no-cot".
    n_len: int
        Maximum number of lenghts for sequences.
    split_probas: float, or list of float
        Percentage of train/test split, eventually specified by length.
    n_datas: int, or list of int
        Maximum number of data to generate for a given length.
    save_dir: str
        Path of the directory where to save the data.
    cot: bool
        Wether to use chain-of-thought.
    kwargs: keyword arguments
        Arugment specific to each problem.
    """
    match problem:
        case "binary-copy":
            problem = BinaryCopy(save_dir=save_dir, cot=cot)
        case "parity":
            problem = Parity(save_dir=save_dir, cot=cot)
        case "polynomial":
            problem = Polynomial(save_dir=save_dir, cot=cot, **kwargs)
        case "mix":
            problem = MixedDataset(save_dir=save_dir, cot=cot, **kwargs)
        case _:
            raise ValueError(f"Problem {problem} not recognized.")

    if isinstance(split_probas, float) and isinstance(n_datas, int):
        n_datas = [n_datas for _ in range(n_len)]

    problem.generate_datafiles(n_datas, split_probas, rng=RNG)


if __name__ == "__main__":
    import fire

    from cot.config import logging_datefmt, logging_format, logging_level

    logging.basicConfig(
        format=logging_format,
        datefmt=logging_datefmt,
        style="{",
        level=logging_level,
        handlers=[logging.StreamHandler()],
    )

    fire.Fire(data_processing)
