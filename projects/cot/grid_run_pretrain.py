"""
Example of pre-training grid run.

License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

import json
import logging
import traceback
from dataclasses import asdict, dataclass
from itertools import product
from uuid import uuid4

from cot.config import CHECK_DIR, DATA_DIR
from cot.data import data_processing
from cot.train import train
from cot.utils import JsonEncoder

logger = logging.getLogger(__name__)


@dataclass
class MainConfig:
    # Problem
    problem: str = "binary-copy"
    cot: bool = True

    # Data
    data_dir: str = None
    n_len: int = 16
    split_probas: float = 0.5
    n_data_per_len: int = 1024

    # Extra data options
    data_mix: float = 0.5
    mod: int = 11

    # Model
    emb_dim: int = 128
    pos_dim: int = None
    freeze_pos: bool = False
    n_head: int = 1
    n_layer: int = 2

    # Optimization
    n_epochs: int = 1000
    sgd: bool = False
    batch_size: int = 256
    learning_rate: float = 3e-4

    # Extra optimization option
    emb_dropout: float = 0.1

    # Checkpointing
    checkpoint: bool = True
    checkpoint_freq: int = 100
    overwrite_checkpoint: bool = False
    load_checkpoint: bool = False
    check_dir: str = None

    # Evaluation
    full_eval: bool = True
    eval_freq: int = 10

    # Run id
    run_id: int = 0

    def __post_init__(self):
        unique_id = str(uuid4())
        if self.data_dir == "special":
            self.data_dir = DATA_DIR / unique_id

        if self.check_dir == "special":
            self.check_dir = CHECK_DIR / unique_id


def run_experiment(
    config,
    run_data=True,
    run_train=True,
):
    """
    Run one experiments associated with one config file

    Parameters
    ----------
    config: Config class
    """

    if run_data:
        data_processing(
            problem=config.problem,
            n_len=config.n_len,
            split_probas=config.split_probas,
            n_data_per_len=config.n_data_per_len,
            save_dir=config.data_dir,
            cot=config.cot,
            data_mix=config.data_mix,
            mod=config.mod,
        )

    if run_train:
        train(
            problem=config.problem,
            cot=config.cot,
            data_dir=config.data_dir,
            n_len=config.n_len,
            emb_dim=config.emb_dim,
            pos_dim=config.pos_dim,
            freeze_pos=config.freeze_pos,
            n_head=config.n_head,
            n_layer=config.n_layer,
            n_epochs=config.n_epochs,
            sgd=config.sgd,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            emb_dropout=config.emb_dropout,
            checkpoint=config.checkpoint,
            checkpoint_freq=config.checkpoint_freq,
            overwrite_checkpoint=config.overwrite_checkpoint,
            load_checkpoint=config.load_checkpoint,
            check_dir=config.check_dir,
            full_eval=config.full_eval,
            eval_freq=config.eval_freq,
        )


def get_unique_data(config, seeds):
    data_dirs = {}
    for seed in seeds:
        unique_id = str(uuid4())
        data_dir = DATA_DIR / unique_id
        setattr(config, "data_dir", data_dir)
        run_experiment(config, run_data=True, run_train=False)
        data_dirs[seed] = config.data_dir
    return data_dirs


def run_grid(
    num_tasks=1,
    task_id=1,
    config_filename=None,
):
    """
    Run a grid of experiments

    Parameters
    ----------
    num_tasks: int
        Number of threads to split the grid run into.
    task_id: int
        Id of the current thread.
    config_filename: str
        Where to save the configuration that generate the runs.
    """

    grid = {
        "problem": ["polynomial", "parity"],
        "run_id": range(100),
    }

    CHECK_DIR.mkdir(parents=True, exist_ok=True)

    if config_filename is None:
        config_filename = "config"

    for i, values in enumerate(product(*grid.values())):
        # Handling the grid concurrently with many tasks
        if i % num_tasks != (task_id - 1):
            continue

        config = MainConfig(
            check_dir="special",
            data_dir="special",
        )

        for k, v in zip(grid.keys(), values):
            setattr(config, k, v)

        config_dict = asdict(config)
        with open(CHECK_DIR / f"{config_filename}.jsonl", "a") as f:
            json.dump(config_dict, f, cls=JsonEncoder, indent=4)
            f.write("\n")

        logger.info(f"{config=}")

        try:
            run_experiment(config, run_data=True, run_train=True)
        except Exception as e:
            logger.warning(f"Error for configuration: {config}.")
            logger.warning(traceback.format_exc())
            logger.warning(e)
            continue


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

    fire.Fire(run_grid)
