"""
Script to launch scaling laws experiments

License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

import argparse
import json
import logging
import sys
from math import sqrt
from pathlib import Path
from random import seed

import torch as th
import torch.nn.functional as F
from utils import AssMem, AssMemExp, AssMemLearnable

# Reproducibility and Logging

seed(0)
logger = logging.getLogger("script")
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="{asctime} {levelname} [{filename}:{lineno}] {message}",
    style="{",
    datefmt="%H:%M:%S",
    level="INFO",
    handlers=[
        # Log to stderr, which is catched by SLURM into log files
        logging.StreamHandler(),
    ],
)


def run_exp(config):
    assert config.ntrain % config.batch_size == 0
    niter = config.ntrain // config.batch_size

    res = {
        "model": config.model,
        "optimizer": config.optimizer,
        "lr": config.learning_rate,
        "N": config.N,
        "M": config.M,
        "d": config.d,
        "ntrain": config.ntrain,
        "batch_size": config.batch_size,
        "niter": niter,
        "layernorm": config.layernorm,
        "seed": config.seed,
    }
    logging.info(repr(res))

    th.manual_seed(config.seed)

    all_x = th.arange(config.N)
    proba = (all_x + 1.0) ** (-config.alpha)
    proba /= proba.sum()
    all_y = all_x % config.M

    # model
    if config.model == "matrix":
        model = AssMem(config.d, config.N, config.M, use_ln=config.layernorm)
    elif config.model == "learnable":
        model = AssMemLearnable(config.d, config.N, config.M, use_ln=config.layernorm)
    elif config.model == "exponential":
        model = AssMemExp(config.d, config.N, config.M, use_ln=config.layernorm)
    else:
        assert False, config.model

    # optimizer
    if config.optimizer == "SGD":
        opti = th.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "Adam":
        d = config.d
        if config.model == "learnable":
            param_groups = [
                {"params": [model.E, model.UT], "lr": config.learning_rate / sqrt(d)},
                {"params": [model.W], "lr": config.learning_rate / d},
            ]
            opti = th.optim.Adam(param_groups, betas=(0, 0), weight_decay=config.weight_decay)
        elif config.model == "exponential":
            opti = th.optim.Adam(
                model.parameters(),
                lr=config.learning_rate / sqrt(d),
                betas=(0, 0),
                weight_decay=config.weight_decay,
            )
        else:
            opti = th.optim.Adam(
                model.parameters(),
                lr=config.learning_rate / d,
                betas=(0, 0),
                weight_decay=config.weight_decay,
            )
    else:
        assert False, config.optimizer

    X = th.multinomial(proba, config.ntrain, replacement=True)
    Y = X % config.M

    logging.info(f"running {niter} iters with batch size {config.batch_size}")
    for i in range(niter):
        # print('.', end='', flush=True)
        x = X[i * config.batch_size : (i + 1) * config.batch_size]
        y = Y[i * config.batch_size : (i + 1) * config.batch_size]

        out = model(x)
        loss = F.cross_entropy(out, y)
        opti.zero_grad()
        loss.backward()
        opti.step()
    # print('', flush=True)

    with th.no_grad():
        pred = model(all_x).argmax(dim=-1)
        error = proba[pred != all_y].sum().item()
        error_head = (pred != all_y)[:30].float().mean().item()
        error_tail = (pred != all_y)[-30:].float().mean().item()

    res["error"] = error
    res["error_head"] = error_head
    res["error_tail"] = error_tail
    return res


if __name__ == "__main__":
    # Configuration

    parser = argparse.ArgumentParser(
        description="Training Memory Model",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["exponential", "learnable", "matrix"],
        default="matrix",
        help="learn the matrix W (matrix), the embeddings (exponential, which add a ReLU layer), or both",
    )
    poptim = parser.add_argument_group("Optimizer")
    poptim.add_argument(
        "--optimizer",
        type=str,
        choices=["Adam", "SGD"],
        default="Adam",
        help="optimizer",
    )
    poptim.add_argument(
        "-bs",
        "--batch-size",
        default=1024,
        type=int,
        help="batch size",
    )
    poptim.add_argument(
        "-ntr",
        "--ntrain",
        default=1024 * 10,
        type=int,
        help="number of training tokens",
    )
    poptim.add_argument(
        "-lr",
        "--learning-rate",
        default=1,
        type=float,
        help="learning rate",
    )
    poptim.add_argument(
        "-wd",
        "--weight-decay",
        default=0,
        type=float,
        help="weight decay",
    )
    pset = parser.add_argument_group("Settings")
    pset.add_argument(
        "-N",
        default=100,
        type=int,
        help="input vocabulary size",
    )
    pset.add_argument(
        "-a",
        "--alpha",
        default=2,
        type=float,
        help="Zipf law parameter",
    )
    pset.add_argument(
        "-M",
        default=5,
        type=int,
        help="output vocabulary size",
    )
    pset.add_argument(
        "-d",
        "--d",
        default=10,
        type=int,
        help="dimension",
    )
    pset.add_argument(
        "--seed",
        default=42,
        type=int,
        help="seed",
    )
    pset.add_argument(
        "--layernorm",
        action="store_true",
    )
    pset.add_argument(
        "--interactive",
        action="store_true",
    )
    pset.add_argument(
        "--root_dir",
        default="/mnt/home/abietti/ceph/assoc/res",
        help="root dir",
    )
    pset.add_argument(
        "--name",
        default="test",
        help="experiment name",
    )
    pset.add_argument(
        "--num_tasks",
        default=100,
        type=int,
        help="number of tasks",
    )
    pset.add_argument(
        "--task_id",
        default=1,
        type=int,
        help="task id, from 1 to num_tasks",
    )
    pset.add_argument(
        "--task_offset",
        default=0,
        type=int,
        help="offset for task filenames",
    )

    config = parser.parse_args()

    if config.interactive:
        res = run_exp(config)
        print(res)
        sys.exit(0)

    outdir = Path(config.root_dir) / config.name
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = open(outdir / f"{config.name}_{config.task_offset + config.task_id}.jsonl", "w")

    grid = {
        "d": [10, 20, 50, 100, 200, 500, 1000],
        "seed": list(range(42, 52)),
        # 'seed': list(range(50, 52)),
        "learning_rate": [0.1, 1.0, 10.0, 100.0],
        "batch_size": [16, 64, 256, 1024],
        "ntrain": [1024, 1024 * 10, 1024 * 100],
        "layernorm": [False, True],
        "optimizer": ["SGD", "Adam"],
        "model": ["matrix", "exponential", "learnable"],
        # 'model': ['exponential', 'learnable'],
    }

    from itertools import product

    for i, vals in enumerate(product(*grid.values())):
        if i % config.num_tasks != (config.task_id - 1):
            continue
        for k, v in zip(grid.keys(), vals):
            setattr(config, k, v)
        res = run_exp(config)
        print(json.dumps(res), file=outfile, flush=True)
