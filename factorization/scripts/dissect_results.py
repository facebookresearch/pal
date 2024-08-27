"""
Dissect results script

License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

import json
import logging
import subprocess
from pathlib import Path

import numpy as np

from cot.config import (
    CHECK_DIR,
    SAVE_DIR,
    logging_datefmt,
    logging_format,
    logging_level,
)
from cot.evals import EvaluationIO

logger = logging.getLogger(__name__)
logging.basicConfig(
    format=logging_format,
    datefmt=logging_datefmt,
    style="{",
    level=logging_level,
    handlers=[logging.StreamHandler()],
)

exp = 1
attention_eval = True
problems = ["polynomial", "polynomial-no-cot", "parity", "parity-no-cot"]

save_dir = SAVE_DIR / f"res-cot-exp{exp}"
save_dir.mkdir(parents=True, exist_ok=True)

all_configs = []
with open(CHECK_DIR / f"exp{exp}.jsonl", "r") as f:
    json_str = f.read()
    json_objs = json_str.split("}\n")
    for json_obj in json_objs:
        if json_obj:
            try:
                all_configs.append(json.loads(json_obj + "}"))
            except Exception as e:
                logger.info(e)
                logger.info(json_obj)
                continue


X = np.arange(4, 32)
Y = np.arange(4, 64)
Z = np.arange(0, 1001, 10)

Z1, Z2, Z3, Z4 = {}, {}, {}, {}

for problem in problems:
    Z1[problem] = np.full((len(X), len(Y), len(Z)), -1, dtype=float)
    Z2[problem] = np.full((len(X), len(Y), len(Z)), -1, dtype=float)

    if attention_eval:
        Z3[problem] = np.full((len(X), len(Y), len(Z)), -1, dtype=float)
        Z4[problem] = np.full((len(X), len(Y), len(Z)), -1, dtype=float)


logger.info("Parsing results.")
for config in all_configs:
    data_dir = Path(config["data_dir"])
    problem = config["problem"]
    problem = problem + ("" if config["cot"] else "-no-cot")
    n_len = config["n_len"]
    emb_dim = config["emb_dim"]
    check_dir = Path(config["check_dir"])

    try:
        meaning, data = EvaluationIO.load_eval(check_dir / "eval.csv")
    except Exception as e:
        logger.warning(e)
        logger.warning("Problem with", problem, emb_dim, n_len)
        continue

    timestamps = data[:, 0]
    eval_dim = (data.shape[1] - 1) // 2
    nd_meaning = np.array([stri[:-6] for stri in meaning[1 : 1 + eval_dim]])

    train_evals = data[:, 1 : 1 + eval_dim]
    test_evals = data[:, 1 + eval_dim :]

    min_len = 4
    train_acc = train_evals[:, min_len - 1 : n_len].mean(axis=1)
    test_acc = test_evals[:, min_len - 1 : n_len].mean(axis=1)

    if attention_eval:
        res = np.ones((2, n_len + 1 - min_len, len(Z)), dtype=float)
        res *= -1
        for i, eval_prefix in enumerate(["attn0_peaky_thres", "attn1_peaky_thres"]):
            for j, length in enumerate(range(min_len, n_len + 1)):
                eval_name = f"{eval_prefix}_{length}"

                ind = np.argmax(nd_meaning == eval_name)
                assert ind != 0

                test_res = test_evals[:, ind]
                res[i, j, : len(test_evals)] = test_res

        res = res.mean(axis=1)

    x = np.argmax(X == n_len)
    y = np.argmax(Y == emb_dim)
    try:
        Z1[problem][x, y, : len(train_acc)] = train_acc
        Z2[problem][x, y, : len(test_acc)] = test_acc
        if attention_eval:
            Z3[problem][x, y] = res[0]
            Z4[problem][x, y] = res[1]
    except Exception as e:
        logger.error(e)
        logger.error("Problem '{problem}' does not match excepted values.")

    logger.info(f"done with {problem}, {emb_dim}, {n_len}")


logging.info("Saving results.")
for problem in problems:
    np.save(save_dir / f"train_acc_{problem}.npy", Z1[problem])
    np.save(save_dir / f"test_acc_{problem}.npy", Z2[problem])
    if attention_eval:
        np.save(save_dir / f"attn0_{problem}.npy", Z3[problem])
        np.save(save_dir / f"attn1_{problem}.npy", Z4[problem])


logger.info("Deleting checkpoints.")
for config in all_configs:
    data_dir = Path(config["data_dir"])
    check_dir = Path(config["check_dir"])

    try:
        subprocess.run(["rm", "-rf", data_dir])
    except Exception as e:
        logger.info(e)

    try:
        subprocess.run(["rm", "-rf", check_dir])
    except Exception as e:
        logger.info(e)
