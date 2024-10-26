"""
I/O utils to manage experimental config files

License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

import logging
from itertools import product

import numpy as np
import pandas as pd

from factorization.io.configs import aggregate_configs, get_paths, load_configs

logger = logging.getLogger(__name__)


def load_experimental_result(config: dict[str, any], decorators: list[str] = None, final: bool = False) -> pd.DataFrame:
    """
    Load experimemt result as a DataFrame.

    Parameters
    ----------
    config
        Configuration dictionary.
    decorators
        List of config hyperparameters to include in the DataFrame.
    final:
        If True, only the last epoch is returned.

    Returns
    -------
    output
        DataFrame with the experimental results
    """
    string_decorators = ["input_factors", "output_factors", "parents"]
    save_dir, _ = get_paths(config["save_ext"])
    save_dir = save_dir / config["unique_id"]
    if not save_dir.exists():
        logger.info(f"Skipping {config['unique_id']}, no data found.")
        return

    for key in string_decorators:
        if key in config:
            config[key] = str(config[key])

    losses = np.load(save_dir / "losses.npy")
    if len(losses.shape) == 2:
        columns = ["loss", "test_loss"]
        train_entropy = -losses[-1, 0]
        test_entropy = -losses[-1, 1]
    else:
        columns = ["loss"]
        train_entropy = -losses[-1]
        test_entropy = 0
    if final:
        epochs = [len(losses) - 1]
        losses = losses[-2:-1]
    else:
        epochs = range(1, len(losses))
        losses = losses[:-1]
    output = pd.DataFrame(losses, columns=columns).assign(
        **{key: config[key] for key in decorators}
        | {"epoch": epochs, "train_entropy": train_entropy, "test_entropy": test_entropy}
    )
    return output


def load_experimental_results(
    all_configs: list[dict[str, any]], decorators: list[str] = None, final: bool = False, **kwargs: dict[str, any]
) -> pd.DataFrame:
    """
    Load all experimental results related to the aggregated configuration file.

    Parameters
    ----------
    all_configs
        List with all experimental configurations.
    decorators
        List of config hyperparameters to include in the DataFrame.
    final
        If True, only the last epoch is returned.
    kwargs
        Hyperparameters arguments to filter the data.
        If the value is a list, the data will be filtered according to the values in the list.
        Otherwise, the data will be filtered according to the exact value.

    Returns
    -------
    all_data
        DataFrame with all experimental results.
    """
    if decorators is None:
        decorators = [
            "input_factors",
            "output_factors",
            "nb_parents",
            "beta",
            "parents",
            "alphas",
            "data_split",
            "emb_dim",
            "ffn_dim",
            "nb_layers",
            "learning_rate",
            "batch_size",
            "scheduler",
            "mode",
            "seed",
            "graph_seed",
            "unique_id",
            "input_size",
            "output_size",
            "compression_complexity",
            "statistical_complexity",
        ]
    list_keyword = ["input_factors", "output_factors", "parents"]

    all_data = []
    for experience in all_configs:

        # filter data according to the requested kwargs
        skip = False
        for key, values in kwargs.items():
            exp_value = experience[key]
            if isinstance(values, list) and key not in list_keyword:
                if exp_value not in values:
                    skip = True
                    break
            elif exp_value != values:
                skip = True
                break
        if skip:
            continue

        try:
            all_data.append(load_experimental_result(experience, decorators, final=final))
        except FileNotFoundError as e:
            logger.warning(f"Error reading {experience}.")
            logger.warning(e)
            continue

    return pd.concat(all_data, ignore_index=True)


def get_stats(res, study_factors, xaxis="epoch", **kwargs):
    """
    Get statistics for the given DataFrame.

    Parameters
    ----------
    res
        DataFrame with the experimental results.
    name
        Name of the DataFrame.
    study_factors
        List of hyperparameters to study.
    index
        Name of the parameter to use as index, default is "epoch".
    kwargs
        Information to drop from the data.
    """
    all_factors = [
        "input_factors",
        "output_factors",
        "nb_parents",
        "beta",
        "parents",
        "alphas",
        "data_split",
        "emb_dim",
        "ffn_dim",
        "nb_layers",
        "learning_rate",
        "batch_size",
        "scheduler",
        "mode",
        "seed",
        "graph_seed",
        "unique_id",
        "input_size",
        "output_size",
        "compression_complexity",
        "statistical_complexity",
    ]
    study_factors = [key for key in study_factors if key not in list(kwargs.keys())]
    ignored = ["loss", "test_loss", "train_entropy", "test_entropy"] + [
        key for key in all_factors if key not in study_factors + [xaxis]
    ]
    columns = [col for col in res.columns if col not in ignored]

    if "test_loss" not in res.columns:
        mean = res.groupby(columns)["loss"].mean().reset_index()
        std = res.groupby(columns)["loss"].std().reset_index()
    else:
        mean = res.groupby(columns)[["loss", "test_loss"]].mean().reset_index()
        std = res.groupby(columns)[["loss", "test_loss"]].std().reset_index()

    mean.set_index(xaxis, inplace=True)
    std.set_index(xaxis, inplace=True)

    all_mean = []
    all_std = []

    keys = [key for key in study_factors if key != xaxis]
    all_vals = [np.sort(res[key].unique()).tolist() for key in keys]

    for vals in product(*all_vals):
        ind = np.ones(len(mean), dtype=bool)
        for key, val in zip(keys, vals):
            ind &= mean[key] == val
        all_mean.append(mean[ind])
        all_std.append(std[ind])
    return all_mean, all_std, keys


if __name__ == "__main__":
    import fire

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    fire.Fire(
        {
            "aggregate": aggregate_configs,
            "load": load_configs,
        }
    )
