"""
I/O utils to manage experimental config files

License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

import json
import logging

import numpy as np
import pandas as pd

from factorization.config import CONFIG_DIR, SAVE_DIR
from typing import Optional

logger = logging.getLogger(__name__)


def get_paths(save_ext: Optional[str] = None) -> None:
    """
    Get used file paths.

    Parameters
    ----------
    save_ext
        Experiments folder identifier.

    Returns
    -------
    save_dir
        Experiments folder path.
    config_file
        Configuration file path.
    """
    if save_ext is None:
        save_dir = SAVE_DIR
        config_file = CONFIG_DIR / "base.json"
    else:
        save_dir = SAVE_DIR / save_ext
        config_file = CONFIG_DIR / f"{save_ext}.json"
    return save_dir, config_file


def aggregate_configs(save_ext: str = None) -> None:
    """
    Aggregate all configuration files from the subdirectories of `SAVE_DIR`.

    Parameters
    ----------
    save_ext
        Experiments folder identifier.
    """
    save_dir, agg_config_file = get_paths(save_ext)

    all_configs = []
    for sub_dir in save_dir.iterdir():
        if sub_dir.is_dir():
            config_file = sub_dir / "config.json"
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)
                all_configs.append(config)
            except Exception as e:
                logger.warning(f"Error reading configuration file {config_file}.")
                logger.warning(e)
                continue

    agg_config_file.parent.mkdir(exist_ok=True, parents=True)
    logging.info(f"Saving config in {agg_config_file}")
    with open(agg_config_file, "w") as f:
        f.write("[\n")
        for config in all_configs:
            json.dump(config, f)
            if config != all_configs[-1]:
                f.write(",\n")
        f.write("\n]")


# Load experimental results


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
    string_decorators = [
        "log_input_factors",
        "output_factors",
        "alphas",
    ]
    save_dir, _ = get_paths(config["save_ext"])
    save_dir = save_dir / config["id"]
    if not save_dir.exists():
        logger.info(f"Skipping {config['id']}, no data found.")
        return

    for key in string_decorators:
        if key in config:
            config[key] = str(config[key])

    losses = np.load(save_dir / "losses.npy")
    if len(losses.shape) == 2:
        columns = ["loss", "test_loss"]
    else:
        columns = ["loss"]
    if final:
        epochs = [len(losses)]
        losses = losses[-1:]
    else:
        epochs = range(1, len(losses) + 1)
    output = pd.DataFrame(losses, columns=columns).assign(
        **{key: config[key] for key in decorators} | {"epoch": epochs}
    )
    return output


def load_configs(save_ext: str = None) -> list[dict[str, any]]:
    """
    Load all configurations from the aggregated configuration file.

    Returns
    -------
    save_ext
        Experiments folder identifier.

    Returns
    -------
    all_configs
        List of all configurations.
    """
    all_configs = []
    _, config_file = get_paths(save_ext)
    with open(config_file, "r") as f:
        all_configs = json.load(f)
    for config in all_configs:
        config["save_ext"] = save_ext
    return all_configs


def load_experimental_results(
    all_configs: list[dict[str, any]],
    decorators: Optional[list[str]] = None,
    final: bool = False,
    **kwargs: dict[str, any],
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
            "log_input_factors",
            "output_factors",
            "data_emb_dim",
            "alphas",
            "data_split",
            "emb_dim",
            "ffn_dim",
            "nb_layers",
            "learning_rate",
            "batch_size",
            "mode",
            "seed",
            "id",
            # "input_size",
            # "output_size",
            # "data_complexity",
            # "nb_factors",
        ]

    all_data = []
    for experience in all_configs:

        # filter data according to the requested kwargs
        skip = False
        for key, values in kwargs.items():
            exp_value = experience[key]
            if isinstance(values, list):
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


# CLI Wrapper


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
        }
    )
