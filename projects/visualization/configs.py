"""
Utils

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
import torch

from factorization.config import CONFIG_DIR, SAVE_DIR

logger = logging.getLogger(__name__)


def get_paths(save_ext: str = None, suffix: str = None) -> None:
    """
    Get used file paths.

    Parameters
    ----------
    save_ext
        Experiments folder identifier.
    suffix
        Configuration file suffix.

    Returns
    -------
    save_dir
        Experiments folder path.
    config_file
        Configuration file path.
    """
    config_dir = CONFIG_DIR / suffix if suffix is not None else CONFIG_DIR
    if save_ext is None:
        save_dir = SAVE_DIR
        config_file = config_dir / "base.jsonl"
    else:
        save_dir = SAVE_DIR / save_ext
        config_file = config_dir / f"{save_ext}.jsonl"
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
        for config in all_configs:
            json.dump(config, f)
            f.write("\n")


def recover_config(unique_id: str = None, save_ext: str = None, suffix: str = None) -> dict[str, any]:
    """
    Recover the configuration file for a given unique ID.

    Parameters
    ----------
    unique_id
        Unique identifier for the configuration file.
    save_ext
        Experiments folder identifier.
    suffix
        Configuration file suffix.

    Returns
    -------
    config
        Configuration dictionary.
    save_ext
        Experiments folder identifier.
    """
    save_dir, _ = get_paths(save_ext, suffix=suffix)

    try:
        config_file = save_dir / str(unique_id) / "config.json"
        with open(config_file, "r") as f:
            config = json.load(f)
    except FileNotFoundError as e:
        logger.info(f"Configuration file for {unique_id} not found.")
        logger.info(e)
        config = recover_config_from_aggregated(unique_id, save_ext=save_ext, suffix=suffix)
    return config


def recover_config_from_aggregated(unique_id: str, save_ext: str = None, suffix: str = None) -> dict[str, any]:
    """
    Recover the configuration file for a given unique ID from the aggregated file.

    Parameters
    ----------
    unique_id
        Unique identifier for the configuration file.
    save_ext
        Experiments folder identifier.
    suffix
        Configuration file suffix.

    Returns
    -------
    config
        Configuration dictionary.
    """
    _, config_file = get_paths(save_ext, suffix=suffix)
    with open(config_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        try:
            config = json.loads(line)
        except json.JSONDecodeError:
            continue
        if config["id"] == unique_id:
            break
    return config


# Load experimental results


def load_configs(save_ext: str = None, suffix: str = None) -> list[dict[str, any]]:
    """
    Load all configurations from the aggregated configuration file.

    Returns
    -------
    save_ext
        Experiments folder identifier.
    suffix
        Configuration file suffix.

    Returns
    -------
    all_configs
        List of all configurations.
    """
    all_configs = []
    _, config_file = get_paths(save_ext, suffix=suffix)
    with open(config_file, "r") as f:
        for line in f.readlines():
            try:
                config = json.loads(line)
                if config["mlp_lr_discount"] is None:
                    config["mlp_lr_discount"] = 1
                config["save_ext"] = save_ext
                all_configs.append(config)
            except Exception as e:
                logger.info(f"Error when loading: {line}")
                logger.info(e)
    return all_configs


def load_experimental_results(
    all_configs: list[dict[str, any]], decorators: list[str] = None, **kwargs: dict[str, any]
) -> pd.DataFrame:
    """
    Load all experimental results related to the aggregated configuration file.

    Parameters
    ----------
    all_configs
        List with all experimental configurations.
    decorators
        List of config hyperparameters to include in the DataFrame.
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
        decorators = ["batch_size", "ffn_bias", "ffn_dim", "ffn_dropout", "lr", "mlp_lr_discount", "seed", "id"]

    all_data = []
    for experience in all_configs:
        save_dir, _ = get_paths(experience["save_ext"])
        save_dir = save_dir / experience["id"]
        if not save_dir.exists():
            logger.info(f"Skipping {experience['id']}, no data found.")
            continue

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
            all_data.append(
                pd.DataFrame(
                    torch.stack(
                        [
                            np.load(save_dir / "accs.pkl", allow_pickle=True),
                            np.load(save_dir / "test_accs.pkl", allow_pickle=True),
                            np.load(save_dir / "losses.pkl", allow_pickle=True),
                            np.load(save_dir / "test_losses.pkl", allow_pickle=True),
                        ]
                    ).T,
                    columns=["acc", "test_acc", "loss", "test_loss"],
                ).assign(
                    **{key: experience[key] for key in decorators} | {"epoch": range(1, experience["nb_epochs"] + 1)}
                )
            )
        except FileNotFoundError as e:
            logger.warning(f"Error reading {experience}.")
            logger.warning(e)
            continue

    return pd.concat(all_data, ignore_index=True)


def filter_configs(save_ext: str = None, tol: float = 0.0):
    _, pos_config_file = get_paths(save_ext, suffix="success")
    _, neg_config_file = get_paths(save_ext, suffix="failure")
    pos_config_file.parent.mkdir(exist_ok=True, parents=True)
    neg_config_file.parent.mkdir(exist_ok=True, parents=True)
    with open(pos_config_file, "w") as f:
        pass
    with open(neg_config_file, "w") as f:
        pass
    all_configs = load_configs(save_ext)

    for experience in all_configs:
        save_dir, _ = get_paths(experience["save_ext"])
        save_dir = save_dir / experience["id"]
        if not save_dir.exists():
            logger.info(f"Skipping {experience['id']}, no data found.")
            continue

        try:
            accs = np.load(save_dir / "test_accs.pkl", allow_pickle=True)
            if accs[-1] >= 1 - tol:
                with open(pos_config_file, "a") as f:
                    json.dump(experience, f)
                    f.write("\n")
            else:
                with open(neg_config_file, "a") as f:
                    json.dump(experience, f)
                    f.write("\n")
        except FileNotFoundError as e:
            logger.warning(f"Error reading {experience}.")
            logger.warning(e)
            continue


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
            "filter": filter_configs,
        }
    )
