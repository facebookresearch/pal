"""
Config Management Utils.

License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

import json
import logging

from ..config import CONFIG_DIR, SAVE_DIR

logger = logging.getLogger(__name__)


def get_paths(save_ext: str = None) -> None:
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
