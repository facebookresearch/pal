import json
import logging

from factorization.config import SAVE_DIR

logger = logging.getLogger(__name__)


CONFIG_FILE = SAVE_DIR / "config.jsonl"


def aggregate_configs() -> None:
    """
    Aggregate all configuration files from the subdirectories of `SAVE_DIR`.
    """
    all_configs = []
    for sub_dir in SAVE_DIR.iterdir():
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

    with open(CONFIG_FILE, "w") as f:
        for config in all_configs:
            json.dump(config, f)
            f.write("\n")


def recover_config(unique_id: str = None) -> dict[str, any]:
    """
    Recover the configuration file for a given unique ID.

    Parameters
    ----------
    unique_id
        Unique identifier for the configuration file.

    Returns
    -------
    config
        Configuration dictionary.
    """
    try:
        config_file = SAVE_DIR / str(unique_id) / "config.json"
        with open(config_file, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        config = recover_config_from_aggregated(unique_id)
    return config


def recover_config_from_aggregated(unique_id: str) -> dict[str, any]:
    """
    Recover the configuration file for a given unique ID from the aggregated file.

    Parameters
    ----------
    unique_id
        Unique identifier for the configuration file.

    Returns
    -------
    config
        Configuration dictionary.
    """
    with open(CONFIG_FILE, "r") as f:
        lines = f.readlines()
    for line in lines:
        try:
            config = json.loads(line)
        except json.JSONDecodeError:
            continue
        if config["id"] == unique_id:
            break
    return config


# CLI Wrapper


if __name__ == "__main__":
    import fire

    fire.Fire(aggregate_configs)
