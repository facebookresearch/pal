import json
import logging

from factorization.config import CONFIG_DIR, SAVE_DIR

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
        config_file = CONFIG_DIR / "base.jsonl"
    else:
        save_dir = SAVE_DIR / save_ext
        config_file = CONFIG_DIR / f"{save_ext}.jsonl"
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

    CONFIG_DIR.mkdir(exist_ok=True, parents=True)
    logging.info(f"Saving config in {agg_config_file}")
    with open(agg_config_file, "w") as f:
        for config in all_configs:
            json.dump(config, f)
            f.write("\n")


def recover_config(unique_id: str = None, save_ext: str = None) -> dict[str, any]:
    """
    Recover the configuration file for a given unique ID.

    Parameters
    ----------
    unique_id
        Unique identifier for the configuration file.
    save_ext
        Experiments folder identifier.

    Returns
    -------
    config
        Configuration dictionary.
    save_ext
        Experiments folder identifier.
    """
    save_dir, _ = get_paths(save_ext)

    try:
        config_file = save_dir / str(unique_id) / "config.json"
        with open(config_file, "r") as f:
            config = json.load(f)
    except FileNotFoundError as e:
        logger.info(f"Configuration file for {unique_id} not found.")
        logger.info(e)
        config = recover_config_from_aggregated(unique_id, save_ext=save_ext)
    return config


def recover_config_from_aggregated(unique_id: str, save_ext: str = None) -> dict[str, any]:
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
    _, config_file = get_paths(save_ext)
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


# CLI Wrapper


if __name__ == "__main__":
    import fire

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()]
    )

    fire.Fire(aggregate_configs)
