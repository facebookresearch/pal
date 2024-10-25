"""
Launcher scripts.

License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

import json
import logging
import traceback
from itertools import product

logger = logging.getLogger(__name__)


def run_grid(
    run_from_config,
    config_class,
    grid: dict[str, list[any]],
    num_tasks: int = 1,
    task_id: int = 1,
    job_id: int = 1,
    save_weight: bool = False,
    nb_seeds: int = 1,
    nb_graph_seeds: int = None,
    **kwargs: dict[str, any],
) -> None:
    """
    Run a grid of configurations for training.

    Parameters
    ----------
    run_from_config:
        The function to run from the configuration.
    config_class:
        The configuration class to use.
    num_tasks:
        The total number of tasks to run concurrently.
    task_id:
        The ID of the current task.
    job_id:
        The ID of the current job.
    save_weight:
        Whether to save the weights.
    nb_seeds:
        The number of seeds to run.
    nb_graph_seeds:
        The number of seeds for graph randomness.
    """
    logger.info(f"Running task {task_id}/{num_tasks}.")

    grid |= {
        "seed": range(nb_seeds),
        "graph_seed": range(nb_graph_seeds) if nb_graph_seeds is not None else [None],
        "save_weights": [save_weight],
    }

    nb_configs = sum(1 for _ in product(*grid.values()))
    logger.info(f"Running {nb_configs} configurations with {num_tasks} tasks.")

    for i, values in enumerate(product(*grid.values())):
        # Handling the grid concurrently with many tasks
        if i % num_tasks != (task_id - 1):
            continue

        # setup configuration
        config_dict = dict(zip(grid.keys(), values)) | kwargs
        config_dict["interactive"] = False
        config = config_class(**config_dict)

        with open(config.save_dir / "task_id", "w") as f:
            f.write(str(task_id))
        with open(config.save_dir / "job_id", "w") as f:
            f.write(str(job_id))

        try:
            run_from_config(config)
        except Exception as e:
            logger.warning(f"Error for configuration: {config}.")
            logger.warning(traceback.format_exc())
            logger.warning(e)
            continue


# -----------------------------------------------------------------------------
# JSON interface
# -----------------------------------------------------------------------------


def run_json(
    run_from_config,
    config_class,
    file: str,
    num_tasks: int = 1,
    task_id: int = 1,
    job_id: int = 1,
    **kwargs: dict[str, any],
) -> None:
    """
    Run experiments from a JSON file.

    Parameters
    ----------
    run_from_config:
        The function to run from the configuration.
    config_class:
        The configuration class to use.
    num_tasks:
        The total number of tasks to run concurrently.
    task_id:
        The ID of the current task.
    job_id:
        The ID of the current job.
    file:
        The path to the JSONL file.
    kwargs:
        Additional arguments to override the configuration.
    """
    with open(file, "r") as f:
        all_configs = json.load(f)
    for i, config_dict in enumerate(all_configs):
        # Handling the grid concurrently with many tasks
        if i % num_tasks != (task_id - 1):
            continue
        try:
            config_dict |= kwargs
            config = config_class(**config_dict)
            with open(config.save_dir / "task_id", "w") as f:
                f.write(str(task_id))
            with open(config.save_dir / "job_id", "w") as f:
                f.write(str(job_id))
            run_from_config(config)
        except Exception as e:
            logger.warning(f"Error when loading: {config_dict}")
            logger.warning(traceback.format_exc())
            logger.warning(e)


def run_grid_json(run_from_config, config_class, file: str, **kwargs: dict[str, any]) -> None:
    """
    Run grid experiments from a JSON file.

    Parameters
    ----------
    run_from_config:
        The function to run from the configuration.
    config_class:
        The configuration class to use.
    num_tasks:
        The total number of tasks to run concurrently.
    kwargs:
        Additional arguments to pass to `run_grid`.
    """
    with open(file, "r") as f:
        all_grids = json.load(f)
    for grid in all_grids:
        try:
            run_grid(run_from_config, config_class, grid=grid, **kwargs)
        except Exception as e:
            logger.warning(f"Error when loading: {grid}")
            logger.warning(traceback.format_exc())
            logger.warning(e)
