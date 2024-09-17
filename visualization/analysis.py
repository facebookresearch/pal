"""
Analysis scripts

License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

# %% Imports

import json
import logging
import pickle
import subprocess
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import rc

from factorization.config import CONFIG_FILE, IMAGE_DIR, SAVE_DIR

logger = logging.getLogger(__name__)

rc("font", family="serif", size=8)
usetex = not subprocess.run(["which", "pdflatex"]).returncode
usetex = False
rc("text", usetex=usetex)
if usetex:
    rc("text.latex", preamble=r"\usepackage{times}")


# %% Utils


def load_configs() -> pd.DataFrame:
    """
    Load all configurations from the aggregated configuration file.

    Returns
    -------
    all_configs
        DataFrame with all configurations.
    """
    all_configs = pd.read_json(CONFIG_FILE, lines=True)
    ind = np.isnan(all_configs["mlp_lr_discount"])
    all_configs.loc[ind, "mlp_lr_discount"] = 1
    return all_configs


def load_experimental_results(
    all_configs: pd.DataFrame, decorators: list[str], **kwargs: dict[str, any]
) -> pd.DataFrame:
    """
    Load all experimental results related to the aggregated configuration file.

    Parameters
    ----------
    all_configs
        DataFrame with all experimental configurations.
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
    all_data = []
    for experience in all_configs.itertuples():
        if not Path(SAVE_DIR / experience.id).exists():
            logger.info(f"Skipping {experience.id}, no data found.")
            continue

        # filter data according to the requested kwargs
        skip = False
        for key, values in kwargs.items():
            exp_value = getattr(experience, key)
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
                            np.load(SAVE_DIR / experience.id / "accs.pkl", allow_pickle=True),
                            np.load(SAVE_DIR / experience.id / "test_accs.pkl", allow_pickle=True),
                            np.load(SAVE_DIR / experience.id / "losses.pkl", allow_pickle=True),
                            np.load(SAVE_DIR / experience.id / "test_losses.pkl", allow_pickle=True),
                        ]
                    ).T,
                    columns=["acc", "test_acc", "loss", "test_loss"],
                ).assign(
                    **{key: getattr(experience, key) for key in decorators}
                    | {"epoch": range(1, experience.nb_epochs + 1)}
                )
            )
        except FileNotFoundError as e:
            logger.warning(f"Error reading {experience}.")
            logger.warning(e)
            continue

    return pd.concat(all_data, ignore_index=True)


# Extract Run Information


def extract_all_runs_info(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and annotate runs for further analysis.

    Parameters
    ----------
    data
        DataFrame containing all experimental results.

    Returns
    -------
    out
        DataFrame extracting runs information.
    """
    columns = ["batch_size", "lr", "mlp_lr_discount", "ffn_dim", "seed"]
    out = data.groupby(columns)["test_acc"].idxmax().reset_index()
    out["argmax"] = out["test_acc"]
    out["high"] = data["test_acc"].iloc[out["argmax"]].reset_index(drop=True)
    out["argmax"] = data["epoch"].iloc[out["argmax"]].reset_index(drop=True)
    out["test_acc"] = data[data.epoch == 1000].groupby(columns)["test_acc"].mean().reset_index()["test_acc"]
    return out


def extract_run_info(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Extract average run based on filters.

    Parameters
    ----------
    data
        DataFrame containing all experimental results.
    kwargs
        Hyperparameters arguments to filter the data.

    Returns
    -------
    out
        DataFrame extracting run information.
    """
    ind = np.ones(data.shape[0], dtype=bool)
    for key, value in kwargs.items():
        if value is not None:
            ind &= data[key] == value
    exp_data = data[ind].reset_index()
    exp_data["id"] = 0
    exp_data["success"] = exp_data["test_acc"] > 0.99

    mean = exp_data.groupby(["epoch"]).mean()
    std = exp_data.groupby(["epoch"]).std()
    return mean, std


# %%


# %%


# Accuracy and Loss


def plot_losses(unique_id: int, file_format: str = "pdf", title: str = None) -> None:
    """
    Plot the losses for a given unique ID.

    Parameters
    ----------
    unique_id
        Unique identifier for the configuration file.
    file_format
        File format for the image.
    title
        Title for the plot.
    """
    save_dir = SAVE_DIR / unique_id
    losses = pickle.load(open(save_dir / "losses.pkl", "rb"))
    test_losses = pickle.load(open(save_dir / "test_losses.pkl", "rb"))
    accs = pickle.load(open(save_dir / "accs.pkl", "rb"))
    test_accs = pickle.load(open(save_dir / "test_accs.pkl", "rb"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(losses, label="train")
    axes[0].plot(test_losses, label="test")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].legend()
    axes[1].plot(accs, label="train")
    axes[1].plot(test_accs, label="test")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].legend()
    axes[1].grid()

    save_dir = IMAGE_DIR / "losses"
    save_dir.mkdir(exist_ok=True, parents=True)
    fig.suptitle(title if title else f"Losses for configuration {unique_id}")
    fig.savefig(save_dir / f"{unique_id}.{file_format}", bbox_inches="tight")


def plot_all_losses(file_format: str = "pdf") -> None:
    """
    Plot the losses for all configurations in the aggregated config file.

    Parameters
    ----------
    file_format
        File format for the image.

    Nota Bene
    ---------
    In order to annotate the plots with the quantities of interest, change the `title` variable in this function.
    """
    with open(CONFIG_FILE, "r") as f:
        lines = f.readlines()

    for line in lines:
        try:
            config = json.loads(line)
        except json.JSONDecodeError:
            logger.warning(f"Error reading configuration file {line}.")
            continue
        try:
            title = f"bsz={config['batch_size']}, lr={config['lr']}, ffn_dim={config['ffn_dim']}, seed={config['seed']}"
            plot_losses(config["id"], file_format=file_format, title=title)
            logger.info(f"Losses for configuration {config['id']} plotted.")
        except Exception as e:
            logger.warning(f"Error for configuration: {config}.")
            logger.warning(traceback.format_exc())
            logger.warning(e)
            continue


# CLI Wrapper


if __name__ == "__main__":
    import fire

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()]
    )
    fire.Fire(
        {
            "losses": plot_all_losses,
        }
    )

    all_configs = load_configs()
    keys = ["batch_size", "lr", "mlp_lr_discount", "ffn_dim", "seed", "id"]
    data = load_experimental_results(all_configs, keys)

    run_info = extract_run_info(data)
# %%
