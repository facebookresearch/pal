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
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc

from configs import get_paths, load_configs, load_experimental_results
from factorization.config import IMAGE_DIR, USETEX

logger = logging.getLogger(__name__)

rc("font", family="serif", size=8)
rc("text", usetex=USETEX)
if USETEX:
    rc("text.latex", preamble=r"\usepackage{times}")


# Extract Run Information


def extract_runs_info(data: pd.DataFrame) -> pd.DataFrame:
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
    nb_epochs = data["epoch"].max()
    columns = ["batch_size", "lr", "mlp_lr_discount", "ffn_dim", "seed"]
    out = data.groupby(columns)["test_acc"].idxmax().reset_index()
    out["argmax"] = out["test_acc"]
    out["high"] = data["test_acc"].iloc[out["argmax"]].reset_index(drop=True)
    out["argmax"] = data["epoch"].iloc[out["argmax"]].reset_index(drop=True)
    out["test_acc"] = data[data.epoch == nb_epochs].groupby(columns)["test_acc"].mean().reset_index()["test_acc"]
    return out


def extract_averaged_info(data: pd.DataFrame, average_key: list[str], **kwargs) -> pd.DataFrame:
    """
    Extract average run based on filters.

    Parameters
    ----------
    data
        DataFrame containing all experimental results.
    average_key
        List of keys to condition the averaging.
    kwargs
        Hyperparameters arguments to filter the data.

    Returns
    -------
    mean
        Mean DataFrame.
    std
        Standard deviation DataFrame.
    """
    ind = np.ones(data.shape[0], dtype=bool)
    for key, value in kwargs.items():
        if value is not None:
            ind &= data[key] == value
    exp_data = data[ind]
    exp_data.loc[:, "id"] = 0

    group = exp_data.groupby(average_key)
    mean = group.mean().reset_index()
    std = group.std().reset_index()
    return mean, std


# Ablation study plot


def show_ablation(seed: bool = False, key: str = "test_acc", file_format: str = "png"):
    """
    Ablation study code, relative to `train.py`

    Parameters
    ----------
    seed
        Whether to plot the ablation study for each seed.
    key
        Key to plot. Wether `test_acc` or `success`.
    file_format
        File format for the images.
    """
    all_data = {}
    exp_ids = ["batch_size", "ffn_dim", "lr", "mlp_lr"]
    group_keys = ["batch_size", "ffn_dim", "lr", "mlp_lr_discount", "seed", "id"]

    for exp_id in exp_ids:
        logger.info(f"Loading data for {exp_id}.")
        all_configs = load_configs(exp_id)
        data = load_experimental_results(all_configs, group_keys)
        data["success"] = data["test_acc"] > 0.98
        all_data[exp_id] = data

    image_dir = IMAGE_DIR / "seed"
    image_dir.mkdir(exist_ok=True, parents=True)
    all_seeds = np.unique(all_data[exp_ids[0]]["seed"])
    nb_epochs = all_data[exp_ids[0]]["epoch"].max()

    group_keys = ["batch_size", "ffn_dim", "lr", "mlp_lr_discount"]
    if seed:
        for seed in all_seeds:
            logger.info(f"Processing seed {seed}.")
            kwargs = {"epoch": nb_epochs, "seed": seed}

            fig, axes = plt.subplots(1, len(exp_ids), figsize=(4 * len(exp_ids), 4))
            for i, exp_id in enumerate(exp_ids):
                mean, std = extract_averaged_info(all_data[exp_id], group_keys, **kwargs)
                if exp_id == "mlp_lr":
                    exp_id = "mlp_lr_discount"
                axes[i].plot(mean[exp_id], mean[key])
                axes[i].set_title(f"Ablation over {exp_id}", fontsize=12)
                axes[i].set_xscale("log")
                axes[i].grid()
            # fig.suptitle("Ablation study for a fixed seed", fontsize=12)
            fig.savefig(image_dir / f"{seed}_{key}.{file_format}", bbox_inches="tight")
    else:
        kwargs = {"epoch": nb_epochs}

        fig, axes = plt.subplots(1, len(exp_ids), figsize=(5 * len(exp_ids), 5))
        for i, exp_id in enumerate(exp_ids):
            mean, std = extract_averaged_info(all_data[exp_id], group_keys, **kwargs)
            if exp_id == "mlp_lr":
                exp_id = "mlp_lr_discount"
            axes[i].plot(mean[exp_id], mean[key])
            axes[i].fill_between(mean[exp_id], mean[key] - std[key], mean[key] + std[key], alpha=0.2)
            axes[i].set_title(f"Ablation over {exp_id}", fontsize=12)
            axes[i].set_xscale("log")
            axes[i].grid()
        # fig.suptitle("Test Accuracy (study over 100 runs)", fontsize=12)
        fig.savefig(image_dir / f"all_{key}.{file_format}", bbox_inches="tight")


# Accuracy and Loss


def plot_losses(unique_id: int, file_format: str = "pdf", title: str = None, save_ext: str = None) -> None:
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
    save_ext
        Experiments folder identifier.
    """
    save_dir, _ = get_paths(save_ext)
    save_dir = save_dir / unique_id
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

    save_ext = save_ext if save_ext is not None else "base"
    save_dir = IMAGE_DIR / save_ext
    save_dir.mkdir(exist_ok=True, parents=True)
    fig.suptitle(title if title else f"Losses for configuration {unique_id}")
    fig.savefig(save_dir / f"{unique_id}.{file_format}", bbox_inches="tight")


def plot_all_losses(file_format: str = "pdf", save_ext: str = None) -> None:
    """
    Plot the losses for all configurations in the aggregated config file.

    Parameters
    ----------
    file_format
        File format for the image.
    save_ext
        Experiments folder identifier

    Nota Bene
    ---------
    In order to annotate the plots with the quantities of interest, change the `title` variable in this function.
    """
    _, config_file = get_paths(save_ext)
    with open(config_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        try:
            config = json.loads(line)
        except json.JSONDecodeError:
            logger.warning(f"Error reading configuration file {line}.")
            continue
        try:
            title = f"bsz={config['batch_size']}, lr={config['lr']}, ffn_dim={config['ffn_dim']}, seed={config['seed']}"
            plot_losses(config["id"], file_format=file_format, title=title, save_ext=save_ext)
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
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    fire.Fire(
        {
            "ablation": show_ablation,
            "losses": plot_all_losses,
        }
    )
# %%
