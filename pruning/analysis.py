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

import matplotlib.pyplot as plt
from matplotlib import rc

from factorization.config import CONFIG_FILE, IMAGE_DIR, SAVE_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()]
)

rc("font", family="serif", size=8)
usetex = not subprocess.run(["which", "pdflatex"]).returncode
usetex = False
rc("text", usetex=usetex)
if usetex:
    rc("text.latex", preamble=r"\usepackage{times}")


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

    fire.Fire(
        {
            "losses": plot_all_losses,
        }
    )

# %%
