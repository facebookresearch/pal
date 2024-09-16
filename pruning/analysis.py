# %% Imports

import json
import logging
import pickle
import subprocess
import traceback

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import rc

from factorization.config import IMAGE_DIR, SAVE_DIR
from factorization.models.softmax_model import Model, ModelConfig, RMSNorm

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

CONFIG_FILE = SAVE_DIR / "config.jsonl"


# Utils


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


# Generate visualization


def show_frame(unique_id: int, epoch: int, file_format: str = None):
    """
    Show a single frame for a given unique ID.

    Parameters
    ----------
    unique_id
        Unique identifier for the configuration file.
    epoch
        Epoch to show.
    file_format
        File format for the image.
    """
    config = recover_config(unique_id)
    assert config["save_weights"], f"Weights were not saved for ID {unique_id}."
    assert epoch < config["nb_epochs"], f"Epoch {epoch} is greater than the number of epochs {config['nb_epochs']}."
    visualization_backend(unique_id, start_frame=epoch, end_frame=None, file_format=file_format)


def generate_animation(unique_id: int, num_tasks: int = 1, task_id: int = 1):
    """
    Generate an animation for a given unique ID.

    Parameters
    ----------
    unique_id
        Unique identifier for the configuration file.
    num_tasks
        Number of tasks to divide the animation into.
    task_id
        Current task ID.
    """
    config = recover_config(unique_id)
    assert config["save_weights"], f"Weights were not saved for ID {unique_id}."

    ani_length = config["nb_epochs"]
    assert (
        ani_length % num_tasks == 0
    ), f"Number of tasks {num_tasks} does not divide the number of epochs {ani_length}."
    block_length = ani_length // num_tasks
    start_frame = (task_id - 1) * block_length
    end_frame = task_id * block_length

    visualization_backend(unique_id, start_frame, end_frame)


def visualization_backend(unique_id: int, start_frame: int, end_frame: int = None, file_format: str = None):
    """
    Backend for the visualization functions.

    If `end_frame` is `None`, the function will save a single frame.
    Otherwise, it will save a video from `start_frame` to `end_frame`.

    Parameters
    ----------
    unique_id
        Unique identifier for the configuration file.
    start_frame
        Frame to start from.
    end_frame
        Frame to end at.
    file_format
        File format for the image or video.
    """

    # configuration and saved computations

    config = recover_config(unique_id)
    vocab_size = config["vocab_size"]
    length = config["seq_length"]
    sparsity_index = config["sparsity_index"]
    ffn_dim = config["ffn_dim"]
    assert config["save_weights"], f"Weights were not saved for ID {unique_id}."

    save_dir = SAVE_DIR / unique_id
    weights = pickle.load(open(save_dir / "weights.pkl", "rb"))
    losses = pickle.load(open(save_dir / "losses.pkl", "rb"))
    test_losses = pickle.load(open(save_dir / "test_losses.pkl", "rb"))
    accs = pickle.load(open(save_dir / "accs.pkl", "rb"))
    test_accs = pickle.load(open(save_dir / "test_accs.pkl", "rb"))

    DEVICE = "cpu"

    # modules

    config = ModelConfig(
        vocab_size=config["nb_emb"],
        emb_dim=config["emb_dim"],
        seq_length=config["seq_length"],
        ffn_dim=config["ffn_dim"],
        ffn_bias=config["ffn_bias"],
    )
    model = Model(config)
    model.eval()
    model.to(DEVICE)
    norm = RMSNorm()

    # data

    prefix = [
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 1],
        [0, 1, 1, 0, 1],
        [0, 1, 1, 1, 1],
        [1, 1, 0, 1, 1],
    ]
    suffixes = [
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 1],
        [0, 1, 1, 0, 0, 1, 0],
    ]
    inputs = torch.tensor([pre + suf for pre in prefix for suf in suffixes], device=DEVICE)
    targets = inputs[:, :sparsity_index].sum(dim=1) % vocab_size
    pos_inputs = inputs[targets == 1]
    neg_inputs = inputs[targets == 0]

    tmpx = torch.linspace(-1, 1, 50)
    tmpy = torch.linspace(-1, 1, 50)
    X_mlp, Y_mlp = torch.meshgrid(tmpx, tmpy)
    grid_mlp = torch.stack([X_mlp, Y_mlp], dim=-1).to(DEVICE).view(-1, 2)

    tmpx = torch.linspace(-2.5, 2.5, 50)
    tmpy = torch.linspace(-3.5, 2.5, 50)
    X_out, Y_out = torch.meshgrid(tmpx, tmpy)
    grid_out = torch.stack([X_out, Y_out], dim=-1).to(DEVICE).view(-1, 2)

    # create Animation

    WIDTH = 20
    HEIGHT = 20

    fig, axes = plt.subplots(4, 4, figsize=(WIDTH, HEIGHT))

    text_fontsize = 8
    title_fontsize = 12

    pos_marker = "o"
    neg_marker = "s"

    def update(frame):
        for i in range(4):
            for j in range(4):
                axes[i, j].clear()

        token_emb = weights[frame]["token_emb.weight"][:vocab_size]
        pos_emb = weights[frame]["pos_emb.weight"]
        emb = (token_emb.unsqueeze(1) + pos_emb).reshape(-1, 2)
        norm_emb = norm(emb)

        query = weights[frame]["softmax.query.weight"]
        value = weights[frame]["softmax.value.weight"]
        emb_val = norm_emb @ value.T

        model.load_state_dict(weights[frame])
        with torch.no_grad():
            pos_seq_emb, attn = model.softmax(norm(model.token_emb(pos_inputs) + model.pos_emb.weight), verbose=True)
            neg_seq_emb = model.softmax(norm(model.token_emb(neg_inputs) + model.pos_emb.weight))

            norm_pos_seq = norm(pos_seq_emb)
            norm_neg_seq = norm(neg_seq_emb)
            fc1 = model.mlp.fc1.weight.detach()
            fc2 = model.mlp.fc2.weight.detach()

            pos_seq_mlp = model.mlp(norm_pos_seq)
            neg_seq_mlp = model.mlp(norm_neg_seq)

            pos_seq_res = pos_seq_emb + pos_seq_mlp
            neg_seq_res = neg_seq_emb + neg_seq_mlp

            out_mlp = F.softmax(model.output(grid_mlp + model.mlp(norm(grid_mlp))), dim=-1)[..., 1].view(X_mlp.shape)
            out_out = F.softmax(model.output(grid_out), dim=-1)[..., 1].view(X_out.shape)
            pos_seq_prob = F.softmax(model.output(pos_seq_res), dim=-1)[:, :vocab_size]
            neg_seq_prob = F.softmax(model.output(neg_seq_res), dim=-1)[:, :vocab_size]

        ind = (0, 0)
        axes[*ind].scatter(
            token_emb[:, 0],
            token_emb[:, 1],
            c=np.arange(vocab_size),
            cmap="tab20",
            s=100,
        )
        for i, (x, y) in enumerate(token_emb):
            axes[*ind].text(x, y, i, fontsize=text_fontsize)
        axes[*ind].grid()
        axes[*ind].set_title("Token Embeddings $E$", fontsize=title_fontsize)

        ind = (0, 1)
        axes[*ind].scatter(
            pos_emb[:sparsity_index, 0],
            pos_emb[:sparsity_index, 1],
            c=np.arange(sparsity_index),
            cmap="tab20",
            s=100,
            marker=pos_marker,
        )
        axes[*ind].scatter(
            pos_emb[sparsity_index:, 0],
            pos_emb[sparsity_index:, 1],
            c=np.arange(sparsity_index, length),
            cmap="tab20",
            s=100,
            marker=neg_marker,
        )
        for i, (x, y) in enumerate(pos_emb):
            axes[*ind].text(x, y, i, fontsize=text_fontsize)
        axes[*ind].grid()
        axes[*ind].set_title("Position Embeddings $P$", fontsize=title_fontsize)

        ind = (0, 2)
        axes[*ind].scatter([0], [0], c="k", marker="o", s=50)
        axes[*ind].scatter(
            emb[:sparsity_index, 0],
            emb[:sparsity_index, 1],
            c=np.arange(sparsity_index),
            cmap="tab20",
            marker=pos_marker,
            s=100,
        )
        axes[*ind].scatter(
            emb[length : length + sparsity_index, 0],
            emb[length : length + sparsity_index, 1],
            c=np.arange(sparsity_index),
            cmap="tab20",
            marker=pos_marker,
            s=100,
        )
        axes[*ind].scatter(
            emb[sparsity_index:length, 0],
            emb[sparsity_index:length, 1],
            c=np.arange(sparsity_index, length),
            cmap="tab20",
            marker=neg_marker,
            s=100,
        )
        axes[*ind].scatter(
            emb[length + sparsity_index :, 0],
            emb[length + sparsity_index :, 1],
            c=np.arange(sparsity_index, length),
            cmap="tab20",
            marker=neg_marker,
            s=100,
        )
        for i, (x, y) in enumerate(emb):
            axes[*ind].text(x, y, (i // 12, i % 12), fontsize=text_fontsize)
        axes[*ind].grid()
        axes[*ind].set_title("Embeddings $E + P$", fontsize=title_fontsize)

        ind = (0, 3)
        axes[*ind].scatter(
            norm_emb[:sparsity_index, 0],
            norm_emb[:sparsity_index, 1],
            c=np.arange(sparsity_index),
            cmap="tab20",
            marker=pos_marker,
            s=100,
        )
        axes[*ind].scatter(
            norm_emb[length : length + sparsity_index, 0],
            norm_emb[length : length + sparsity_index, 1],
            c=np.arange(sparsity_index),
            cmap="tab20",
            marker=pos_marker,
            s=100,
        )
        axes[*ind].scatter(
            norm_emb[sparsity_index:length, 0],
            norm_emb[sparsity_index:length, 1],
            c=np.arange(sparsity_index, length),
            cmap="tab20",
            marker=neg_marker,
            s=100,
        )
        axes[*ind].scatter(
            norm_emb[length + sparsity_index :, 0],
            norm_emb[length + sparsity_index :, 1],
            c=np.arange(sparsity_index, length),
            cmap="tab20",
            marker=neg_marker,
            s=100,
        )
        for i, (x, y) in enumerate(norm_emb):
            axes[*ind].text(x, y, (i // 12, i % 12), fontsize=text_fontsize)
        axes[*ind].arrow(0, 0, query[0, 0], query[0, 1], head_width=0.1, head_length=0.1, fc="r", ec="r")
        axes[*ind].text(0, 0, "query", fontsize=text_fontsize + 2, color="r")
        axes[*ind].set_title(r"Normed Embeddings $Z(x,t) \propto E(x) + P(t)$", fontsize=title_fontsize)

        ind = (1, 0)
        axes[*ind].imshow(attn, cmap="Blues", vmin=0, vmax=0.2)
        axes[*ind].plot([4.5, 4.5], [-1, len(attn)], color="C3")
        axes[*ind].set_title("Attention vectors", fontsize=title_fontsize)
        axes[*ind].axis("off")

        ind = (1, 1)
        axes[*ind].scatter(
            emb_val[:sparsity_index, 0],
            emb_val[:sparsity_index, 1],
            c=np.arange(sparsity_index),
            cmap="tab20",
            marker=pos_marker,
            s=100,
        )
        axes[*ind].scatter(
            emb_val[length : length + sparsity_index, 0],
            emb_val[length : length + sparsity_index, 1],
            c=np.arange(sparsity_index),
            cmap="tab20",
            marker=pos_marker,
            s=100,
        )
        axes[*ind].scatter(
            emb_val[sparsity_index:length, 0],
            emb_val[sparsity_index:length, 1],
            c=np.arange(sparsity_index, length),
            cmap="tab20",
            marker=neg_marker,
            s=100,
        )
        axes[*ind].scatter(
            emb_val[length + sparsity_index :, 0],
            emb_val[length + sparsity_index :, 1],
            c=np.arange(sparsity_index, length),
            cmap="tab20",
            marker=neg_marker,
            s=100,
        )
        for i, (x, y) in enumerate(emb_val):
            axes[*ind].text(x, y, (i // 12, i % 12), fontsize=text_fontsize)
        axes[*ind].grid()
        axes[*ind].set_title("Value", fontsize=title_fontsize)

        ind = (1, 2)
        axes[*ind].scatter([0], [0], c="k", marker="o", s=50)
        axes[*ind].scatter(
            pos_seq_emb[:, 0],
            pos_seq_emb[:, 1],
            c=np.arange(pos_seq_emb.shape[0]),
            marker=pos_marker,
            cmap="tab20b",
            s=100,
        )
        axes[*ind].scatter(
            neg_seq_emb[:, 0],
            neg_seq_emb[:, 1],
            c=np.arange(neg_seq_emb.shape[0]),
            marker=neg_marker,
            cmap="tab20b",
            s=100,
        )
        for i, (x, y) in enumerate(pos_seq_emb):
            t = axes[*ind].text(x, y, pos_inputs[i].numpy().tolist(), fontsize=text_fontsize)
            t.set_alpha(0.3)
        for i, (x, y) in enumerate(neg_seq_emb):
            t = axes[*ind].text(x, y, neg_inputs[i].numpy().tolist(), fontsize=text_fontsize)
            t.set_alpha(0.3)
        axes[*ind].grid()
        axes[*ind].set_title(r"Sequence Embeddings $\xi$", fontsize=title_fontsize)

        ind = (1, 3)
        axes[*ind].contourf(X_mlp, Y_mlp, out_mlp, cmap="coolwarm", vmin=0, vmax=1)
        axes[*ind].scatter(
            pos_seq_emb[:, 0],
            pos_seq_emb[:, 1],
            c=np.arange(pos_seq_emb.shape[0]),
            marker=pos_marker,
            cmap="tab20b",
            s=100,
        )
        axes[*ind].scatter(
            neg_seq_emb[:, 0],
            neg_seq_emb[:, 1],
            c=np.arange(neg_seq_emb.shape[0]),
            marker=neg_marker,
            cmap="tab20b",
            s=100,
        )
        for i, (x, y) in enumerate(pos_seq_emb):
            t = axes[*ind].text(x, y, pos_inputs[i].numpy().tolist(), fontsize=text_fontsize)
            t.set_alpha(0.3)
        for i, (x, y) in enumerate(neg_seq_emb):
            t = axes[*ind].text(x, y, neg_inputs[i].numpy().tolist(), fontsize=text_fontsize)
            t.set_alpha(0.3)
        axes[*ind].set_title(r"Transform level lines: $\xi \to p(y=1|\xi)$", fontsize=title_fontsize)

        ind = (2, 0)
        axes[*ind].scatter(
            norm_pos_seq[:, 0],
            norm_pos_seq[:, 1],
            c=np.arange(norm_pos_seq.shape[0]),
            marker=pos_marker,
            cmap="tab20b",
            s=100,
        )
        axes[*ind].scatter(
            norm_neg_seq[:, 0],
            norm_neg_seq[:, 1],
            c=np.arange(norm_neg_seq.shape[0]),
            marker=neg_marker,
            cmap="tab20b",
            s=100,
        )
        for i, (x, y) in enumerate(norm_pos_seq):
            t = axes[*ind].text(x, y, pos_inputs[i].numpy().tolist(), fontsize=text_fontsize)
            t.set_alpha(0.3)
        for i, (x, y) in enumerate(norm_neg_seq):
            t = axes[*ind].text(x, y, neg_inputs[i].numpy().tolist(), fontsize=text_fontsize)
            t.set_alpha(0.3)
        axes[*ind].set_title(r"Normed Input: $\xi / \|\xi\|$", fontsize=title_fontsize)

        ind = (2, 1)
        axes[*ind].scatter([0], [0], c="k", marker="o", s=50)
        axes[*ind].scatter(
            fc1[:, 0],
            fc1[:, 1],
            c=np.arange(ffn_dim),
            cmap="tab20",
            marker="^",
            s=100,
        )
        for i in range(ffn_dim):
            axes[*ind].plot([0, fc1[i, 0]], [0, fc1[i, 1]], alpha=0.2)
        axes[*ind].scatter(
            norm_pos_seq[:, 0],
            norm_pos_seq[:, 1],
            c=np.arange(norm_pos_seq.shape[0]),
            marker=pos_marker,
            cmap="tab20b",
            s=100,
            alpha=0.1,
        )
        axes[*ind].scatter(
            norm_neg_seq[:, 0],
            norm_neg_seq[:, 1],
            c=np.arange(norm_neg_seq.shape[0]),
            marker=neg_marker,
            cmap="tab20b",
            s=100,
            alpha=0.1,
        )
        axes[*ind].grid()
        axes[*ind].set_title("MLP receptors", fontsize=title_fontsize)

        ind = (2, 2)
        axes[*ind].scatter([0], [0], c="k", marker="o", s=50)
        axes[*ind].scatter(
            fc2[0],
            fc2[1],
            c=np.arange(ffn_dim),
            cmap="tab20",
            marker="v",
            s=100,
        )
        axes[*ind].scatter(
            fc1[:, 0],
            fc1[:, 1],
            c=np.arange(ffn_dim),
            cmap="tab20",
            marker="^",
            s=100,
            alpha=0.2,
        )
        for i in range(ffn_dim):
            axes[*ind].plot([0, fc1[i, 0]], [0, fc1[i, 1]], alpha=0.1)
        axes[*ind].scatter(
            norm_pos_seq[:, 0],
            norm_pos_seq[:, 1],
            c=np.arange(norm_pos_seq.shape[0]),
            marker=pos_marker,
            cmap="tab20b",
            s=100,
            alpha=0.1,
        )
        axes[*ind].scatter(
            norm_neg_seq[:, 0],
            norm_neg_seq[:, 1],
            c=np.arange(norm_neg_seq.shape[0]),
            marker=neg_marker,
            cmap="tab20b",
            s=100,
            alpha=0.1,
        )
        axes[*ind].grid()
        axes[*ind].set_title("MLP assemblers", fontsize=title_fontsize)

        ind = (2, 3)
        axes[*ind].scatter(
            pos_seq_mlp[:, 0],
            pos_seq_mlp[:, 1],
            c=np.arange(pos_seq_mlp.shape[0]),
            marker=pos_marker,
            cmap="tab20b",
            s=100,
        )
        axes[*ind].scatter(
            neg_seq_mlp[:, 0],
            neg_seq_mlp[:, 1],
            c=np.arange(neg_seq_mlp.shape[0]),
            marker=neg_marker,
            cmap="tab20b",
            s=100,
        )
        for i, (x, y) in enumerate(pos_seq_mlp):
            t = axes[*ind].text(x, y, pos_inputs[i].numpy().tolist(), fontsize=text_fontsize)
            t.set_alpha(0.3)
        for i, (x, y) in enumerate(neg_seq_mlp):
            t = axes[*ind].text(x, y, neg_inputs[i].numpy().tolist(), fontsize=text_fontsize)
            t.set_alpha(0.3)

        axes[*ind].scatter(
            pos_seq_res[:, 0],
            pos_seq_res[:, 1],
            c=np.arange(pos_seq_res.shape[0]),
            marker=pos_marker,
            cmap="tab20b",
            s=100,
            alpha=0.2,
        )
        axes[*ind].scatter(
            neg_seq_res[:, 0],
            neg_seq_res[:, 1],
            c=np.arange(neg_seq_res.shape[0]),
            marker=neg_marker,
            cmap="tab20b",
            s=100,
            alpha=0.2,
        )
        # for i, (x, y) in enumerate(pos_seq_res):
        #     t = axes[*ind].text(x, y, pos_inputs[i].numpy().tolist(), fontsize=text_fontsize)
        #     t.set_alpha(0.3)
        # for i, (x, y) in enumerate(neg_seq_res):
        #     t = axes[*ind].text(x, y, neg_inputs[i].numpy().tolist(), fontsize=text_fontsize)
        #     t.set_alpha(0.3)
        axes[*ind].set_title("Transformed Sequences with residual", fontsize=title_fontsize)

        ind = (3, 0)
        axes[*ind].contourf(X_out, Y_out, out_out, cmap="coolwarm", vmin=0, vmax=1)
        axes[*ind].scatter(
            pos_seq_mlp[:, 0],
            pos_seq_mlp[:, 1],
            c=np.arange(pos_seq_res.shape[0]),
            marker=pos_marker,
            cmap="tab20b",
            s=100,
        )
        axes[*ind].scatter(
            neg_seq_mlp[:, 0],
            neg_seq_mlp[:, 1],
            c=np.arange(neg_seq_res.shape[0]),
            marker=neg_marker,
            cmap="tab20b",
            s=100,
        )
        for i, (x, y) in enumerate(pos_seq_mlp):
            t = axes[*ind].text(x, y, pos_inputs[i].numpy().tolist(), fontsize=text_fontsize)
            t.set_alpha(0.3)
        for i, (x, y) in enumerate(neg_seq_mlp):
            t = axes[*ind].text(x, y, neg_inputs[i].numpy().tolist(), fontsize=text_fontsize)
            t.set_alpha(0.3)
        axes[*ind].set_title("Output level lines", fontsize=title_fontsize)

        ind = (3, 1)
        axes[*ind].scatter(
            pos_seq_prob[:, 0],
            pos_seq_prob[:, 1],
            c=np.arange(pos_seq_prob.shape[0]),
            marker=pos_marker,
            cmap="tab20b",
            s=100,
        )
        axes[*ind].scatter(
            neg_seq_prob[:, 0],
            neg_seq_prob[:, 1],
            c=np.arange(neg_seq_prob.shape[0]),
            marker=neg_marker,
            cmap="tab20b",
            s=100,
        )
        for i, (x, y) in enumerate(pos_seq_prob):
            t = axes[*ind].text(x, y, inputs[i].numpy().tolist(), fontsize=text_fontsize)
            t.set_alpha(0.3)
        for i, (x, y) in enumerate(neg_seq_prob):
            t = axes[*ind].text(x, y, inputs[i].numpy().tolist(), fontsize=text_fontsize)
            t.set_alpha(0.3)
        axes[*ind].set_title("Output", fontsize=title_fontsize)

        ind = (3, 2)
        axes[*ind].plot(losses[: frame + 1], label="train")
        axes[*ind].plot(test_losses[: frame + 1], label="test")
        axes[*ind].set_title("Loss", fontsize=title_fontsize)
        axes[*ind].legend()
        axes[*ind].set_xlabel("Iterations")
        axes[*ind].set_ylabel("Loss")

        ind = (3, 3)
        axes[*ind].plot(accs[: frame + 1], label="train")
        axes[*ind].plot(test_accs[: frame + 1], label="test")
        axes[*ind].set_title("Accuracy", fontsize=title_fontsize)
        axes[*ind].legend()
        axes[*ind].set_xlabel("Iterations")
        axes[*ind].set_ylabel("Accuracy")

    if end_frame is None:
        update(start_frame)
        save_dir = IMAGE_DIR / "frames"
        save_dir.mkdir(exist_ok=True, parents=True)
        if file_format is None:
            file_format = "pdf"
        fig.savefig(save_dir / f"{unique_id}_{start_frame}.{file_format}", bbox_inches="tight")

    else:
        save_dir = IMAGE_DIR / "videos" / "parts" / str(unique_id)
        save_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Saving video ID {unique_id} from frame {start_frame} to frame {end_frame}.")
        ani = animation.FuncAnimation(fig, update, frames=range(start_frame, end_frame), repeat=False)
        if file_format is None:
            file_format = "mp4"
        if file_format == "gif":
            writer = "imagemagick"
        else:
            writer = "ffmpeg"
        ani.save(save_dir / f"{start_frame}_{end_frame}.{file_format}", writer=writer, fps=20)


# Aggregate videos


def aggregate_video(unique_id):
    """
    TODO: sort files by name.
    """
    film_dir = SAVE_DIR / unique_id / "animation"
    files_to_aggregate = [str(file) for file in film_dir.iterdir() if file.is_file()]
    logger.info(f"Aggregating {len(files_to_aggregate)} videos for ID {unique_id}.")
    logger.debug(files_to_aggregate)
    clips = [mpy.VideoFileClip(file) for file in files_to_aggregate]
    concat_clip = mpy.concatenate_videoclips(clips)
    film_dir = SAVE_DIR / "film"
    film_dir.mkdir(exist_ok=True)
    concat_clip.write_videofile(str(film_dir / f"{unique_id}.mp4"))


# CLI Wrapper


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "config": aggregate_configs,
            "losses": plot_all_losses,
            "animation": generate_animation,
            "frame": show_frame,
            "aggregate": aggregate_video,
        }
    )

# %%
