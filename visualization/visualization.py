"""
Visualization scripts

License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

# Imports

import logging
import pickle

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import numpy as np
import torch
import torch.nn.functional as F
from configs import get_paths, load_configs, recover_config
from matplotlib import rc

from factorization.config import IMAGE_DIR, USETEX
from factorization.models.softmax_model import Model, ModelConfig, RMSNorm

logger = logging.getLogger(__name__)

rc("font", family="serif", size=8)
rc("text", usetex=USETEX)
if USETEX:
    rc("text.latex", preamble=r"\usepackage{times}")


# Front-end


def show_frame(unique_id: int, epoch: int, file_format: str = None, save_ext: str = None, title: str = None):
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
    save_ext
        Experiments folder identifier.
    title
        Title for the plot.
    """
    config = recover_config(unique_id, save_ext=save_ext)
    assert config["save_weights"], f"Weights were not saved for ID {unique_id}."
    assert epoch <= config["nb_epochs"], f"Epoch {epoch} is greater than the number of epochs {config['nb_epochs']}."
    visualization_backend(
        unique_id, start_frame=epoch, end_frame=None, file_format=file_format, save_ext=save_ext, title=title
    )


def generate_animation(
    unique_id: int,
    num_tasks: int = 1,
    task_id: int = 1,
    file_format: str = None,
    save_ext: str = None,
    title: str = None,
):
    """
    Generate an animation for a given unique ID.

    Parameters
    ----------
    unique_id
        Unique identifier for the configuration file.
    num_tasks
        Number of tasks to divide the animation into.
    file_format
        File format for the video.
    task_id
        Current task ID.
    save_ext
        Experiments folder identifier.
    title
        Title for the plot.
    """
    config = recover_config(unique_id, save_ext=save_ext)
    assert config["save_weights"], f"Weights were not saved for ID {unique_id}."

    ani_length = config["nb_epochs"]
    assert (
        ani_length % num_tasks == 0
    ), f"Number of tasks {num_tasks} does not divide the number of epochs {ani_length}."
    block_length = ani_length // num_tasks
    start_frame = (task_id - 1) * block_length
    end_frame = task_id * block_length

    visualization_backend(unique_id, start_frame, end_frame, file_format=file_format, save_ext=save_ext, title=title)


def generate_all_animations(
    save_ext: str = None,
    num_tasks: int = 1,
    num_tasks_per_videos: int = 1,
    task_id: int = 1,
    file_format: str = None,
    title_key: str = None,
):
    """
    Generate all animations for a given configuration file.

    Parameters
    ----------
    save_ext
        Experiments folder identifier.
    num_tasks
        Number of tasks to divide the animation into.
    num_tasks_per_videos
        Number of tasks per video.
    task_id
        Current task ID.
    file_format
        File format for the video.
    title_key
        Key for the title in the configuration file.
    """
    all_configs = load_configs(save_ext)
    ind = 0
    for experiment in all_configs:
        for video_task_id in range(1, num_tasks_per_videos + 1):
            ind += 1
            if ind % num_tasks != task_id - 1:
                continue

            unique_id = experiment["id"]
            assert experiment["save_weights"], f"Weights were not saved for ID {unique_id}."

            ani_length = experiment["nb_epochs"]
            assert (
                ani_length % num_tasks_per_videos == 0
            ), f"Number of tasks {num_tasks_per_videos} does not divide the number of epochs {ani_length}."

            block_length = ani_length // num_tasks_per_videos
            start_frame = (video_task_id - 1) * block_length
            end_frame = video_task_id * block_length

            logger.info(f"Generating animation for ID {unique_id} from frame {start_frame} to frame {end_frame}.")
            title = f"{title_key}: {experiment[title_key]}" if title_key is not None else None
            visualization_backend(
                unique_id, start_frame, end_frame, file_format=file_format, save_ext=save_ext, title=title
            )


def aggregate_video(unique_id: int, save_ext: str = None):
    """
    Aggregate videos for a given unique ID.

    Parameters
    ----------
    unique_id
        Unique identifier for the configuration file.
    save_ext
        Experiments folder identifier.
    """
    save_ext = save_ext if save_ext is not None else "base"
    film_dir = IMAGE_DIR / "videos" / save_ext / "parts" / str(unique_id)
    files_to_aggregate = sorted([str(file) for file in film_dir.iterdir() if file.is_file()])
    logger.info(f"Aggregating {len(files_to_aggregate)} videos for ID {unique_id}.")
    clips = [mpy.VideoFileClip(file) for file in files_to_aggregate]
    concat_clip = mpy.concatenate_videoclips(clips)
    film_dir = IMAGE_DIR / "videos" / save_ext / "film"
    film_dir.mkdir(exist_ok=True)
    concat_clip.write_videofile(str(film_dir / f"{unique_id}.mp4"))


def aggregate_all_videos(save_ext: str = None):
    """
    Aggregate videos for a given unique ID.

    Parameters
    ----------
    unique_id
        Unique identifier for the configuration file.
    save_ext
        Experiments folder identifier.
    """
    all_configs = load_configs(save_ext)
    for experiment in all_configs:
        try:
            aggregate_video(experiment["id"], save_ext=save_ext)
        except Exception as e:
            logger.error(f"Error for ID {experiment['id']}: {e}")
            continue


# Back-end


def visualization_backend(
    unique_id: int,
    start_frame: int,
    end_frame: int = None,
    file_format: str = None,
    save_ext: str = None,
    title: str = None,
    plot_config: str = None,
):
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
    save_ext
        Experiments folder identifier.
    title
        Title for the plot.
    plot_config
        Configuration for the plot.
    """

    # configuration and saved computations

    config = recover_config(unique_id, save_ext=save_ext)
    vocab_size = config["vocab_size"]
    length = config["seq_length"]
    sparsity_index = config["sparsity_index"]
    ffn_dim = config["ffn_dim"]
    assert config["save_weights"], f"Weights were not saved for ID {unique_id}."

    save_dir, _ = get_paths(save_ext)
    save_dir = save_dir / unique_id
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

    # variables

    kwargs = {
        "DEVICE": DEVICE,
        "vocab_size": vocab_size,
        "sparsity_index": sparsity_index,
        "length": length,
        "ffn_dim": ffn_dim,
        "losses": losses,
        "test_losses": test_losses,
        "accs": accs,
        "test_accs": test_accs,
        "text_fontsize": 8,
        "title_fontsize": 12,
        "pos_marker": "o",
        "neg_marker": "s",
    }

    # plot configurations

    plot_functions = {
        "show_token_emb": show_token_emb,
        "show_pos_emb": show_pos_emb,
        "show_emb": show_emb,
        "show_norm_emb": show_norm_emb,
        "show_attn": show_attn,
        "show_value": show_value,
        "show_seq_emb": show_seq_emb,
        "show_level_line": show_level_line,
        "show_norm_input": show_norm_input,
        "show_mlp_receptors": show_mlp_receptors,
        "show_mlp_emitters": show_mlp_emitters,
        "show_mlp_output": show_mlp_output,
        "show_output_level_lines": show_output_level_lines,
        "show_output": show_output,
        "show_loss": show_loss,
        "show_acc": show_acc,
    }

    if plot_config is None:
        plot_config = {
            "grid_size": [4, 4],
            "plots": [
                {"type": "show_token_emb", "position": [0, 0]},
                {"type": "show_pos_emb", "position": [0, 1]},
                {"type": "show_emb", "position": [0, 2]},
                {"type": "show_norm_emb", "position": [0, 3]},
                {"type": "show_attn", "position": [1, 0]},
                {"type": "show_value", "position": [1, 1]},
                {"type": "show_seq_emb", "position": [1, 2]},
                {"type": "show_level_line", "position": [1, 3]},
                {"type": "show_norm_input", "position": [2, 0]},
                {"type": "show_mlp_receptors", "position": [2, 1]},
                {"type": "show_mlp_emitters", "position": [2, 2]},
                {"type": "show_mlp_output", "position": [2, 3]},
                {"type": "show_output_level_lines", "position": [3, 0]},
                {"type": "show_output", "position": [3, 1]},
                {"type": "show_loss", "position": [3, 2]},
                {"type": "show_acc", "position": [3, 3]},
            ],
        }

    grid_size = plot_config["grid_size"]
    plot_requests = plot_config["plots"]

    WIDTH = 5 * grid_size[1]
    HEIGHT = 5 * grid_size[0]
    fig, axes = plt.subplots(*grid_size, figsize=(WIDTH, HEIGHT))
    if title is not None:
        fig.suptitle(title)

    # frame creation

    def update(frame):
        for ax in axes.flat:
            ax.clear()

        model.load_state_dict(weights[frame])
        variables = ComputationCache(
            {
                "weights": weights[frame],
                "model": model,
                "norm": norm,
                "frame": frame,
            }
            | kwargs
        )

        for plot_request in plot_requests:
            plot_type = plot_request["type"]
            position = tuple(plot_request["position"])
            plot_func = plot_functions.get(plot_type)

            if plot_func:
                ax = axes[position]
                plot_func(ax, variables)
            else:
                logger.info(f"Plot type {plot_type} not recognized.")

    if end_frame is None:
        update(start_frame)
        save_dir = IMAGE_DIR / "frames" / save_ext
        save_dir.mkdir(exist_ok=True, parents=True)
        if file_format is None:
            file_format = "pdf"
        fig.savefig(save_dir / f"{unique_id}_{start_frame}.{file_format}", bbox_inches="tight")

    else:
        save_ext = save_ext if save_ext is not None else "base"
        save_dir = IMAGE_DIR / "videos" / save_ext / "parts" / str(unique_id)
        save_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Saving video ID {unique_id} from frame {start_frame} to frame {end_frame}.")
        ani = animation.FuncAnimation(fig, update, frames=range(start_frame, end_frame), repeat=False)
        if file_format is None:
            file_format = "mp4"
        if file_format == "gif":
            writer = "imagemagick"
        else:
            writer = "ffmpeg"
        ani.save(save_dir / f"{start_frame:0>5}.{file_format}", writer=writer, fps=20)


class ComputationCache:
    def __init__(self, variables: dict[str, any]):
        self.locals = variables

    def __getitem__(self, key):
        if key in self.locals:
            return self.locals[key]
        method_name = f"get_{key}"
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            with torch.no_grad():
                value = method()
            self.locals[key] = value
            return value
        raise KeyError(f"{key} not found in ComputationCache.")

    def get_inputs(self):
        DEVICE = self["DEVICE"]
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
        return torch.tensor([pre + suf for pre in prefix for suf in suffixes], device=DEVICE)

    def get_targets(self):
        inputs = self["inputs"]
        vocab_size = self["vocab_size"]
        sparsity_index = self["sparsity_index"]
        return inputs[:, :sparsity_index].sum(dim=1) % vocab_size

    def get_pos_inputs(self):
        inputs = self["inputs"]
        targets = self["targets"]
        return inputs[targets == 1]

    def get_neg_inputs(self):
        inputs = self["inputs"]
        targets = self["targets"]
        return inputs[targets == 0]

    def get_mlp_meshgrid(self):
        xlim = (min(pos_seq_emb[:, 0].min(), neg_seq_res[:, 0].min()),
                max(pos_seq_emb[:, 0].max(), neg_seq_res[:, 0].max()))
        xdelta = (xlim[1] - xlim[0]) * 0.1
        ylim = (min(pos_seq_emb[:, 1].min(), neg_seq_res[:, 1].min()),
                max(pos_seq_emb[:, 1].max(), neg_seq_res[:, 1].max()))
        ydelta = (ylim[1] - ylim[0]) * 0.1
        tmpx = torch.linspace(xlim[0] - xdelta, xlim[1] + xdelta, 50)
        tmpy = torch.linspace(ylim[0] - ydelta, ylim[1] + ydelta, 50)
        return torch.meshgrid(tmpx, tmpy)

    def get_X_mlp(self):
        return self["mlp_meshgrid"][0]

    def get_Y_mlp(self):
        return self["mlp_meshgrid"][1]

    def get_grid_mlp(self):
        X_mlp = self["X_mlp"]
        Y_mlp = self["Y_mlp"]
        DEVICE = self["DEVICE"]
        return torch.stack([X_mlp, Y_mlp], dim=-1).to(DEVICE).view(-1, 2)

    def get_out_meshgrid(self):
        pos_seq_res = self["pos_seq_res"]
        neg_seq_res = self["neg_seq_res"]

        xlim = (min(pos_seq_res[:, 0].min(), neg_seq_res[:, 0].min()),
                max(pos_seq_res[:, 0].max(), neg_seq_res[:, 0].max()))
        xdelta = (xlim[1] - xlim[0]) * 0.1
        ylim = (min(pos_seq_res[:, 1].min(), neg_seq_res[:, 1].min()),
                max(pos_seq_res[:, 1].max(), neg_seq_res[:, 1].max()))
        ydelta = (ylim[1] - ylim[0]) * 0.1
        tmpx = torch.linspace(xlim[0] - xdelta, xlim[1] + xdelta, 50)
        tmpy = torch.linspace(ylim[0] - ydelta, ylim[1] + ydelta, 50)
        return torch.meshgrid(tmpx, tmpy)

    def get_X_out(self):
        return self["out_meshgrid"][0]

    def get_Y_out(self):
        return self["out_meshgrid"][1]

    def get_grid_out(self):
        X_out = self["X_out"]
        Y_out = self["Y_out"]
        DEVICE = self["DEVICE"]
        return torch.stack([X_out, Y_out], dim=-1).to(DEVICE).view(-1, 2)

    def get_token_emb(self):
        weights = self["weights"]
        vocab_size = self["vocab_size"]
        return weights["token_emb.weight"][:vocab_size]

    def get_pos_emb(self):
        weights = self["weights"]
        return weights["pos_emb.weight"]

    def get_emb(self):
        token_emb = self["token_emb"]
        pos_emb = self["pos_emb"]
        return (token_emb.unsqueeze(1) + pos_emb).reshape(-1, 2)

    def get_norm_emb(self):
        emb = self["emb"]
        norm = self["norm"]
        return norm(emb)

    def get_query(self):
        weights = self["weights"]
        return weights["softmax.query.weight"]

    def get_emb_val(self):
        norm_emb = self["norm_emb"]
        weights = self["weights"]
        value = weights["softmax.value.weight"]
        return norm_emb @ value.T

    def get_attn(self):
        model = self["model"]
        pos_inputs = self["pos_inputs"]
        norm = self["norm"]
        _, out = model.softmax(norm(model.token_emb(pos_inputs) + model.pos_emb.weight), verbose=True)
        return out

    def get_pos_seq_emb(self):
        model = self["model"]
        pos_inputs = self["pos_inputs"]
        norm = self["norm"]
        out, _ = model.softmax(norm(model.token_emb(pos_inputs) + model.pos_emb.weight), verbose=True)
        return out

    def get_neg_seq_emb(self):
        model = self["model"]
        neg_inputs = self["neg_inputs"]
        norm = self["norm"]
        return model.softmax(norm(model.token_emb(neg_inputs) + model.pos_emb.weight))

    def get_norm_pos_seq(self):
        norm = self["norm"]
        pos_seq_emb = self["pos_seq_emb"]
        return norm(pos_seq_emb)

    def get_norm_neg_seq(self):
        norm = self["norm"]
        neg_seq_emb = self["neg_seq_emb"]
        return norm(neg_seq_emb)

    def get_fc1(self):
        model = self["model"]
        return model.mlp.fc1.weight.detach()

    def get_fc2(self):
        model = self["model"]
        return model.mlp.fc2.weight.detach()

    def get_pos_seq_mlp(self):
        model = self["model"]
        norm_pos_seq = self["norm_pos_seq"]
        return model.mlp(norm_pos_seq)

    def get_neg_seq_mlp(self):
        model = self["model"]
        norm_neg_seq = self["norm_neg_seq"]
        return model.mlp(norm_neg_seq)

    def get_pos_seq_res(self):
        pos_seq_emb = self["pos_seq_emb"]
        pos_seq_mlp = self["pos_seq_mlp"]
        self.locals["pos_seq_res"] = pos_seq_emb + pos_seq_mlp
        return self.locals["pos_seq_res"]

    def get_neg_seq_res(self):
        neg_seq_emb = self["neg_seq_emb"]
        neg_seq_mlp = self["neg_seq_mlp"]
        return neg_seq_emb + neg_seq_mlp

    def get_out_mlp(self):
        model = self["model"]
        X_mlp = self["X_mlp"]
        grid_mlp = self["grid_mlp"]
        norm = self["norm"]
        return F.softmax(model.output(grid_mlp + model.mlp(norm(grid_mlp))), dim=-1)[..., 1].view(X_mlp.shape)

    def get_out_out(self):
        model = self["model"]
        X_out = self["X_out"]
        grid_out = self["grid_out"]
        return F.softmax(model.output(grid_out), dim=-1)[..., 1].view(X_out.shape)

    def get_pos_seq_prob(self):
        model = self["model"]
        vocab_size = self["vocab_size"]
        pos_seq_res = self["pos_seq_res"]
        return F.softmax(model.output(pos_seq_res), dim=-1)[:, :vocab_size]

    def get_neg_seq_prob(self):
        model = self["model"]
        vocab_size = self["vocab_size"]
        neg_seq_res = self["neg_seq_res"]
        return F.softmax(model.output(neg_seq_res), dim=-1)[:, :vocab_size]


def show_token_emb(ax, kwargs):
    token_emb = kwargs["token_emb"]
    vocab_size = len(token_emb)
    ax.scatter(
        token_emb[:, 0],
        token_emb[:, 1],
        c=np.arange(vocab_size),
        cmap="tab20",
        s=100,
    )
    for i, (x, y) in enumerate(token_emb):
        ax.text(x, y, i, fontsize=kwargs["text_fontsize"])
    ax.grid()
    ax.set_title("Token Embeddings $E$", fontsize=kwargs["title_fontsize"])


def show_pos_emb(ax, kwargs):
    pos_emb = kwargs["pos_emb"]
    sparsity_index = kwargs["sparsity_index"]
    length = kwargs["length"]
    ax.scatter(
        pos_emb[:sparsity_index, 0],
        pos_emb[:sparsity_index, 1],
        c=np.arange(sparsity_index),
        cmap="tab20",
        s=100,
        marker=kwargs["pos_marker"],
    )
    ax.scatter(
        pos_emb[sparsity_index:, 0],
        pos_emb[sparsity_index:, 1],
        c=np.arange(sparsity_index, length),
        cmap="tab20",
        s=100,
        marker=kwargs["neg_marker"],
    )
    for i, (x, y) in enumerate(pos_emb):
        ax.text(x, y, i, fontsize=kwargs["text_fontsize"])
    ax.grid()
    ax.set_title("Position Embeddings $P$", fontsize=kwargs["title_fontsize"])


def show_emb(ax, kwargs):
    emb = kwargs["emb"]
    sparsity_index = kwargs["sparsity_index"]
    length = kwargs["length"]
    ax.scatter([0], [0], c="k", marker="o", s=50)
    ax.scatter(
        emb[:sparsity_index, 0],
        emb[:sparsity_index, 1],
        c=np.arange(sparsity_index),
        cmap="tab20",
        marker=kwargs["pos_marker"],
        s=100,
    )
    ax.scatter(
        emb[length : length + sparsity_index, 0],
        emb[length : length + sparsity_index, 1],
        c=np.arange(sparsity_index),
        cmap="tab20",
        marker=kwargs["pos_marker"],
        s=100,
    )
    ax.scatter(
        emb[sparsity_index:length, 0],
        emb[sparsity_index:length, 1],
        c=np.arange(sparsity_index, length),
        cmap="tab20",
        marker=kwargs["neg_marker"],
        s=100,
    )
    ax.scatter(
        emb[length + sparsity_index :, 0],
        emb[length + sparsity_index :, 1],
        c=np.arange(sparsity_index, length),
        cmap="tab20",
        marker=kwargs["neg_marker"],
        s=100,
    )
    for i, (x, y) in enumerate(emb):
        ax.text(x, y, (i // 12, i % 12), fontsize=kwargs["text_fontsize"])
    ax.grid()
    ax.set_title("Embeddings $E + P$", fontsize=kwargs["title_fontsize"])


def show_norm_emb(ax, kwargs):
    norm_emb = kwargs["norm_emb"]
    query = kwargs["query"]
    sparsity_index = kwargs["sparsity_index"]
    length = kwargs["length"]
    ax.scatter(
        norm_emb[:sparsity_index, 0],
        norm_emb[:sparsity_index, 1],
        c=np.arange(sparsity_index),
        cmap="tab20",
        marker=kwargs["pos_marker"],
        s=100,
    )
    ax.scatter(
        norm_emb[length : length + sparsity_index, 0],
        norm_emb[length : length + sparsity_index, 1],
        c=np.arange(sparsity_index),
        cmap="tab20",
        marker=kwargs["pos_marker"],
        s=100,
    )
    ax.scatter(
        norm_emb[sparsity_index:length, 0],
        norm_emb[sparsity_index:length, 1],
        c=np.arange(sparsity_index, length),
        cmap="tab20",
        marker=kwargs["neg_marker"],
        s=100,
    )
    ax.scatter(
        norm_emb[length + sparsity_index :, 0],
        norm_emb[length + sparsity_index :, 1],
        c=np.arange(sparsity_index, length),
        cmap="tab20",
        marker=kwargs["neg_marker"],
        s=100,
    )
    for i, (x, y) in enumerate(norm_emb):
        ax.text(x, y, (i // 12, i % 12), fontsize=kwargs["text_fontsize"])
    ax.arrow(0, 0, query[0, 0], query[0, 1], head_width=0.1, head_length=0.1, fc="r", ec="r")
    ax.text(0, 0, "query", fontsize=kwargs["text_fontsize"] + 2, color="r")
    ax.set_title(r"Normed Embeddings $Z(x,t) \propto E(x) + P(t)$", fontsize=kwargs["title_fontsize"])


def show_attn(ax, kwargs):
    attn = kwargs["attn"]
    ax.imshow(attn, cmap="Blues", vmin=0, vmax=0.2)
    ax.plot([4.5, 4.5], [-1, len(attn)], color="C3")
    ax.set_title("Attention vectors", fontsize=kwargs["title_fontsize"])
    ax.axis("off")


def show_value(ax, kwargs):
    emb_val = kwargs["emb_val"]
    sparsity_index = kwargs["sparsity_index"]
    length = kwargs["length"]
    ax.scatter(
        emb_val[:sparsity_index, 0],
        emb_val[:sparsity_index, 1],
        c=np.arange(sparsity_index),
        cmap="tab20",
        marker=kwargs["pos_marker"],
        s=100,
    )
    ax.scatter(
        emb_val[length : length + sparsity_index, 0],
        emb_val[length : length + sparsity_index, 1],
        c=np.arange(sparsity_index),
        cmap="tab20",
        marker=kwargs["pos_marker"],
        s=100,
    )
    ax.scatter(
        emb_val[sparsity_index:length, 0],
        emb_val[sparsity_index:length, 1],
        c=np.arange(sparsity_index, length),
        cmap="tab20",
        marker=kwargs["neg_marker"],
        s=100,
    )
    ax.scatter(
        emb_val[length + sparsity_index :, 0],
        emb_val[length + sparsity_index :, 1],
        c=np.arange(sparsity_index, length),
        cmap="tab20",
        marker=kwargs["neg_marker"],
        s=100,
    )
    for i, (x, y) in enumerate(emb_val):
        ax.text(x, y, (i // 12, i % 12), fontsize=kwargs["text_fontsize"])
    ax.grid()
    ax.set_title("Value", fontsize=kwargs["title_fontsize"])


def show_seq_emb(ax, kwargs):
    pos_seq_emb = kwargs["pos_seq_emb"]
    neg_seq_emb = kwargs["neg_seq_emb"]
    pos_inputs = kwargs["pos_inputs"]
    neg_inputs = kwargs["neg_inputs"]
    ax.scatter([0], [0], c="k", marker="o", s=50)
    ax.scatter(
        pos_seq_emb[:, 0],
        pos_seq_emb[:, 1],
        c=np.arange(pos_seq_emb.shape[0]),
        marker=kwargs["pos_marker"],
        cmap="tab20b",
        s=100,
    )
    ax.scatter(
        neg_seq_emb[:, 0],
        neg_seq_emb[:, 1],
        c=np.arange(neg_seq_emb.shape[0]),
        marker=kwargs["neg_marker"],
        cmap="tab20b",
        s=100,
    )
    for i, (x, y) in enumerate(pos_seq_emb):
        t = ax.text(x, y, pos_inputs[i].numpy().tolist(), fontsize=kwargs["text_fontsize"])
        t.set_alpha(0.3)
    for i, (x, y) in enumerate(neg_seq_emb):
        t = ax.text(x, y, neg_inputs[i].numpy().tolist(), fontsize=kwargs["text_fontsize"])
        t.set_alpha(0.3)
    ax.grid()
    ax.set_title(r"Sequence Embeddings $\xi$", fontsize=kwargs["title_fontsize"])


def show_level_line(ax, kwargs):
    X_mlp = kwargs["X_mlp"]
    Y_mlp = kwargs["Y_mlp"]
    out_mlp = kwargs["out_mlp"]
    pos_seq_emb = kwargs["pos_seq_emb"]
    neg_seq_emb = kwargs["neg_seq_emb"]
    pos_inputs = kwargs["pos_inputs"]
    neg_inputs = kwargs["neg_inputs"]
    ax.contourf(X_mlp, Y_mlp, out_mlp, cmap="coolwarm", vmin=0, vmax=1)
    ax.scatter(
        pos_seq_emb[:, 0],
        pos_seq_emb[:, 1],
        c=np.arange(pos_seq_emb.shape[0]),
        marker=kwargs["pos_marker"],
        cmap="tab20b",
        s=100,
    )
    ax.scatter(
        neg_seq_emb[:, 0],
        neg_seq_emb[:, 1],
        c=np.arange(neg_seq_emb.shape[0]),
        marker=kwargs["neg_marker"],
        cmap="tab20b",
        s=100,
    )
    for i, (x, y) in enumerate(pos_seq_emb):
        t = ax.text(x, y, pos_inputs[i].numpy().tolist(), fontsize=kwargs["text_fontsize"])
        t.set_alpha(0.3)
    for i, (x, y) in enumerate(neg_seq_emb):
        t = ax.text(x, y, neg_inputs[i].numpy().tolist(), fontsize=kwargs["text_fontsize"])
        t.set_alpha(0.3)
    ax.set_title(r"Transform level lines: $\xi \to p(y=1|\xi)$", fontsize=kwargs["title_fontsize"])


def show_norm_input(ax, kwargs):
    norm_pos_seq = kwargs["norm_pos_seq"]
    norm_neg_seq = kwargs["norm_neg_seq"]
    pos_inputs = kwargs["pos_inputs"]
    neg_inputs = kwargs["neg_inputs"]
    ax.scatter(
        norm_pos_seq[:, 0],
        norm_pos_seq[:, 1],
        c=np.arange(norm_pos_seq.shape[0]),
        marker=kwargs["pos_marker"],
        cmap="tab20b",
        s=100,
    )
    ax.scatter(
        norm_neg_seq[:, 0],
        norm_neg_seq[:, 1],
        c=np.arange(norm_neg_seq.shape[0]),
        marker=kwargs["neg_marker"],
        cmap="tab20b",
        s=100,
    )
    for i, (x, y) in enumerate(norm_pos_seq):
        t = ax.text(x, y, pos_inputs[i].numpy().tolist(), fontsize=kwargs["text_fontsize"])
        t.set_alpha(0.3)
    for i, (x, y) in enumerate(norm_neg_seq):
        t = ax.text(x, y, neg_inputs[i].numpy().tolist(), fontsize=kwargs["text_fontsize"])
        t.set_alpha(0.3)
    ax.set_title(r"Normed Input: $\xi / \|\xi\|$", fontsize=kwargs["title_fontsize"])


def show_mlp_receptors(ax, kwargs):
    fc1 = kwargs["fc1"]
    norm_pos_seq = kwargs["norm_pos_seq"]
    norm_neg_seq = kwargs["norm_neg_seq"]
    ffn_dim = kwargs["ffn_dim"]
    ax.scatter([0], [0], c="k", marker="o", s=50)
    ax.scatter(
        fc1[:, 0],
        fc1[:, 1],
        c=np.arange(ffn_dim),
        cmap="tab20",
        marker="^",
        s=100,
    )
    for i in range(ffn_dim):
        ax.plot([0, fc1[i, 0]], [0, fc1[i, 1]], alpha=0.2)
    ax.scatter(
        norm_pos_seq[:, 0],
        norm_pos_seq[:, 1],
        c=np.arange(norm_pos_seq.shape[0]),
        marker=kwargs["pos_marker"],
        cmap="tab20b",
        s=100,
        alpha=0.1,
    )
    ax.scatter(
        norm_neg_seq[:, 0],
        norm_neg_seq[:, 1],
        c=np.arange(norm_neg_seq.shape[0]),
        marker=kwargs["neg_marker"],
        cmap="tab20b",
        s=100,
        alpha=0.1,
    )
    ax.grid()
    ax.set_title("MLP receptors", fontsize=kwargs["title_fontsize"])


def show_mlp_emitters(ax, kwargs):
    fc2 = kwargs["fc2"]
    fc1 = kwargs["fc1"]
    norm_pos_seq = kwargs["norm_pos_seq"]
    norm_neg_seq = kwargs["norm_neg_seq"]
    ffn_dim = kwargs["ffn_dim"]
    ax.scatter([0], [0], c="k", marker="o", s=50)
    ax.scatter(
        fc2[0],
        fc2[1],
        c=np.arange(ffn_dim),
        cmap="tab20",
        marker="v",
        s=100,
    )
    ax.scatter(
        fc1[:, 0],
        fc1[:, 1],
        c=np.arange(ffn_dim),
        cmap="tab20",
        marker="^",
        s=100,
        alpha=0.2,
    )
    for i in range(ffn_dim):
        ax.plot([0, fc1[i, 0]], [0, fc1[i, 1]], alpha=0.2)
    ax.scatter(
        norm_pos_seq[:, 0],
        norm_pos_seq[:, 1],
        c=np.arange(norm_pos_seq.shape[0]),
        marker=kwargs["pos_marker"],
        cmap="tab20b",
        s=100,
        alpha=0.1,
    )
    ax.scatter(
        norm_neg_seq[:, 0],
        norm_neg_seq[:, 1],
        c=np.arange(norm_neg_seq.shape[0]),
        marker=kwargs["neg_marker"],
        cmap="tab20b",
        s=100,
        alpha=0.1,
    )
    ax.grid()
    ax.set_title("MLP assemblers", fontsize=kwargs["title_fontsize"])


def show_mlp_output(ax, kwargs):
    pos_seq_mlp = kwargs["pos_seq_mlp"]
    neg_seq_mlp = kwargs["neg_seq_mlp"]
    pos_seq_res = kwargs["pos_seq_res"]
    neg_seq_res = kwargs["neg_seq_res"]
    pos_inputs = kwargs["pos_inputs"]
    neg_inputs = kwargs["neg_inputs"]
    ax.scatter(
        pos_seq_mlp[:, 0],
        pos_seq_mlp[:, 1],
        c=np.arange(pos_seq_mlp.shape[0]),
        marker=kwargs["pos_marker"],
        cmap="tab20b",
        s=100,
    )
    ax.scatter(
        neg_seq_mlp[:, 0],
        neg_seq_mlp[:, 1],
        c=np.arange(neg_seq_mlp.shape[0]),
        marker=kwargs["neg_marker"],
        cmap="tab20b",
        s=100,
    )
    for i, (x, y) in enumerate(pos_seq_mlp):
        t = ax.text(x, y, pos_inputs[i].numpy().tolist(), fontsize=kwargs["text_fontsize"])
        t.set_alpha(0.3)
    for i, (x, y) in enumerate(neg_seq_mlp):
        t = ax.text(x, y, neg_inputs[i].numpy().tolist(), fontsize=kwargs["text_fontsize"])
        t.set_alpha(0.3)

    ax.scatter(
        pos_seq_res[:, 0],
        pos_seq_res[:, 1],
        c=np.arange(pos_seq_res.shape[0]),
        marker=kwargs["pos_marker"],
        cmap="tab20b",
        s=100,
        alpha=0.2,
    )
    ax.scatter(
        neg_seq_res[:, 0],
        neg_seq_res[:, 1],
        c=np.arange(neg_seq_res.shape[0]),
        marker=kwargs["neg_marker"],
        cmap="tab20b",
        s=100,
        alpha=0.2,
    )
    ax.set_title("Transformed Sequences with residual", fontsize=kwargs["title_fontsize"])


def show_output_level_lines(ax, kwargs):
    X_out = kwargs["X_out"]
    Y_out = kwargs["Y_out"]
    out_out = kwargs["out_out"]
    pos_seq_mlp = kwargs["pos_seq_mlp"]
    neg_seq_mlp = kwargs["neg_seq_mlp"]
    pos_inputs = kwargs["pos_inputs"]
    neg_inputs = kwargs["neg_inputs"]
    ax.contourf(X_out, Y_out, out_out, cmap="coolwarm", vmin=0, vmax=1)
    ax.scatter(
        pos_seq_mlp[:, 0],
        pos_seq_mlp[:, 1],
        c=np.arange(pos_seq_mlp.shape[0]),
        marker=kwargs["pos_marker"],
        cmap="tab20b",
        s=100,
    )
    ax.scatter(
        neg_seq_mlp[:, 0],
        neg_seq_mlp[:, 1],
        c=np.arange(neg_seq_mlp.shape[0]),
        marker=kwargs["neg_marker"],
        cmap="tab20b",
        s=100,
    )
    for i, (x, y) in enumerate(pos_seq_mlp):
        t = ax.text(x, y, pos_inputs[i].numpy().tolist(), fontsize=kwargs["text_fontsize"])
        t.set_alpha(0.3)
    for i, (x, y) in enumerate(neg_seq_mlp):
        t = ax.text(x, y, neg_inputs[i].numpy().tolist(), fontsize=kwargs["text_fontsize"])
        t.set_alpha(0.3)
    ax.set_title("Output level lines", fontsize=kwargs["title_fontsize"])


def show_output(ax, kwargs):
    pos_seq_prob = kwargs["pos_seq_prob"]
    neg_seq_prob = kwargs["neg_seq_prob"]
    inputs = kwargs["inputs"]
    ax.scatter(
        pos_seq_prob[:, 0],
        pos_seq_prob[:, 1],
        c=np.arange(pos_seq_prob.shape[0]),
        marker=kwargs["pos_marker"],
        cmap="tab20b",
        s=100,
    )
    ax.scatter(
        neg_seq_prob[:, 0],
        neg_seq_prob[:, 1],
        c=np.arange(neg_seq_prob.shape[0]),
        marker=kwargs["neg_marker"],
        cmap="tab20b",
        s=100,
    )
    for i, (x, y) in enumerate(pos_seq_prob):
        t = ax.text(x, y, inputs[i].numpy().tolist(), fontsize=kwargs["text_fontsize"])
        t.set_alpha(0.3)
    for i, (x, y) in enumerate(neg_seq_prob):
        t = ax.text(x, y, inputs[i].numpy().tolist(), fontsize=kwargs["text_fontsize"])
        t.set_alpha(0.3)
    ax.set_title("Output", fontsize=kwargs["title_fontsize"])


def show_loss(ax, kwargs):
    frame = kwargs["frame"]
    losses = kwargs["losses"]
    test_losses = kwargs["test_losses"]
    ax.plot(losses[:frame:], label="train")
    ax.plot(test_losses[:frame], label="test")
    ax.set_title("Loss", fontsize=kwargs["title_fontsize"])
    ax.legend()
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")


def show_acc(ax, kwargs):
    frame = kwargs["frame"]
    accs = kwargs["accs"]
    test_accs = kwargs["test_accs"]
    ax.plot(accs[:frame], label="train")
    ax.plot(test_accs[:frame], label="test")
    ax.set_title("Accuracy", fontsize=kwargs["title_fontsize"])
    ax.legend()
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Accuracy")


# CLI Wrapper


if __name__ == "__main__":
    import fire

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()]
    )

    fire.Fire(
        {
            "animation": generate_animation,
            "all_animation": generate_all_animations,
            "frame": show_frame,
            "aggregate": aggregate_video,
            "all_aggregate": aggregate_all_videos,
        }
    )
