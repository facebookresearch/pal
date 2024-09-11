# %% Imports

import copy
import pickle
import subprocess
import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import rc
from tqdm import tqdm

from factorization.models.softmax_model import Model, RMSNorm

sys.path.append(str(Path("..").resolve()))


SAVE_DIR = Path(".").resolve() / "results"
DEVICE = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
SEED = None
SEED = 200
if SEED:
    RNG = np.random.default_rng(SEED)
    np.random.seed(seed=SEED)
    torch.manual_seed(seed=SEED)

rc("font", family="serif", size=8)
usetex = not subprocess.run(["which", "pdflatex"]).returncode
usetex = False
rc("text", usetex=usetex)
if usetex:
    rc("text.latex", preamble=r"\usepackage{times}")


# %% Utils


def copy_weights(model):
    if model.output.weight.device == torch.device("cpu"):
        return {k: copy.deepcopy(v) for k, v in model.state_dict().items()}
    else:
        return {k: v.cpu().detach() for k, v in model.state_dict().items()}


# %% Data

vocab_size = 2
# vocab_size = 4
bsz = 2048
length = 12
sparsity_index = 5

# modular addition problem on some subset of the input only
data = np.random.rand(bsz, length) // (1 / vocab_size)
targets = data[:, :sparsity_index].sum(axis=1) % vocab_size

test_bsz = 128
test_data = np.random.rand(test_bsz, length) // (1 / vocab_size)
test_targets = test_data[:, :sparsity_index].sum(axis=1) % vocab_size

print(f"Total number of unique sequences {vocab_size ** length}")


# %% Model

emb_dim = 2
# ffn_dim = 4 * emb_dim
ffn_dim = 10
vocab_size = 2
torch.manual_seed(20)

model = Model(emb_dim=emb_dim, vocab_size=vocab_size, length=length, ffn_dim=ffn_dim)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

model.to(device=DEVICE)


# %% Training Loop

niter = 4_000  # 2

X = torch.from_numpy(data).to(dtype=torch.long, device=DEVICE)
Y = torch.from_numpy(targets).to(dtype=torch.long, device=DEVICE)

X_test = torch.from_numpy(test_data).to(dtype=torch.long, device=DEVICE)
Y_test = torch.from_numpy(test_targets).to(dtype=torch.long, device=DEVICE)

# optimizer
lambda_l1 = 1e-4
lr = 1e-2  # emb_dim == 2 & ffn_dim == 32 & reg_l1 & model_seed 20
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

losses = torch.zeros(niter)
test_losses = torch.zeros(niter)
accs = torch.zeros(niter)
test_accs = torch.zeros(niter)

weights = [copy_weights(model)]

for i in (bar := tqdm(range(niter))):
    optimizer.zero_grad()

    # compute loss
    score = model(X, verbose=False)
    loss = F.cross_entropy(score.view((-1, vocab_size)), Y.view(-1))
    reg_loss = lambda_l1 * sum(p.abs().sum() for p in model.parameters())

    loss.backward()
    reg_loss.backward()
    optimizer.step()

    # record statistics
    with torch.no_grad():
        losses[i] = loss.item()
        accs[i] = (score.argmax(-1) == Y).float().mean()
        score_test = model(X_test)
        test_losses[i] = F.cross_entropy(score_test.view((-1, vocab_size)), Y_test.view(-1))
        test_accs[i] = (score_test.argmax(-1) == Y_test).float().mean()
        weights.append(copy_weights(model))

    bar.set_postfix(loss=losses[i].item(), acc=accs[i].item(), test_acc=test_accs[i].item())


# %% Saving the model

pickle.dump(weights, open(SAVE_DIR / "weights.pkl", "wb"))
pickle.dump(losses, open(SAVE_DIR / "losses.pkl", "wb"))
pickle.dump(test_losses, open(SAVE_DIR / "test_losses.pkl", "wb"))
pickle.dump(accs, open(SAVE_DIR / "accs.pkl", "wb"))
pickle.dump(test_accs, open(SAVE_DIR / "test_accs.pkl", "wb"))


# %% Loading results

weights = pickle.load(open(SAVE_DIR / "weights.pkl", "rb"))
losses = pickle.load(open(SAVE_DIR / "losses.pkl", "rb"))
test_losses = pickle.load(open(SAVE_DIR / "test_losses.pkl", "rb"))
accs = pickle.load(open(SAVE_DIR / "accs.pkl", "rb"))
test_accs = pickle.load(open(SAVE_DIR / "test_accs.pkl", "rb"))


# %% Accuracy and Loss

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(losses, label="train")
axes[0].plot(test_losses, label="test")
axes[0].set_title("Loss")
axes[0].legend()
axes[1].plot(accs, label="train")
axes[1].plot(test_accs, label="test")
axes[1].set_title("Accuracy")
axes[1].legend()
axes[1].grid()


# %% Visualization of the dynamics

DEVICE = "cpu"

# data
prefix = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 1],
    [0, 1, 1, 0, 0],
    [0, 0, 1, 1, 1],
    [0, 1, 1, 0, 1],
    [0, 1, 1, 1, 1],
    [1, 1, 0, 1, 1],
    [1, 1, 1, 1, 1],
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

tmpx = torch.linspace(-2.5, 2.5, 50)
tmpy = torch.linspace(-3.5, 2.5, 50)
X, Y = torch.meshgrid(tmpx, tmpy)
grid = torch.stack([X, Y], dim=-1).to(DEVICE)

# modules
model.to(DEVICE)
norm = RMSNorm()

# Create Animation

WIDTH = 15
HEIGHT = 20

fig, axes = plt.subplots(4, 3, figsize=(WIDTH, HEIGHT))

text_fontsize = 8
title_fontsize = 12

pos_marker = "o"
neg_marker = "s"


def update(frame):
    for i in range(4):
        for j in range(3):
            axes[i, j].clear()

    token_emb = weights[frame]["token_emb.weight"]
    pos_emb = weights[frame]["pos_emb.weight"]
    emb = (token_emb.unsqueeze(1) + pos_emb).reshape(-1, 2)
    norm_emb = norm(emb)

    query = weights[frame]["softmax.query.weight"]
    value = weights[frame]["softmax.value.weight"]
    emb_val = norm_emb @ value.T

    model.load_state_dict(weights[frame])
    with torch.no_grad():
        pos_seq_emb = model.softmax(model.norm1(model.token_emb(pos_inputs) + model.pos_emb.weight))
        neg_seq_emb = model.softmax(model.norm1(model.token_emb(neg_inputs) + model.pos_emb.weight))
        pos_seq_mlp = pos_seq_emb + model.mlp(model.norm2(pos_seq_emb))
        neg_seq_mlp = neg_seq_emb + model.mlp(model.norm2(neg_seq_emb))
        outputs = F.softmax(model.output(grid + model.mlp(model.norm2(grid))), dim=-1)[..., 1]
        pos_seq_prob = F.softmax(model.output(pos_seq_mlp), dim=-1)
        neg_seq_prob = F.softmax(model.output(neg_seq_mlp), dim=-1)

    axes[0, 0].scatter(token_emb[:, 0], token_emb[:, 1], c=np.arange(token_emb.shape[0]), cmap="tab20", s=100)
    for i, (x, y) in enumerate(token_emb):
        axes[0, 0].text(x, y, i, fontsize=text_fontsize)
    axes[0, 0].set_title("Token Embeddings", fontsize=title_fontsize)

    axes[0, 1].scatter(pos_emb[:, 0], pos_emb[:, 1], c=np.arange(pos_emb.shape[0]), cmap="tab20", s=100)
    for i, (x, y) in enumerate(pos_emb):
        axes[0, 1].text(x, y, i, fontsize=text_fontsize)
    axes[0, 1].set_title("Position Embeddings", fontsize=title_fontsize)

    axes[0, 2].scatter(emb[:, 0], emb[:, 1], c=np.arange(emb.shape[0]), cmap="tab20", s=100)
    for i, (x, y) in enumerate(emb):
        axes[0, 2].text(x, y, (i // 12, i % 12), fontsize=text_fontsize)
    axes[0, 2].set_title("Embeddings", fontsize=title_fontsize)

    axes[1, 0].scatter(norm_emb[:, 0], norm_emb[:, 1], c=np.arange(norm_emb.shape[0]), cmap="tab20", s=100)
    for i, (x, y) in enumerate(norm_emb):
        axes[1, 0].text(x, y, (i // 12, i % 12), fontsize=text_fontsize)
    axes[1, 0].arrow(0, 0, query[0, 0], query[0, 1], head_width=0.1, head_length=0.1, fc="r", ec="r")
    axes[1, 0].text(0, 0, "query", fontsize=text_fontsize + 2, color="r")
    axes[1, 0].set_title("Normed Embeddings", fontsize=title_fontsize)

    axes[1, 1].scatter(emb_val[:, 0], emb_val[:, 1], c=np.arange(emb_val.shape[0]), cmap="tab20", s=100)
    for i, (x, y) in enumerate(emb_val):
        axes[1, 1].text(x, y, (i // 12, i % 12), fontsize=text_fontsize)
    axes[1, 1].set_title("Value", fontsize=title_fontsize)

    axes[1, 2].scatter(
        pos_seq_emb[:, 0], pos_seq_emb[:, 1], c=np.arange(pos_seq_emb.shape[0]), marker=pos_marker, cmap="tab20b", s=100
    )
    axes[1, 2].scatter(
        neg_seq_emb[:, 0], neg_seq_emb[:, 1], c=np.arange(neg_seq_emb.shape[0]), marker=neg_marker, cmap="tab20b", s=100
    )
    for i, (x, y) in enumerate(pos_seq_emb):
        t = axes[1, 2].text(x, y, pos_inputs[i].numpy().tolist(), fontsize=text_fontsize)
        t.set_alpha(0.3)
    for i, (x, y) in enumerate(neg_seq_emb):
        t = axes[1, 2].text(x, y, neg_inputs[i].numpy().tolist(), fontsize=text_fontsize)
        t.set_alpha(0.3)
    axes[1, 2].set_title("Sequence Embeddings", fontsize=title_fontsize)

    axes[2, 0].scatter(
        pos_seq_mlp[:, 0], pos_seq_mlp[:, 1], c=np.arange(pos_seq_mlp.shape[0]), marker=pos_marker, cmap="tab20b", s=100
    )
    axes[2, 0].scatter(
        neg_seq_mlp[:, 0], neg_seq_mlp[:, 1], c=np.arange(neg_seq_mlp.shape[0]), marker=neg_marker, cmap="tab20b", s=100
    )
    for i, (x, y) in enumerate(pos_seq_mlp):
        t = axes[2, 0].text(x, y, inputs[i].numpy().tolist(), fontsize=text_fontsize)
        t.set_alpha(0.3)
    for i, (x, y) in enumerate(neg_seq_mlp):
        t = axes[2, 0].text(x, y, inputs[i].numpy().tolist(), fontsize=text_fontsize)
        t.set_alpha(0.3)
    axes[2, 0].set_title("MLP transform", fontsize=title_fontsize)

    CS = axes[2, 1].contourf(Y, X, outputs, cmap="coolwarm", vmin=0, vmax=1)
    axes[2, 1].scatter(
        pos_seq_mlp[:, 0], pos_seq_mlp[:, 1], c=np.arange(pos_seq_mlp.shape[0]), marker=pos_marker, cmap="tab20b", s=100
    )
    axes[2, 1].scatter(
        neg_seq_mlp[:, 0], neg_seq_mlp[:, 1], c=np.arange(neg_seq_mlp.shape[0]), marker=neg_marker, cmap="tab20b", s=100
    )
    for i, (x, y) in enumerate(pos_seq_mlp):
        t = axes[2, 1].text(x, y, inputs[i].numpy().tolist(), fontsize=text_fontsize)
        t.set_alpha(0.3)
    for i, (x, y) in enumerate(neg_seq_mlp):
        t = axes[2, 1].text(x, y, inputs[i].numpy().tolist(), fontsize=text_fontsize)
        t.set_alpha(0.3)
    axes[2, 1].set_title("MLP level lines", fontsize=title_fontsize)
    # show colorbar
    cbar = plt.colorbar(CS, ax=axes[2, 1])

    axes[2, 2].scatter(
        pos_seq_prob[:, 0],
        pos_seq_prob[:, 1],
        c=np.arange(pos_seq_prob.shape[0]),
        marker=pos_marker,
        cmap="tab20b",
        s=100,
    )
    axes[2, 2].scatter(
        neg_seq_prob[:, 0],
        neg_seq_prob[:, 1],
        c=np.arange(neg_seq_prob.shape[0]),
        marker=neg_marker,
        cmap="tab20b",
        s=100,
    )
    for i, (x, y) in enumerate(pos_seq_prob):
        t = axes[2, 2].text(x, y, inputs[i].numpy().tolist(), fontsize=text_fontsize)
        t.set_alpha(0.3)
    for i, (x, y) in enumerate(neg_seq_prob):
        t = axes[2, 2].text(x, y, inputs[i].numpy().tolist(), fontsize=text_fontsize)
        t.set_alpha(0.3)
    axes[2, 2].set_title("Output", fontsize=title_fontsize)

    axes[3, 0].plot(losses[: frame + 1], label="train")
    axes[3, 0].plot(test_losses[: frame + 1], label="test")
    axes[3, 0].set_title("Loss", fontsize=title_fontsize)
    axes[3, 0].legend()
    axes[3, 0].set_xlabel("Iterations")
    axes[3, 0].set_ylabel("Loss")

    axes[3, 1].plot(accs[: frame + 1], label="train")
    axes[3, 1].plot(test_accs[: frame + 1], label="test")
    axes[3, 1].set_title("Accuracy", fontsize=title_fontsize)
    axes[3, 1].legend()
    axes[3, 1].set_xlabel("Iterations")
    axes[3, 1].set_ylabel("Accuracy")


update(-1)
# length = 4000
# block_length = length // 20
# for i in range(20):
#     ani = animation.FuncAnimation(fig, update, frames=range(i * block_length, (i + 1) * block_length), repeat=False)
#     ani.save(SAVE_DIR / f"full_{i}.mp4", writer="ffmpeg", fps=20)

# %% Concatenate the different videos

# List of .mp4 files to concatenate
file_list = [str(SAVE_DIR / f"full_{i}.mp4") for i in range(18)]
clips = [mpy.VideoFileClip(file) for file in file_list]
concat_clip = mpy.concatenate_videoclips(clips)
concat_clip.write_videofile(str(SAVE_DIR / "full_visu.mp4"))

# %%

print(len(markers))
print(len(inputs))

# %%

import matplotlib.pyplot as plt

# Create some sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Create a list of markers
markers = ["o", "s", "^", "v", "<"]

# Create a scatter plot with different markers for each data point
plt.scatter(x, y, marker=markers)

# Show the plot
plt.show()

# %%