"""
License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import rc

sys.path.append(".")
from config import SAVE_DIR
from model import AssociativeMemory

os.makedirs(SAVE_DIR, exist_ok=True)

torch.manual_seed(42)


WIDTH = 8.5  # inches (from ICML style file)
HEIGHT = 8.5 / 1.618  # golden ratio

rc("font", family="serif", size=8)
usetex = not subprocess.run(["which", "pdflatex"]).returncode
rc("text", usetex=usetex)
if usetex:
    rc("text.latex", preamble=r"\usepackage{times}")

# hyparameters
n = 2
d = 2

log_p = np.logspace(-0.5, 0.5, 10)
ps = 1 / (1 + np.exp(-log_p))
lrs = np.logspace(-1, 1, 10)
alphas = np.linspace(0.5, 1, 10)


def f(x, epsilon=0):
    return x


# data
all_x = torch.arange(n)
all_y = f(all_x)
U = torch.eye(n)

all_accuracies = np.empty((len(ps), len(alphas), len(lrs)), dtype=int)
nb_epoch = 1000

for i_p, p in enumerate(ps):
    proba = torch.tensor([p] + [1 - p] * (n - 1))

    for i_a, alpha in enumerate(alphas):

        E = torch.eye(n)
        E[1, 0] = alpha
        E[1, 1] = np.sqrt(1 - alpha**2)
        model = AssociativeMemory(E, U)

        for i_l, lr in enumerate(lrs):
            accuracy = torch.zeros(nb_epoch)

            model.W.data[:] = 0
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0)

            for i in range(nb_epoch):
                # W_t[i] = model.W.detach()

                # full batch
                x = all_x
                y = all_y

                # compute loss
                score = model(x)
                loss = (proba * F.cross_entropy(score, y, reduction="none")).sum()

                # record statistics
                with torch.no_grad():
                    pred = model.fit(all_x)
                    accuracy[i] = proba[pred == all_y].sum().item()

                # update parameters with gradient descent
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # conv = W_t[-1].clone()
            # conv /= conv.norm()
            # max_proj_norm = torch.einsum('bii->b', W_t @ conv.T)
            # ortho_proj_norm = torch.norm(W_t - max_proj_norm.reshape(-1, 1, 1) * conv, dim=(1, 2))
            accuracy[-1] = 1
            all_accuracies[i_p, i_a, i_l] = accuracy.argmax().item()
np.save("exp_learning_rates.npy", all_accuracies)


X, Y = np.meshgrid(alphas, lrs)
for i in range(len(ps)):
    fig, ax = plt.subplots(figsize=(0.15 * WIDTH, 0.15 * HEIGHT))
    c = ax.contourf(X, Y, np.log10(all_accuracies[0].T), cmap="RdBu_r", levels=10)
    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\eta$")
    ax.set_title(r"$p_1 = {:.2f}$".format(ps[i]))
    ax.tick_params(axis="both", which="major", labelsize=6)
    cbar = fig.colorbar(c, ax=ax)
    fig.savefig(os.path.join(SAVE_DIR, f"exp_learning_rates_p_{i}.pdf"), bbox_inches="tight", pad_inches=0)

X, Y = np.meshgrid(log_p, lrs)
for i in range(len(alphas)):
    fig, ax = plt.subplots(figsize=(0.15 * WIDTH, 0.15 * HEIGHT))
    c = ax.contourf(X, Y, all_accuracies[:, i, :].T, cmap="RdBu_r")
    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log(p_1 / p_2)$")
    ax.set_ylabel(r"$\eta$")
    ax.set_title(r"$\alpha = {:.2f}$".format(alphas[i]))
    cbar = fig.colorbar(c, ax=ax)
    ax.tick_params(axis="both", which="major", labelsize=6)
    fig.savefig(os.path.join(SAVE_DIR, f"exp_learning_rates_alpha_{i}.pdf"), bbox_inches="tight", pad_inches=0)
