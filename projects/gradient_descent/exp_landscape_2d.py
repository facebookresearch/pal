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
p = 0.75  # probability of the first tokens


def f(x, epsilon=0):
    return x


# data
all_x = torch.arange(n)
all_y = f(all_x)
proba = torch.tensor([p, 1 - p])
U = torch.eye(n)

for alpha, sign in zip([-0.5, 0.5, 0.95], ["neg", "pos", "spike"]):

    E = torch.eye(n)
    E[1, 0] = alpha
    E[1, 1] = np.sqrt(1 - alpha**2)
    model = AssociativeMemory(E, U)

    for lim, res in zip([1, 10], ["close", "far"]):
        num = 50
        gamma_0, gamma_1 = np.meshgrid(np.linspace(-lim, lim, num=num), np.linspace(-lim, lim, num=num))

        Ws = np.zeros((num * num, d, d))
        Ws[:, 0, 0] = gamma_0.flatten()
        Ws[:, 1, 0] = gamma_1.flatten()
        Ws[:, 0, 1] = -Ws[:, 0, 0]
        Ws[:, 1, 1] = -Ws[:, 1, 0]
        Ws /= 2
        Ws = torch.tensor(Ws, dtype=torch.float32)

        score = E @ (Ws @ U.T)

        assert (score[:, 0, 0] - score[:, 0, 1] == torch.tensor(gamma_0.flatten(), dtype=torch.float32)).all()
        if alpha == 0:
            assert (score[:, 1, 0] - score[:, 1, 1] == torch.tensor(gamma_1.flatten(), dtype=torch.float32)).all()

        log_likelihood = F.log_softmax(score, dim=2)
        log_likelihood = log_likelihood[:, all_x, all_y]
        train_loss = (log_likelihood * (-proba)).mean(dim=1)
        Z = train_loss.numpy()
        fig, ax = plt.subplots(figsize=(0.2 * WIDTH, 0.2 * HEIGHT))
        c = ax.contour(gamma_0, gamma_1, Z.reshape(num, num), levels=20, colors="k", linewidths=0.5, linestyles="solid")

        pred = score.argmax(dim=2)
        pred = (pred != all_y).to(float)
        c = ax.contourf(gamma_0, gamma_1, (pred * proba).sum(dim=1).reshape((num, num)), cmap="Blues_r", alpha=0.75)
        # ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(rf"$\alpha={alpha}, p_1={p}$", fontsize=10)
        fig.savefig(SAVE_DIR / f"landscape_{sign}_{res}.pdf", bbox_inches="tight", pad_inches=0)
