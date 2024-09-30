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
from matplotlib import colormaps, rc
from scipy.linalg import eigh

sys.path.append(".")
from config import SAVE_DIR
from model import AssociativeMemory, get_embeddings

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
alpha = 0.95  # angle between the two tokens
nb_epoch = 35


def f(x, epsilon=0):
    return x


# data
all_x = torch.arange(n)
all_y = f(all_x)
proba = torch.tensor([p, 1 - p])
U = torch.eye(n)

E = torch.eye(n)
E[1, 0] = alpha
E[1, 1] = np.sqrt(1 - alpha**2)
model = AssociativeMemory(E, U)


lim = 10
num = 50
gamma_0, gamma_1 = np.meshgrid(np.linspace(-1, 8, num=num), np.linspace(-17, 10, num=num))

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

_gamma_0 = Ws[:, 0, 0] - Ws[:, 0, 1]
_gamma_1 = Ws[:, 1, 0] - Ws[:, 1, 1]
assert (torch.abs(_gamma_0 - gamma_0.flatten()) < 1e-6).all()
assert (torch.abs(_gamma_1 - gamma_1.flatten()) < 1e-6).all()

log_likelihood = F.log_softmax(score, dim=2)
log_likelihood = log_likelihood[:, all_x, all_y]
train_loss = (log_likelihood * (-proba)).mean(dim=1)
Z_train = train_loss.numpy()

pred = score.argmax(dim=2)
pred = (pred != all_y).to(float)
Z_accuracy = (proba * pred).sum(dim=1)

gammas_0 = []
gammas_1 = []
all_losses = []
all_accuracies = []

for lr in [10, 1]:
    W_t = torch.zeros((nb_epoch, d, d))
    losses = torch.zeros(nb_epoch)
    accuracy = torch.zeros(nb_epoch)

    model.W.data[:] = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0)

    for i in range(nb_epoch):
        W_t[i] = model.W.detach()
        # get batch of data
        # x = torch.multinomial(proba, batch_size, replacement=True)
        # y = f(x)

        # full batch
        x = all_x
        y = all_y

        # compute loss
        score = model(x)
        loss = (proba * F.cross_entropy(score, y, reduction="none")).sum()

        # record statistics
        losses[i] = loss.item()
        with torch.no_grad():
            pred = model.fit(all_x)
            accuracy[i] = proba[pred == all_y].sum().item()

        # update parameters with gradient descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
    gammas_0.append((W_t[:, 0, 0] - W_t[:, 0, 1]).numpy())
    gammas_1.append((W_t[:, 1, 0] - W_t[:, 1, 1]).numpy())
    all_losses.append(losses)
    all_accuracies.append(accuracy)


fig, ax = plt.subplots(figsize=(0.2 * WIDTH, 0.2 * HEIGHT))
c = ax.contour(gamma_0, gamma_1, Z_train.reshape(num, num), levels=20, colors="k", linewidths=0.5, linestyles="--")
c = ax.contourf(gamma_0, gamma_1, Z_accuracy.reshape((num, num)), cmap="Blues_r", alpha=0.5)

leg = []
for i in range(2):
    (a,) = ax.plot(
        gammas_0[i],
        gammas_1[i],
        color={0: "C2", 1: "C3"}[i],
        linewidth={0: 1, 1: 1}[i],
        linestyle={0: "solid", 1: "solid"}[i],
    )
    leg.append(a)
ax.legend(leg, [r"$\eta=10$", r"$\eta=1$"], loc="upper right", fontsize=6, frameon=True, ncol=2)
ax.set_title(rf"$\alpha={alpha}, p_1={p}$", fontsize=10)
fig.savefig(SAVE_DIR / "spike_trajectory.pdf", bbox_inches="tight", pad_inches=0)

fig, ax = plt.subplots(figsize=(0.2 * WIDTH, 0.2 * HEIGHT))
(a,) = ax.plot(all_losses[0], color="C2", linewidth=1)
ax.plot(all_losses[1], color="C3", linewidth=1)
(b,) = ax.plot(1 - all_accuracies[0], color="C2", linewidth=1, linestyle="--")
ax.plot(1 - all_accuracies[1], color="C3", linewidth=1, linestyle="--")
ax.legend([a, b], [r"${\cal L}(W_t)$", r"${\cal L}_{01}(W_t)$"], loc="upper right", fontsize=6, frameon=True, ncol=1)
ax.set_title(r"$t\to {\cal L}(W_t)$", fontsize=10)
fig.savefig(SAVE_DIR / "spike_loss.pdf", bbox_inches="tight", pad_inches=0)
