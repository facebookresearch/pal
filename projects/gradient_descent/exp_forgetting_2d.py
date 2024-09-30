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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import rc

sys.path.append(".")
from config import SAVE_DIR
from model import AssociativeMemory, get_embeddings

os.makedirs(SAVE_DIR, exist_ok=True)

WIDTH = 8.5  # inches (from ICML style file)
HEIGHT = 8.5 / 1.618  # golden ratio

WIDTH *= 2 / 1.8
HEIGHT *= 2 / 1.8

rc("font", family="serif", size=8)
usetex = not subprocess.run(["which", "pdflatex"]).returncode
rc("text", usetex=usetex)
if usetex:
    rc("text.latex", preamble=r"\usepackage{times}")


seeds = [39, 81, 7, 9]
ns = [3, 3, 30, 30]
x_mins = [-10, -2, -5, -1]
x_maxs = [10, 2, 5, 0.5]
y_mins = [-10, -2, -5, -0.5]
y_maxs = [10, 2, 5, 1.5]

# x_mins = [-10, -2, -1.5, -1]
# x_maxs = [10, 2, -.5, .5]
# y_mins = [-10, -2, 1, -.5]
# y_maxs = [10, 2, 3, 1.5]

# hyparameters
d = 2


def f(x, epsilon=0):
    return x % 2


for seed, n, x_min, x_max, y_min, y_max in zip(seeds, ns, x_mins, x_maxs, y_mins, y_maxs):
    torch.manual_seed(seed)

    # data
    all_x = torch.arange(n)
    all_y = f(all_x)

    alpha = 1
    proba = (all_x + 1.0) ** (-alpha)
    proba /= proba.sum()

    E = get_embeddings(all_x.max() + 1, d, norm=True)
    U = torch.eye(d)

    model = AssociativeMemory(E, U)

    alpha = E[1:] @ E[0]

    num = 100
    gamma_0, gamma_1 = np.meshgrid(np.linspace(x_min, x_max, num=num), np.linspace(y_min, y_max, num=num))

    # lim = 1
    # gamma_0, gamma_1 = np.meshgrid(np.linspace(-lim, lim, num=num), np.linspace(-lim, lim, num=num))

    Ws = np.zeros((num * num, d, d))
    Ws[:, 0, 0] = gamma_0.flatten()
    Ws[:, 1, 0] = gamma_1.flatten()
    Ws[:, 0, 1] = -Ws[:, 0, 0]
    Ws[:, 1, 1] = -Ws[:, 1, 0]
    Ws /= 2
    Ws = torch.tensor(Ws, dtype=torch.float32)

    score = E @ (Ws @ U.T)
    log_likelihood = F.log_softmax(score, dim=2)
    log_likelihood = log_likelihood[:, all_x, all_y]
    train_loss = (log_likelihood * (-proba)).mean(dim=1)
    Z_train = train_loss.numpy()

    pred = score.argmax(dim=2)
    pred = (pred != all_y).to(float)
    Z_accuracy = (proba * pred).sum(dim=1)

    fig, ax = plt.subplots(figsize=(0.18 * WIDTH, 0.18 * HEIGHT))
    c = ax.contour(
        gamma_0, gamma_1, Z_train.reshape(num, num), levels=20, colors="k", linewidths=0.5, linestyles="solid"
    )
    c = ax.contourf(gamma_0, gamma_1, Z_accuracy.reshape((num, num)), levels=20, cmap="Blues_r", alpha=0.75)

    i = np.argmin(Z_train)
    acc = Z_accuracy[i].item()

    Z_accuracy[np.abs(gamma_0.flatten()) < 1] = 1
    Z_accuracy[np.abs(gamma_1.flatten()) < 1] = 1
    j = np.argmin(Z_accuracy.numpy())
    acc_min = Z_accuracy[j].item()
    ax.set_title(rf"$N={n}, " + "{\cal E}=" + f"{int(100 * (acc - acc_min)) / 100}$", fontsize=10)
    if n == 30 or (n == 3 and seed == 81):
        ax.scatter(gamma_0.flatten()[i], gamma_1.flatten()[i], c="r", s=10)

    plt.savefig(SAVE_DIR / f"forgetting_2d_{n}_{seed}.pdf", bbox_inches="tight", pad_inches=0)


n = 10
lim = 10

seed = 11

torch.manual_seed(seed)

# data
all_x = torch.arange(n)
all_y = f(all_x)

alpha = 2
proba = (all_x + 1.0) ** (-alpha)
proba /= proba.sum()

E = get_embeddings(all_x.max() + 1, d, norm=True)
U = torch.eye(d)

model = AssociativeMemory(E, U)

num = 100
gamma_0, gamma_1 = np.meshgrid(np.linspace(-lim, lim, num=num), np.linspace(-lim, lim, num=num))

Ws = np.zeros((num * num, d, d))
Ws[:, 0, 0] = gamma_0.flatten()
Ws[:, 1, 0] = gamma_1.flatten()
Ws[:, 0, 1] = -Ws[:, 0, 0]
Ws[:, 1, 1] = -Ws[:, 1, 0]
Ws /= 2
Ws = torch.tensor(Ws, dtype=torch.float32)

score = E @ (Ws @ U.T)
log_likelihood = F.log_softmax(score, dim=2)
log_likelihood = log_likelihood[:, all_x, all_y]
train_loss = (log_likelihood * (-proba)).mean(dim=1)
Z_train = train_loss.numpy()

pred = score.argmax(dim=2)
pred = (pred != all_y).to(float)
Z_accuracy = (proba * pred).sum(dim=1)

i = np.argmin(Z_train)

batch_size = 10
nb_epoch = 1000
lrs = [1e-1]

gammas_0 = []
gammas_1 = []
all_losses = []
all_accuracies = []

for lr in lrs:
    W_t = torch.zeros((nb_epoch, d, d))
    losses = torch.zeros(nb_epoch)
    accuracy = torch.zeros(nb_epoch)

    model.W.data[:] = torch.randn(d, d)
    model.W.data[0, 0] = 7.5
    model.W.data[1, 0] = 2.5
    model.W.data[0, 1] = -7.5
    model.W.data[1, 1] = -2.5
    model.W.data[:] /= 2
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0)

    for i in range(nb_epoch):
        W_t[i] = model.W.detach()
        # get batch of data
        x = torch.multinomial(proba, batch_size, replacement=True)
        y = f(x)

        # # full batch
        # x = all_x
        # y = all_y

        # compute loss
        score = model(x)
        loss = F.cross_entropy(score, y)

        # record statistics
        with torch.no_grad():
            pred = model.fit(all_x)
            accuracy[i] = proba[pred == all_y].sum().item()
            score = model(all_x)
            losses[i] = (proba * F.cross_entropy(score, all_y, reduction="none")).sum().item()

        # update parameters with gradient descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
    gammas_0.append((W_t[:, 0, 0] - W_t[:, 0, 1]).numpy())
    gammas_1.append((W_t[:, 1, 0] - W_t[:, 1, 1]).numpy())
    all_losses.append(losses)
    all_accuracies.append(accuracy)

fig, ax = plt.subplots(figsize=(0.18 * WIDTH, 0.18 * HEIGHT))
c = ax.contour(gamma_0, gamma_1, Z_train.reshape(num, num), levels=10, colors="k", linewidths=0.5, linestyles="--")
c = ax.contourf(gamma_0, gamma_1, Z_accuracy.reshape((num, num)), cmap="Blues_r", alpha=0.5)

leg = []
for i in range(len(lrs)):
    (a,) = ax.plot(gammas_0[i], gammas_1[i], color="C" + str(i + 1), linewidth=1, linestyle="solid")
    leg.append(a)
ax.legend(leg, [rf"$\eta={lrs[i]}$" for i in range(len(lrs))], loc="lower right", fontsize=6, frameon=True, ncol=2)

i = np.argmin(Z_train)
ax.scatter(gamma_0.flatten()[i], gamma_1.flatten()[i], c="r", s=10)
# ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r"SGD trajectory")
fig.savefig(SAVE_DIR / "forgetting_trajectory.pdf", bbox_inches="tight", pad_inches=0)

fig, ax = plt.subplots(figsize=(0.18 * WIDTH, 0.18 * HEIGHT))
for i in range(len(lrs)):
    (a,) = ax.plot(all_losses[i] / 2, color="C" + str(i + 1), linewidth=1, alpha=1)
    (b,) = ax.plot(1 - all_accuracies[i], color="C" + str(i + 1), linewidth=1, linestyle="--")
ax.legend(
    [a, b], [r"${\cal L}(W_t) / 2$", r"${\cal L}_{01}(W_t)$"], loc="upper right", fontsize=6, frameon=True, ncol=1
)
ax.set_title(r"$t\to {\cal L}(W_t)$")
fig.savefig(SAVE_DIR / "forgetting.pdf", bbox_inches="tight", pad_inches=0)
