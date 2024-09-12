# %% Imports

import copy
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from factorization.models.softmax_model import Model

sys.path.append(str(Path("..").resolve()))


SAVE_DIR = Path(".").resolve() / "results"
DEVICE = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
SEED = None
SEED = 200
if SEED:
    RNG = np.random.default_rng(SEED)
    np.random.seed(seed=SEED)
    torch.manual_seed(seed=SEED)


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

SAVE_DIR.mkdir(exist_ok=True, parents=True)
pickle.dump(weights, open(SAVE_DIR / "weights.pkl", "wb"))
pickle.dump(losses, open(SAVE_DIR / "losses.pkl", "wb"))
pickle.dump(test_losses, open(SAVE_DIR / "test_losses.pkl", "wb"))
pickle.dump(accs, open(SAVE_DIR / "accs.pkl", "wb"))
pickle.dump(test_accs, open(SAVE_DIR / "test_accs.pkl", "wb"))

# %%
