# %% Imports

import copy
import logging
import pickle
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from factorization.config import SAVE_DIR
from factorization.data.modular import DataloaderConfig, SMADataloader
from factorization.models.softmax_model import Model

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()]
)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# %% Utils


def copy_weights(model):
    if model.output.weight.device == torch.device("cpu"):
        return {k: copy.deepcopy(v) for k, v in model.state_dict().items()}
    else:
        return {k: v.cpu().detach() for k, v in model.state_dict().items()}


@dataclass
class ExperimentConfig:
    # data
    vocab_size: int = 2
    seq_length: int = 12
    sparsity_index: int = 5
    nb_data: int = 2048

    # optimization
    batch_size: int = None
    nb_epochs: int = 4_000
    lambda_l1: float = 1e-4
    lr: float = 1e-2

    # model
    emb_dim: int = 2
    nb_emb: int = None
    ffn_dim: int = None
    ffn_bias: bool = True
    ffn_dropout: float = 0
    activation: float = "gelu"

    def __post_init__(self):
        if self.batch_size is None:
            self.batch_size = self.nb_data
        if self.nb_emb is None:
            self.nb_emb = self.vocab_size
        if self.ffn_dim is None:
            self.ffn_dim = 4 * self.emb_dim


# %% Main function


def run_experiments(
    seed: int = None,
    save_weights: bool = False,
    vocab_size: int = 2,
    seq_length: int = 12,
    sparsity_index: int = 5,
    nb_data: int = 2048,
    batch_size: int = None,
    nb_epochs: int = 4_000,
    lambda_l1: float = 1e-4,
    lr: float = 1e-2,
    emb_dim: int = 2,
    nb_emb: int = None,
    ffn_dim: int = 10,
    ffn_bias: bool = True,
    ffn_dropout: float = 0,
    activation: float = "gelu",
):
    if seed:
        RNG = np.random.default_rng(seed)
        np.random.seed(seed=seed)
        torch.manual_seed(seed=seed)

    config = ExperimentConfig(
        vocab_size=vocab_size,
        seq_length=seq_length,
        sparsity_index=sparsity_index,
        nb_data=nb_data,
        batch_size=batch_size,
        nb_epochs=nb_epochs,
        lambda_l1=lambda_l1,
        lr=lr,
        emb_dim=emb_dim,
        nb_emb=nb_emb,
        ffn_dim=ffn_dim,
        ffn_bias=ffn_bias,
        ffn_dropout=ffn_dropout,
        activation=activation,
    )

    # Data
    data_config = DataloaderConfig(
        vocab_size=config.vocab_size,
        seq_length=config.seq_length,
        sparsity_index=config.sparsity_index,
        nb_data=config.nb_data,
        batch_size=config.batch_size,
        mode="train",
    )
    train_loader = SMADataloader(data_config, rng=RNG, device=DEVICE)
    data_config.mode = "test"
    test_loader = SMADataloader(data_config, rng=RNG, device=DEVICE)

    logger.info(f"Training set: {train_loader.dataset}")
    logger.info(f"Testing set: {test_loader.dataset}")

    # Model

    tmp = config.vocab_size
    config.vocab_size = config.nb_emb
    model = Model(config)
    config.vocab_size = tmp
    model.to(device=DEVICE)
    logger.info(f"Model with {sum(p.numel() for p in model.parameters())} parameters, running on {DEVICE}")

    # Training Loop

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    losses = torch.zeros(config.nb_epochs)
    test_losses = torch.zeros(config.nb_epochs)
    accs = torch.zeros(config.nb_epochs)
    test_accs = torch.zeros(config.nb_epochs)

    if save_weights:
        weights = [copy_weights(model)]

    X_train = train_loader.dataset.data
    Y_train = train_loader.dataset.targets
    X_test = test_loader.dataset.data
    Y_test = test_loader.dataset.targets

    for epoch in (bar := tqdm(range(config.nb_epochs))):

        for X, Y in train_loader:
            # X = X.to(device=DEVICE)
            # Y = Y.to(device=DEVICE)

            optimizer.zero_grad()

            # forward
            score = model(X, verbose=False)
            loss = F.cross_entropy(score, Y)
            reg_loss = config.lambda_l1 * sum(p.abs().sum() for p in model.parameters())

            # backward
            loss.backward()
            reg_loss.backward()
            optimizer.step()

        # record statistics
        with torch.no_grad():
            score = model(X_train, verbose=False)
            loss = F.cross_entropy(score, Y_train)
            losses[epoch] = loss.item()
            accs[epoch] = (score.argmax(-1) == Y_train).float().mean()
            score_test = model(X_test)
            test_losses[epoch] = F.cross_entropy(score_test, Y_test)
            test_accs[epoch] = (score_test.argmax(-1) == Y_test).float().mean()
            if save_weights:
                weights.append(copy_weights(model))

        bar.set_postfix(loss=losses[epoch].item(), acc=accs[epoch].item(), test_acc=test_accs[epoch].item())

    # Saving results

    SAVE_DIR.mkdir(exist_ok=True, parents=True)
    pickle.dump(losses, open(SAVE_DIR / "losses.pkl", "wb"))
    pickle.dump(test_losses, open(SAVE_DIR / "test_losses.pkl", "wb"))
    pickle.dump(accs, open(SAVE_DIR / "accs.pkl", "wb"))
    pickle.dump(test_accs, open(SAVE_DIR / "test_accs.pkl", "wb"))
    if save_weights:
        pickle.dump(weights, open(SAVE_DIR / "weights.pkl", "wb"))


# %% CLI

if __name__ == "__main__":
    import fire

    fire.Fire(run_experiments)
