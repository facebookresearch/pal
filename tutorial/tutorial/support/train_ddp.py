"""
License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

import os

import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mathllm.config import DATA_DIR
from mathllm.model.mlp import MLP, MLPConfig
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

torch.manual_seed(0)
torch.cuda.manual_seed(0)


def load_problem():
    problem = "single_base"
    save_dir = DATA_DIR / problem
    z = 60

    x_train = torch.load(save_dir / "x_train.pt")
    y_train = torch.load(save_dir / "y_train.pt").to(torch.long)
    x_test = torch.load(save_dir / "x_test.pt")
    y_test = torch.load(save_dir / "y_test.pt").to(torch.long)

    def get_input_repr(x):
        n = x.size(0)
        out = torch.zeros((n, 2 * z))
        out[torch.arange(n), x[:, 0]] = 1
        out[torch.arange(n), x[:, 1] + 60] = 1
        return out

    in_train = get_input_repr(x_train)
    in_test = get_input_repr(x_test)

    config = MLPConfig(
        fan_in=2 * z,
        hidden_dim=z,
        fan_out=z,
        activation="square",
        residual=False,
        dropout=0.3,
    )

    batch_size = 1024

    trainset = TensorDataset(in_train, y_train)
    testset = TensorDataset(in_test, y_test)

    save_period = 2
    n_epochs = 1000

    torch.manual_seed(0)
    # model = MLP(config)
    # optimizer = optim.SGD(model.parameters(), lr=1e1)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1-0.001)
    # return model, optimizer, scheduler, trainset, testset, batch_size, n_epochs, save_period
    return config, trainset, testset, batch_size, n_epochs, save_period


def ddp_setup(rank, world_size, backend="nccl"):
    """
    DDP setup

    Parameters
    ----------
    rank: int
        unique identifier of each process (typically: one process = one GPU)
    world_size: int
        total number of processes
    backend: str
        backend to use for distributed computation, default is 'nccl', option include 'gloo'
    """
    init_process_group(backend=backend, rank=rank, world_size=world_size)


def train(rank, world_size):
    ddp_setup(rank, world_size)

    config, trainset, testset, batch_size, n_epochs, save_period = load_problem()
    model = MLP(config)

    # model, optimizer, scheduler, trainset, testset, batch_size, n_epochs, save_period = load_problem()

    train_loader = DataLoader(
        trainset, pin_memory=True, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(trainset)
    )
    test_loader = DataLoader(
        testset, pin_memory=True, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(testset)
    )
    ddp_model = DDP(model.to(rank), device_ids=[rank])
    ddp_model = torch.compile(ddp_model)

    optimizer = optim.SGD(ddp_model.parameters(), lr=1e1)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1 - 0.001)

    losses = []
    test_losses = []

    for epoch in range(n_epochs):
        # print(f'GPU{rank} | Epoch {epoch}')
        train_loader.sampler.set_epoch(epoch)
        ddp_model.train()
        running_loss = i = 0
        for x, y in train_loader:
            x, y = x.to(rank), y.to(rank)
            optimizer.zero_grad()
            logit = ddp_model(x)
            loss = F.cross_entropy(logit, y)
            loss.backward()
            optimizer.step()
            running_loss *= i / (i + len(x))
            running_loss += loss.item() * (len(x) / (i + len(x)))
            i += len(x)
        scheduler.step()
        losses.append(running_loss)

        with torch.no_grad():
            ddp_model.eval()
            test_loss = i = 0
            for x, y in test_loader:
                x, y = x.to(rank), y.to(rank)
                pred = ddp_model(x).argmax(dim=1)
                test_loss *= i / (i + len(x))
                test_loss += (pred != y).float().mean().item() * (len(x) / (i + len(x)))
                i += len(x)
            test_losses.append(test_loss)

        # only have the first machine saving the model
        if epoch and (not epoch % save_period) and not rank:
            torch.save(ddp_model.module.state_dict(), f"checkpoint.pth")
            print(f"GPU{rank} | Epoch {epoch} | Loss {running_loss:.3f} | Test Loss {test_loss:.3f}")

    if not rank:
        torch.save(losses, "losses.pt")
        torch.save(test_losses, "test_losses.pt")
    destroy_process_group()


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    print("finish training")
