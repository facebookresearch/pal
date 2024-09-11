import argparse
import copy
import torch
import os
import uuid

import numpy as np

from dataclasses import asdict
from itertools import product
from pathlib import Path

import csv
import json

import numpy as np

from pathlib import PosixPath


@dataclass
class TrainConfig:
    data_seed: int
    n_samples: int
    n_dim: int
    noise_cov: float

    n_epochs: int
    batch_size: int
    lr: float

    model_seed: int
    p: float
    n_layers: int
    n_hidden_dim: int
    activation: str

    device: str
    output_dir: Path


def save_to_csv(data, filename):
    if isinstance(data, dict):
        fieldnames = list(data.keys())
    else:
        fieldnames = list(data[0].keys())

    try:
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)
    except Exception:
        with open(filename, 'w') as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
            writer.writeheader()
    # Add row for this experiment
    with open(filename, 'a') as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
        if isinstance(data, dict):
            writer.writerow(data)
        else:
            writer.writerows(data)


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, PosixPath):
            return obj.as_posix()
        elif isinstance(obj, np.integer):
            return int(obj)
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, cls=CustomEncoder)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, help='ID of the task')
    parser.add_argument('--array_size', type=int,
                        help='Number of tasks in the job array')
    args = parser.parse_args()

    config = TrainConfig(
        data_seed=0,
        n_samples=1000,
        n_dim=10,
        noise_cov=0.1,

        n_epochs=250,
        batch_size=32,
        lr=3e-3,

        model_seed=0,
        p=0.5,
        n_layers=5,
        n_hidden_dim=64,
        activation='relu',

        device='cuda' if torch.cuda.is_available() else 'cpu',
        output_dir=Path('outputs/Mk6')
    )

    os.makedirs(config.output_dir, exist_ok=True)

    sweep = {
        'p': np.linspace(0.0, 1.0, 6),
        'lr': [3e-3],
        'activation': ['relu', 'id'],
        'data_seed': np.random.randint(2**20, size=10),
        'model_seed': np.random.randint(2**20, size=10),
        'n_dim': [10, 100, 1000],
        'noise_cov': [0.01, 0.03, 0.1, 0.3],
        # 'n_hidden_dim': [32, 64, 128, 256],
        # 'n_layers': [5, 7, 9, 11],
    }

    sweep_list = product(*sweep.values())
    for i, sweep_params in enumerate(sweep_list):
        if (i % args.array_size) != (args.task % args.array_size):
            continue

        cloned_config = copy.deepcopy(config)

        cloned_config.output_dir = config.output_dir / str(uuid.uuid4())
        created = 0
        while not created:
            try:
                os.makedirs(cloned_config.output_dir, exist_ok=False)
                created = 1
            except FileExistsError:
                cloned_config.output_dir = config.output_dir / str(uuid.uuid4())

        for k, v in zip(sweep.keys(), sweep_params):
            setattr(cloned_config, k, v)

        save_to_csv({
            'task_id': args.task,
            **asdict(cloned_config),
            },
            config.output_dir / 'db.csv')

        train(cloned_config)


if __name__ == '__main__':
    main()
