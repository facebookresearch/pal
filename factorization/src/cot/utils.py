"""
Utils functions

License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

import os
import sys
from json import JSONEncoder
from pathlib import PosixPath

import torch

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------


def set_torch_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Json Serializer
# -----------------------------------------------------------------------------


class JsonEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, PosixPath):
            return str(obj)
        return super().default(obj)


# -----------------------------------------------------------------------------
# Slurm signal handling
# -----------------------------------------------------------------------------


def handle_sig(signum, frame):
    print(f"Requeuing after {signum}...", flush=True)
    os.system(f'scontrol requeue {os.environ["SLURM_ARRAY_JOB_ID"]}_{os.environ["SLURM_ARRAY_TASK_ID"]}')
    sys.exit(-1)


def handle_term(signum, frame):
    print("Received TERM.", flush=True)
