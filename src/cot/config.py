"""
Configuration file

License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

import logging
from pathlib import Path

import numpy as np

# -----------------------------------------------------------------------------
# Tokenizer
# -----------------------------------------------------------------------------

TOKEN_DICT = {
    "polynomial": 58,
    "parity": 59,
    "binary_copy": 60,
    "BoS": 61,
    "EoI": 62,
    "EoS": 63,
}

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent.parent / "data"
CHECK_DIR = Path(__file__).parent.parent.parent / "models"
SAVE_DIR = Path(__file__).parent.parent.parent / "savings"

# -----------------------------------------------------------------------------
# Random seed
# -----------------------------------------------------------------------------

RNG = np.random.default_rng(0)

# -----------------------------------------------------------------------------
# Logging information
# -----------------------------------------------------------------------------

logging_datefmt = "%m-%d %H:%M:%S"
logging_format = "{asctime} {levelname} [{filename}:{lineno}] {message}"
logging_level = logging.INFO
