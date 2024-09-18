"""
Configuration file

License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

import subprocess
from configparser import ConfigParser
from pathlib import Path

import torch

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = Path(__file__).parents[2].resolve()


# Priority to user config
config: dict[str, dict] = ConfigParser()
config.read(ROOT_DIR / "default_config.ini")
config.read(ROOT_DIR / "user_config.ini")

# Paths
# Priority to absolute over relative paths
for name in config["Relative Paths"]:
    globals()[name.upper()] = ROOT_DIR / config["Relative Paths"][name]

for name in config["Absolute Paths"]:
    globals()[name.upper()] = Path(config["Absolute Paths"][name])


# Tex available
USETEX = not subprocess.run(["which", "pdflatex"], stdout=subprocess.DEVNULL).returncode
USETEX = False
