"""
Initialization of the MLP layers.

License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

import math
import mup
import torch.nn as nn
from mup import set_base_shapes


def _mup_init(layer):
    mup.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
    if layer.bias is not None:
        fan_in, _ = mup.init._calculate_fan_in_and_fan_out(layer.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        mup.init.uniform_(layer.bias, -bound, bound)


def mup_init_mlp(model):
    # some base_model can be defined as base
    set_base_shapes(model=model, base=None)
    _mup_init(model.mlp.fc1)
    _mup_init(model.mlp.fc2)


def _torch_init(layer):
    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
    if layer.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(layer.bias, -bound, bound)


def torch_init_mlp(model):
    _torch_init(model.mlp.fc1)
    _torch_init(model.mlp.fc2)
