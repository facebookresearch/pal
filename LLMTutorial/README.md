# Compositionality 

## Objective

Push a learning perspective linked with compositionality.

- Stay grounded by prioritizing experiments first.
- Stay focus by reasoning with and on LLMs first, notably through math tasks.
- Ultimatively we would like to shed lights on mechanisms that allow to learn and compose concepts, so to build better reasoning systems.

## Installation

The code requires Python 3.10+ (for `case` matching).
Here is some installation instruction:
- Install [miniconda](https://docs.conda.io/projects/miniconda/en/latest/).
- Install python in a new conda environment: be mindful to install a version of python that is compatible with PyTorch 2 (e.g., [PyTorch 2.0.1 requires python 3.10-](https://github.com/pytorch/pytorch/blob/2_0_fix_docs/torch/_dynamo/eval_frame.py#L377), and PyTorch 2.1 requires python 3.11- to use `torch.compile`).
```bash
$ conda create -n llm
$ conda activate llm
$ conda install python=3.11 pip
```
- Install Pytorch and check CUDA support: be mindful to install a version that is compatible with your CUDA driver ([example](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/)) (use `nvidia-smi` to check your CUDA driver)
```bash
$ pip install torch --index-url https://download.pytorch.org/whl/cu118
$ python -c "import torch; print(torch.cuda.is_available())"
True
```
- Install this repo
```bash
$ git clone <repo url>
$ cd <repo path>
$ pip install -e .
```