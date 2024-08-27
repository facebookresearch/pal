# Compositionality 

## TODOS

For NeurIPS:
- Redo the positional embedding study for the non-permutation invariant task.

Longer term implementation TODO:
-Logging
    - Good logging and referential for experiments (maybe wandb).
    - log the path to the stderr file linked with an experiment (get the sbatch %a, %j as an argument to the python script).
- Unit tests
- Efficiency
    - Be mindful of useless copy and CPU/GPU transfert of the data (e.g., in the evaluation script).
- Small fixes
    - deal with error when load_checkpoint=True, but there is no checkpoint.
- Submitit related
    - Look into relaunching jobs when they arrive at termination + saving codebase that has launch a run
    - system to save a screenshot of the code before launching an experiment (clean conda env, and install current version of the code with pip install .).
- Thicker codebase
    - A code base with TinyStory.

- Future Investigation:
    - Curriculum learning
    - Generic progress measure
    - What happen when the loss plateau
    - The structure of the learned embedding when the task has a strong geometrical structure
    - The effect of Adam and Layer norm.

## Disorganized thoughts 
The following is not really well organized, but it gives food for thoughts. We should focus on some meaningful experiments that are insightful beyond our toy setup.

Research TODO:
- Look at skill transfert between binary problem and copying problem / look a bit into curriculum... / look also at a mix of data / also mix of length.
- Look at how the substraction of position embeddings is performed by the first MLP / look at the position embeddings (can we freeze them if the embedding dimension is big enough).
- Look at how the state updates ($s = F(s, x)$) is performed by the second MLP when we solve for the parity problem.
- Look at how the work get shared across heads when we have too much capacity, and how we might disambiguate the work done if we know the right circuit.

Simple questions:
- does continuing training make the attention maps cleaner while the accuracy does not change? If yes, we can make a link with grokking and emergence of more functorial pattern (link with sparsity induced bias with SGD - Loucas paper).
- empirical scaling law with respect sequence length, and embedding dimension (and how clean the attention matrices are) / look at optimal data mix.
- check if curriculum change what is learned (i.e., are the iteration heads learned if we only solve the parity problem?)
- ablation with respect to the batch size
- investigate the small weird stuff on the attention map: it seems that there should be a different circuit hidden for a special position.
- if we do a mix parity and binary, maybe we will force more to have the circuit we want
- if we reduce the position embedding dimension, maybe we will have a cleaner structure appearing in the position embedding, that allows to find a rule for substraction, rather than memorize all substraction.
- reuse Wes code (or submitit) to save the codebase before running a script, to avoid that changes to the codebase modify runs that are pending but have not started to run. Look also at wandb and hydra.

## Objective

Understand efficient machine learning, first by focusing on "linguistic rules" found by LLMs, later by studying how math could be solved with AI.

- Ultimatively we would like to shed lights on mechanisms that allow to learn and compose concepts, so to build better reasoning systems.
- We stay grounded by prioritizing experiments first.
- We stay focused by reasoning with and on LLMs first, notably through math tasks.

Code organization:
- we will keep several sub-repo for different projects, with the aim to have some transversal tools for all of them (experimental setup, models, progress measures...)

TODo:
- we need to write a small notes on what are the linguistic rules that we believe LLMs implement: compositionality, Zeillig Harris, ...

## Installation

The code requires Python 3.10+ (for `case` matching).
Here is some installation instruction:
- Install [miniconda](https://docs.conda.io/projects/miniconda/en/latest/).
- Install python in a new conda environment: be mindful to install a version of python that is compatible with PyTorch 2 (e.g., [PyTorch 2.0.1 requires python 3.10-](https://github.com/pytorch/pytorch/blob/2_0_fix_docs/torch/_dynamo/eval_frame.py#L377), and PyTorch 2.1 requires python 3.11- to use `torch.compile`).
```bash
conda create -n llm
conda activate llm
conda install pip
```
- Install Pytorch and check CUDA support: be mindful to install a version that is compatible with your CUDA driver ([example](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/)) (use `nvidia-smi` to check your CUDA driver)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(torch.cuda.is_available())"
True
```
- Install this repo
```bash
git clone <repo url>
cd <repo path>
pip install -e .
```

## Development
For formatting, I recommand using `black`, `flake8`, and `isort`.
Consider automatic formatting when saving files (easy to setup in VSCode, ask ChatGPT to get set up if not confortable with VSCode configuration).

## Organization
The main code is in the `src` folder.
Other folders include:
- `data`: contains data used in the experiments.
- `launchers`: contains bash scripts to launch experiments
- `models`: saves models' weights.
- `notebooks`: used for exploration and visualization.
- `scripts`: contains python scripts to run experiments.
- `tests`: contains tests for the code.
- `tutorial`: contains tutorial notebooks to get started with LLMs' training.
