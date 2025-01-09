# PAL: Predictive Analysis & Laws for Neural Networks

Dismantling large language models parts to understand them better, with the hope of building better models.

## Installation
You can change the paths you want the codebase to operate with by modifying the `user_config.ini` file
```bash
git clone git@github.com:facebookresearch/pal.git
cd pal
pip install -e .
```
This will install the codebase in your current Python environment.
If you want to install it in a special environment, you can create a new one with `conda` or `virtualenv`.

## Usage
Please refer to the `example` folder, and in particular the `bash_script.sh` to get a sense of how to run experiments.

#### Research papers
- Vivien Cabannes, Charles Arnal, Wassim Bouaziz, Alice Yang, Francois Charton, Julia Kempe. *Iteration Head: A Mechanistic Study of Chain-of-Thought*, in NeurIPS 2024. The codebase is in the folder `projects/cot`.

- Vivien Cabannes, Elvis Dohmatob, Alberto Bietti. *Scaling Laws for Associative Memories*, in International Conference on Learning Representations (ICLR), 2024. The codebase is in the folder `projects/scaling_laws`.

- Vivien Cabannes, Berfin Simsek, Alberto Bietti. *Learning Associative Memories with Gradient Descent* in International Conference on Machine Learning (ICML), 2024. The codebase is in the folder `projects/gradient_descent`.

- Ambroise Odonnat, Wassim Bouaziz, Vivien Cabannes. *Easing Optimization Paths: A Circuit Perspective*, in ICASSP 2025. The codebase is in `project/visualization`.

- Ambroise Odonnat, Wassim Bouaziz, Vivien Cabannes. *A Visual Case Study of the Training Dynamics in Neural Networks*, In preparation. The codebase is in `project/visualization`.

- Charles Arnal, Clement Berenfeld, Simon Rosenberg, Vivien Cabannes. *Scaling Laws with Hidden Structure*, In preparation. The codebase is in `projects/factorization`.

## Organization
The main reusable code is in the `src` folder.
The code for our different research streams is in the `projects` folder.
Other folders may include:
- `data`: contains data used in the experiments.
- `models`: saves models' weights.
- `launchers`: contains bash scripts to launch experiments.
- `notebooks`: used for exploration and visualization.
- `scripts`: contains Python scripts to run experiments.
- `tests`: contains tests for the code.
- `tutorial`: contains tutorial notebooks to get started with LLMs' training.
