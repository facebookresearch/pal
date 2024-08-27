# Getting started with LLMs

#### Objective

The goal of those notebooks is to have a clear project to get started with large language models.
A huge part of LLM is engineering to succeed to run things fast (which require a good profiler), on large cluster architecture (to parallelize runs, and  requeue them), to debug easily (have the right tools to look at what is happening).
There are many great libraries to accomplish many things.
We will try to look at things one by one to get a good understanding of the full pipeline.

#### Try it on a V100
Some new features of PyTorch relies on Triton, which is not available on Quadro GPUs.
To try V100, you can launch on interactive session on a V100 with:
```bash
srun --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --partition=devlab --time=1:00:00 --pty bash -i
```
