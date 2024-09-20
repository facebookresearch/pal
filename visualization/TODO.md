
## TODO visualization

- Fine-tuning
    - [ ] Write visualization when $p=3$ with triangles for the sequence associated to $y = 2$.

- Write special train script to put in launchers/ regarding the experiements run with a fixed seed seed=89

- Analyze final configurations to distinguish between different final solutions.
- Analyze the different videos.
    - why and how do we get to different solutions?
    - why and how do we get loss spikes?
- Write the paper.
- Get nice plots for the paper.

Do some scripts to emulate the following:
 - first runs without saving weights
 - iteration over the different configurations
    - recover test losses
    - if not good, remove the configuration (or split into winning config and failling config).
- then relaunch training to save the weights for the one that have worked.

Same for the finetuning only launch from successful runs.
