
## Get nice plots

iid study
- [x] show that speed depends on the sum of the q_i.
- [x] show that speed depends on the sum of the p_i -> it does not.

compression study
- [x] show the threshold for the embedding dimension compared to the sum of the q_i.
- [x] study if we go faster if we take a bigger embedding, or more layers / show that the gain disappear if we normalize for flops

generalization study
- [x] show that the generalization depends on the connectivity in the graph.
- [ ] show that the generalization depends on input factors.
- [ ] show that one layer is the best for generalization.

other study
- [ ] Faire apparaitre le sum p_i * q_i quelque part.
- [ ] show that we have not sheet with hyperparameters.
- [ ] show that the user can use our framework to compute other quantities they may found relevant.

## Logging
Add a way to save the job id, (together with the task id) to be able to easily investigate bugs.
