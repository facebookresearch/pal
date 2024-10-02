
# High level TODO
- Set up experimental pipeline.
- Come with a first set of experiments.
- Refine and Iterate experiments to get insights.

# Concrete stuff
- create a json config file for the experiments.
- script to look at the experiments.

- Change the data generation code to allow for more complex bipartite graph.
    - Create a GraphFactorizedProbas.
    - Add a graph element (edges, connection...) to the SamplerConfig.
    - Change the Sampler.__init__ to take into consideration the changes to the SamplerConfig.


Theoretical questions:
- Do we want to test for generalization, or only for memorization ability?
- If one learn about the factorization, then one can generalize to combination of factors it has not seen before. Then we have to be careful of the split between training and testing.


There is two different questions:
One is about the fact that a MLP can leverage factorization to be more parameter efficient (it can learn to store the full information about the relation between x and y with less parameters)

The second question is about the fact that a factorization structure allows us to circumvent the curse of dimensionality. This is about the generalization ability of the model.
