
# High level TODO
- Set up experimental pipeline.
- Come with a first set of experiments.
- Refine and Iterate experiments to get insights.

# Concrete stuff
- call random generator when creating random variable to be able to set a seed and reproduce the results.
- set up on a clean configuration for the data generation.
- clean the functions Sampler class: access easily to the real conditional proba, and easily generate random samples (makes the names clearer).


- Change the data generation code to allow for more complex bipartite graph.
    - Create a GraphFactorizedProbas.
    - Add a graph element (edges, connection...) to the SamplerConfig.
    - Change the Sampler.__init__ to take into consideration the changes to the SamplerConfig.
