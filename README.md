# purkinje-learning-demo

This codes implements an algorithm to learn the Purkinje network of a patient, given a cardiac anatomy and a reference electrocardiogram (ECG). It is based on:

- The Purkinje network is modeled as a fractal tree, which depends on a set of geometrical parameters.
- The cardiac activation is obtained with the Eikonal equation. Then, with the activation times we compute the surface ECG.

With these models, the necessary steps to learn the network are:

1. A Bayesian optimization approach is used to find networks that produce ECGs similar to the reference.
2. With approximate Bayesian computation we estimate the posterior distribution of the Purkinje network parameters. Then, with rejection sampling we obtain samples from this distribution.

In [ECG_BO_demo.ipynb](./ECG_BO_demo.ipynb) you can run the Bayesian optimization and then find the posterior samples, here it uses a [simplified cardiac geometry](https://github.com/fgalvarez1/cardiac-demo). In [ECG_BO_results.ipynb](./ECG_BO_results.ipynb) you can see the simulation results: the ECGs of the posterior samples and a pairplot showing the parameters distributions.

