# causal_discovery_via_partitioning

<p align="center" markdown="1">
    <img src="https://img.shields.io/badge/Python-3.8-blue.svg" alt="Python Version" height="18">
    <a href="https://arxiv.org/abs/2406.06348"><img src="https://img.shields.io/badge/arXiv-2406.06348-green)" alt="arXiv" height="18"></a>
</p>

<p align="center">
  <a href="#setup">Setup</a> •
  <a href="#usage">Usage</a> •
  <a href="#citation">Citation</a> •
  <a href="https://shahashka.github.io/causal_discovery_via_partitioning/" target="_blank">Documentation</a>
</p>
This is an implementation of a divide-and-conquer framework for causal discovery using a novel causal partition from the paper "Causal Discovery over High-Dimensional Structured Hypothesis Spaces with Causal Graph Partitioning" published in TMLR March, 2025.

A causal partition is a graph partition of a hypothesis space, defined by a superstructure, into overlapping variable sets. A causal partition allows for merging locally estimated graphs without an additional learning step, and provably recovers the Markov Equivalence Class of the true DAG. We can efficiently create a causal partition from any disjoint partition. This means that a causal partition can be an extension to any graph partitioning algorithm.

# Setup
This repository has an R dependency because we use causal discovery algorithms from the R package ```pcalg```. As a result, installation is slightly more complicated than usual. To assist with this, we provide a docker image that can be downloaded and installed as follows:
```bash
docker pull ghcr.io/shahashka/causal_discovery_via_partitioning:main
```
# Usage
See our [tutorial](examples/tutorial.py) for an example of usage. To execute this code, run the following:

```bash
docker run --rm -it ghcr.io/shahashka/causal_discovery_via_partitioning:main python examples/tutorial.py
```
To setup large scale experiments with parameter sweeps over the causal learning algorithms, partitioning algorithms or other parameters, use our ```Experiment``` class which easily enables parallel runs for each configuration! See an example [experiment](simulations/experiment_1_sample_sweep.py).
# Citation

If you find our study helpful, please consider citing us as:

```
@article{shah2025causal,
  title = {Causal Discovery over High-Dimensional Structured Hypothesis Spaces with Causal Graph Partitioning},
  author = {Shah, Ashka and DePavia, Adela and Hudson, Nathaniel and Foster, Ian and Stevens, Rick},
  journal = {Transactions on Machine Learning Research (TMLR)},
  year = {2025},
}

```
