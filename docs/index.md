# Causal Discovery with Scalable Graph Partitioning

This repository contains the code for our novel causal discovery algorithm, designed
to handle high-dimensional problems efficiently. Causal discovery involves identifying
cause-and-effect relationships in data, typically by searching over *directed acyclic
graphs* (DAGs). For large datasets, this process becomes computationally intractable.

Our approach introduces a **causal graph partitioning** method that allows for
divide-and-conquer strategies with theoretical guarantees. By leveraging a
**superstructure**—a set of learned or predefined candidate hypotheses—we can partition
the search space and significantly speed up discovery while preserving accuracy.

The algorithm is validated on synthetic biological networks and scales to networks
with up to $10^4$ variables, making it suitable for complex tasks like gene regulatory
network inference.

## Getting Started

Simulations with our code can be run through the CLI as follows:
```bash
python -m causal_partitioning.simulate \
    --num-variables 100 \
    --num-samples 1000 \
    --num-edges 10 \
    --num-partitions 10 \
    --num-iterations 100 \
    --output-dir ./output
```


## Citing Causal Partitioning
If you use our code, algorithms, or anything related to our work, we would greatly appreciate
if you cite the following [paper](https://arxiv.org/pdf/2406.06348):
```bibtex
@article{shah2024causal,
  title={Causal Discovery over High-Dimensional Structured Hypothesis Spaces with Causal Graph Partitioning},
  author={Shah, Ashka and DePavia, Adela and Hudson, Nathaniel and Foster, Ian and Stevens, Rick},
  journal={arXiv preprint arXiv:2406.06348},
  year={2024}
}
```
