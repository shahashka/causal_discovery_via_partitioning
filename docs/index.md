# Causal Discovery via Partitioning

This page serves as the documentation for the code underlying the following paper on
causal discovery:

## "Causal Discovery over High-Dimensional Structured Hypothesis Spaces with Causal Graph Partitioning"

### Abstract

The aim in many sciences is to understand the mechanisms that underlie the
observed distribution of variables, starting from a set of initial hypotheses.
Causal discovery allows us to infer mechanisms as sets of cause and effect
relationships in a generalized way – without necessarily tailoring to a specific
domain. Causal discovery algorithms search over a structured hypothesis space,
defined by the set of directed acyclic graphs, to find the graph that best explains
the data. For high-dimensional problems, however, this search becomes intractable
and scalable algorithms for causal discovery are needed to bridge the gap. In this
paper, we define a novel causal graph partition that allows for divide-and-conquer
causal discovery with theoretical guarantees. We leverage the idea of a superstructure
– a set of learned or existing candidate hypotheses – to partition the search space.
We prove under certain assumptions that learning with a causal graph partition always
yields the Markov Equivalence Class of the true causal graph. We show our algorithm
achieves comparable accuracy and a faster time to solution for biologically-tuned
synthetic networks and networks up to $10^4$ variables. This makes our method applicable
to gene regulatory network inference and other domains with high-dimensional structured
hypothesis spaces.

### Authors

- **Ashka Shah** (_University of Chicago_)
- **Adela DePavia** (_University of Chicago_)
- **Nathaniel Hudson** (_University of Chicago_, _Argonne National Laboratory_)
- **Ian Foster** (_University of Chicago_, _Argonne National Laboratory_)
- **Rick Stevens** (_University of Chicago_, _Argonne National Laboratory_)

### Citing Our Work

If you use any code or outputs of this work, please cite the paper via the following BibTeX:

```bibtex
@article{shah2025causal,
  title = {Causal Discovery over High-Dimensional Structured Hypothesis Spaces with Causal Graph Partitioning},
  author = {Shah, Ashka and DePavia, Adela and Hudson, Nathaniel and Foster, Ian and Stevens, Rick},
  journal = {Transactions on Machine Learning Research (TMLR)},
  year = {2025},
}
```
