from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, TypeAlias

from numpy import ndarray
from pandas import DataFrame

Node: TypeAlias = Any
Edge: TypeAlias = tuple[Any, Any]

Partition: TypeAlias = dict[int, Iterable[Node]]


class TrueGraph(NamedTuple):
    """Data for a true graph with a list of nodes and a list of edges."""

    nodes: list[Node]
    """Nodes in the true graph."""

    edges: list[Edge]
    """Edges in the true graph."""


class Result(NamedTuple):
    """Result of an experimental run."""

    shd: int
    sid: int
    auc: float
    tpr: float
    fpr: float
    time: float


GraphKind = Literal[
    "erdos_renyi", "small_world", "scale_free", "hierarchical", "ecoli"
]
"""Kinds of graphs that are used for experimental results."""


@dataclass
class GeneratedGraph:
    """A sampled causal graph."""

    true_graph: TrueGraph
    samples: DataFrame
    default_partition: Partition
    bias: ndarray
    variance: ndarray

    def samples_to_numpy(self) -> ndarray:
        return self.samples.drop(columns=["target"]).to_numpy()

    @property
    def nodes(self) -> list[Node]:
        return self.true_graph.nodes

    @property
    def edges(self) -> list[Edge]:
        return self.true_graph.edges
