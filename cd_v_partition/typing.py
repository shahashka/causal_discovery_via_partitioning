from dataclasses import dataclass
from typing import Any, Iterable, Literal, NamedTuple, TypeAlias

from numpy import ndarray
from pandas import DataFrame

Node: TypeAlias = Any
Edge: TypeAlias = tuple[Any, Any]
Community: TypeAlias = int

Partition: TypeAlias = dict[Community, Iterable[Node]]


class TrueGraph(NamedTuple):
    """Give some description here"""
    nodes: list[Node]
    """The nodes ..."""
    edges: list[Edge]
    """The edges are..."""


GraphKind = Literal["erdos_renyi", "small_world", "scale_free"]
"""Definition for the literals of random graphs that are supported."""


@dataclass
class GeneratedGraph:
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
