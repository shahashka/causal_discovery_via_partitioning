from typing import Any, Callable, TypeAlias

from networkx import DiGraph
from numpy import ndarray
from pandas import DataFrame

CausalDiscoveryAlgorithm: TypeAlias = Callable[[DataFrame, float, ndarray], ndarray]
"""Function signature for [causal discovery algorithms][cd_v_partition.algorithms.causal_discovery]."""

FusionAlgorithm: TypeAlias = Callable[[dict[Any, Any], list[ndarray], ndarray], DiGraph]
"""Function signature for [fusion algorithms][cd_v_partition.algorithms.fusion]."""

PartitioningAlgorithm: TypeAlias = Callable[[dict, list[ndarray]], dict]
"""Function signature for [partitioning algorithms][cd_v_partition.algorithms.partitioning]."""
