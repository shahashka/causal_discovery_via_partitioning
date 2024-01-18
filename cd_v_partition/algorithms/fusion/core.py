"""
This file defines the THREE core fusion algorithms we consider for this work.
"""
import itertools
from typing import Any

import networkx as nx
import numpy as np

from cd_v_partition.algorithms.fusion.utils import (
    convert_local_adj_mat_to_graph,
    union_with_overlaps,
    resolve_w_ric_score,
)


def fusion(partition: dict[Any, Any], local_cd_adj_mats: list[np.ndarray], data, cov):
    """Fuse subgraphs by taking the union and resolving conflicts by taking the lower
    scoring edge. Ensure that the edge added does not create a cycle

    Args:
        partition (dict): the partition as a dictionary {comm_id : [nodes]}
        local_cd_adj_mats (list[np.ndarray]): list of adjacency matrices for each local subgraph
        data (): ...
        cov (): ...

    Returns:
        nx.DiGraph: the final global directed graph with all nodes and edges
    """
    # Convert adjacency matrices to nx.DiGraphs, make sure to label nodes using the partition
    local_cd_graphs = convert_local_adj_mat_to_graph(partition, local_cd_adj_mats)

    # Take the union over graphs
    global_graph = union_with_overlaps(local_cd_graphs)

    global_graph_resolved = global_graph.copy()  # TODO this is an expensive copy
    for i, j in global_graph.edges():
        if global_graph.has_edge(j, i):
            #  Resolve conflicts by favoring lower ric_scores
            if global_graph_resolved.has_edge(j, i):
                global_graph_resolved.remove_edge(j, i)
            if global_graph_resolved.has_edge(i, j):
                global_graph_resolved.remove_edge(i, j)

            pa_i = list(global_graph_resolved.predecessors(i))
            pa_j = list(global_graph_resolved.predecessors(j))
            edge = resolve_w_ric_score(
                global_graph_resolved, data, cov, i, j, pa_i, pa_j
            )

            if edge:
                global_graph_resolved.add_edge(edge[0], edge[1])
    return global_graph_resolved


def fusion_basic(
    partition: dict[Any, Any], local_cd_adj_mats: list[np.ndarray]
) -> nx.DiGraph:
    """
    Fuse subgraphs by taking the union and resolving conflicts by taking the higher
    weighted edge (for now). Eventually we want to the proof to inform how the merge happens here
    and we also want to consider finite data affects.

    Args:
        partition (dict): the partition as a dictionary {comm_id : [nodes]}
        local_cd_adj_mats (list[np.ndarray]): list of adjacency matrices for each local subgraph

    Returns:
        The final global directed graph with all nodes and edges
    """
    # Convert adjacency matrices to nx.DiGraphs, make sure to label nodes using the partition
    local_cd_graphs = convert_local_adj_mat_to_graph(partition, local_cd_adj_mats)

    # Take the union over graphs
    global_graph = union_with_overlaps(local_cd_graphs)

    #  Resolve conflicts by favoring higher weights
    global_graph_resolved = global_graph.copy()  # TODO this is an expensive copy
    for i, j in global_graph.edges():
        if global_graph.has_edge(j, i):
            weight_ij = global_graph.get_edge_data(i, j)["weight"]
            weight_ji = global_graph.get_edge_data(j, i)["weight"]
            print("Conflict found, weights: {} {}".format(weight_ij, weight_ji))
            if weight_ij > weight_ji:
                global_graph_resolved.remove_edge(j, i)
            else:
                global_graph_resolved.remove_edge(i, j)

    return global_graph_resolved


def screen_projections(
    partition: dict[Any, Any], local_cd_adj_mats: list[np.ndarray]
) -> nx.DiGraph:
    """
    Fuse subgraphs by taking the union and resolving conflicts by favoring no edge over
    directed edge. Leave bidirected edges as is. This is the method used for 'infinite'
    data limit problems.

    Args:
        partition (dict[Any, Any]): the partition as a dictionary {comm_id : [nodes]}
        local_cd_adj_mats (list[np.ndarray]): list of adjacency matrices for each local subgraph

    Returns:
        nx.DiGraph: the final global directed graph with all nodes and edges
    """
    # Take the union over graphs
    local_cd_graphs = convert_local_adj_mat_to_graph(partition, local_cd_adj_mats)
    global_graph = union_with_overlaps(local_cd_graphs)

    # global_graph = no edge if (no edge in comm1) or (no edge in comm2)
    for comm, adj_comm in zip(partition.values(), local_cd_adj_mats):
        for row, col in itertools.product(
            np.arange(adj_comm.shape[0]), np.arange(adj_comm.shape[0])
        ):
            i = comm[row]
            j = comm[col]
            if (
                not adj_comm[row, col] and not adj_comm[col, row]
            ) and global_graph.has_edge(i, j):
                global_graph.remove_edge(i, j)
    return global_graph
