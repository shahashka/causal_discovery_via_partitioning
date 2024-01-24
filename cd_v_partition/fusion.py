from __future__ import annotations

import itertools
from typing import Any

import networkx as nx
import numpy as np

import itertools
from conditional_independence import partial_correlation_test

# Final fusion step to merge subgraphs
# In infinite data limit this is done by screening for conflicting edges during union over subgraphs


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
    local_cd_graphs = _convert_local_adj_mat_to_graph(partition, local_cd_adj_mats)
    global_graph = _union_with_overlaps(local_cd_graphs)

    # global_graph = no edge if (no edge in comm1) or (no edge in comm2)
    for comm, adj_comm in zip(partition.values(), local_cd_adj_mats):
        if len(comm) > 1:
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


def fusion(partition: dict[Any, Any], local_cd_adj_mats: list[np.ndarray], data: np.ndarray, full_cand_set: bool =False):
    """
    Fuse subgraphs by taking the union and resolving conflicts by taking the lower
    scoring edge. Ensure that the edge added does not create a cycle

    Args:
        partition (dict): the partition as a dictionary {comm_id : [nodes]}

        local_cd_adj_mats (list[np.ndarray]): list of adjacency matrices for each local subgraph
        data (): ...

    Returns:
        nx.DiGraph: the final global directed graph with all nodes and edges
    """
    # Convert adjacency matrices to nx.DiGraphs, make sure to label nodes using the partition
    local_cd_graphs = _convert_local_adj_mat_to_graph(partition, local_cd_adj_mats)

    # Take the union over graphs
    global_graph = _union_with_overlaps(local_cd_graphs)
    cor = np.corrcoef(data.T)

    comms = [set(p) for p in partition.values()]
    overlaps = set.intersection(*comms)

    def get_parents_in_cand_set(cand_set, i, j):
        parents = []
        for (a,b) in cand_set:
            if a==i or a==j:
                parents.append(b)
            elif a==j or b==j:
                parents.append(a)
        return parents

    # Sort the list of possible overlapping edges accordinng to their p-value
    suffstat = {"n": data.shape[0], "C": cor}
    candidate_edges = list(itertools.combinations(overlaps, 2)) if not full_cand_set else _candidate_edges_(partition, suffstat, global_graph, alpha=1e-3)
    if len(candidate_edges) > 0:
        conditioning_set = [
            set.union(
                *[
                    set(global_graph.predecessors(i)),
                    set(global_graph.predecessors(j)),
                    get_parents_in_cand_set(candidate_edges, i, j),
                ]
            )
            for i, j in candidate_edges
        ]
        p_value = [
            partial_correlation_test(suffstat, i, j, S)["p_value"]
            for (i, j), S in zip(candidate_edges, conditioning_set)
        ]
        p_value, candidate_edges = zip(*sorted(zip(p_value, candidate_edges)))

    # Loop through the edge options and favor lower ric_score
    for i, j in candidate_edges:
        if global_graph.has_edge(j, i):
            global_graph.remove_edge(j, i)
        if global_graph.has_edge(i, j):
            global_graph.remove_edge(i, j)

        pa_i = list(global_graph.predecessors(i))
        pa_j = list(global_graph.predecessors(j))
        edge = _resolve_w_ric_score(global_graph, data, cor, i, j, pa_i, pa_j)

        if edge:
            global_graph.add_edge(edge[0], edge[1])
    return global_graph


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
    local_cd_graphs = _convert_local_adj_mat_to_graph(partition, local_cd_adj_mats)

    # Take the union over graphs
    global_graph = _union_with_overlaps(local_cd_graphs)

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


def _convert_local_adj_mat_to_graph(partition, local_cd_adj_mats):
    """
    Helper function to convert the local adjacency matrices (resultant of the causal discovery
    method) into networkx DiGraphs using the correct global node index for each partition.

    Args:
        partition (): ...
        local_cd_adj_mats (): ...

    Returns:
        ...
    """
    local_cd_graphs = []
    for part, adj in zip(partition.items(), local_cd_adj_mats):
        _, node_ids = part
        if len(node_ids) == 1:
            subgraph = nx.DiGraph()
            subgraph.add_nodes_from(node_ids)
        else:
            subgraph = nx.from_numpy_array(adj, create_using=nx.DiGraph)
            subgraph = nx.relabel_nodes(
                subgraph, mapping=dict(zip(np.arange(len(node_ids)), node_ids)), copy=True
            )
        local_cd_graphs.append(subgraph)
    return local_cd_graphs


def _union_with_overlaps(graphs):
    """
    Helper function that reimplements networkx.union_all, except remove the
    requirement that the node sets be disjoint ie we allow for overlapping nodes/edges
    between graphs


    Args:
        graphs (list[nx.DiGraph]): ...

    Returns:
        ...
    """
    R = None
    seen_nodes = set()
    for i, G in enumerate(graphs):
        G_nodes_set = set(G.nodes)
        if i == 0:
            # Union is the same type as first graph
            R = G.__class__()
        seen_nodes |= G_nodes_set
        R.graph.update(G.graph)
        R.add_nodes_from(G.nodes(data=False))
        R.add_edges_from(
            G.edges(keys=True, data=True) if G.is_multigraph() else G.edges(data=True)
        )

    if R is None:
        raise ValueError("cannot apply union_all to an empty list")

    return R


# For a candidate added edge (u,v)
# if the path v->u exists is the graph, then adding the edge will create a cycle
def _detect_cycle(G, edge):
    has_path = nx.has_path(G, edge[1], edge[0])
    return has_path


# From paper https://www.jmlr.org/papers/v21/19-318.html
# Choose no edge, i->j, or j->i based on local RIC score
# Only add an edge if the RIC score for i->j and j->i both are greater than the score with no edge
# max(RIC(i->j), RIC(j->i)) < RIC(i,j))
def _resolve_w_ric_score(G, data, cov, i, j, pa_i, pa_j):
    """
    ...

    Args:
        G ():
        data ():
        cov ():
        i ():
        j ():
        pa_i ():
        pa_j ():

    Returns:
        ...
    """
    l_0i = _loglikelihood(data, i, pa_i, cov)
    l_0j = _loglikelihood(data, j, pa_j, cov)
    l_ij = _loglikelihood(data, j, pa_j + [i], cov)
    l_ji = _loglikelihood(data, i, pa_i + [j], cov)
    p = data.shape[1]
    n = data.shape[0]
    lam = np.log(n) if p <= np.sqrt(n) else 2 * np.log(p)

    add_edge = 2 * np.min([l_ij - l_0j, l_ji - l_0i]) > lam

    # Choose the edge that does not result in a cycle, otherwise choose the
    # minimal scoring edge
    if add_edge:
        if _detect_cycle(G, (i, j)):
            return (j, i)
        elif _detect_cycle(G, (j, i)):
            return (i, j)
        elif l_ji > l_ij:
            return (i, j)
        else:
            return (j, i)
    else:
        return None


def _loglikelihood(samples, node, parents, correlation):
    """Calculate the the log likelihood of the least squares estimate of a node given it's parents

    Args:
        samples (np.ndarray): data matrix where each column corresponds to a random variable
        node (int):the variable (column in data matrix) to calculate the likelhood of
        parents (list of ints): the list of parent ids for the node
        correlation (np.ndarray): the correlation coefficient matrix for the data matrix

    Returns:
        (float) log likelhood value
    """
    cor_nn = correlation[np.ix_([node], [node])]
    cor_pn = correlation[np.ix_(parents, [node])]
    cor_pp = correlation[np.ix_(parents, parents)]
    rss = cor_nn - cor_pn.T.dot(np.linalg.inv(cor_pp)).dot(cor_pn)
    N = samples.shape[0]
    return 0.5 * (-N * (np.log(2 * np.pi) + 1 - np.log(N) + np.log(rss * (N - 1))))

def _candidate_edges_(partition, suffstat, global_graph, alpha):
    A = []
    all_comms = itertools.combinations(partition.values(),2)
    all_edges = []
    for pair_comm in all_comms:
        all_edges += list(itertools.product(pair_comm[0], pair_comm[1]))
    for edge in all_edges:
        i = edge[0]
        j = edge[1]
        if i!=j: # ignore self loops
            N_i = set(global_graph.predecessors(i))
            N_j = set(global_graph.predecessors(j))
            rho = partial_correlation_test(suffstat,i,j,cond_set=N_i.union(N_j),alpha=alpha)
            if rho['reject']:
                A.append((i,j))
    return A
    