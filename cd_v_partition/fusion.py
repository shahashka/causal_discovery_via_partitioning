from __future__ import annotations

import itertools
from typing import Any

import networkx as nx
import numpy as np

import itertools
from conditional_independence import partial_correlation_test

import pdb


# Final fusion step to merge subgraphs
# In infinite data limit this is done by screening for conflicting edges during union over subgraphs
def remove_edges_not_in_ss(
    target_graph: nx.DiGraph, ss_graph: nx.DiGraph
) -> nx.DiGraph:
    """
    Remove edges from target_graph which do not exist in ss_graph

    Args:
        target_graph (nx.DiGraph): the target directed graph
        ss_graph (nx.DiGraph): the superstructure

    Returns:
        nx.DiGraph: target_graph with only edges appearing in ss_graph
    """
    # specify edge orientation to avoid issues with orderings of tuples
    ss_edge_set = set(ss_graph.edges())
    target_edge_set = set(target_graph.edges())
    # find edges in global_graph that are present in ss_graph
    target_edges_in_superstructure = list(target_edge_set.intersection(ss_edge_set))
    weights = [target_graph.edges[e[0],e[1]]['weight'] for e in target_edges_in_superstructure]
    # reset all edges in global_graph
    target_graph.remove_edges_from(list(target_graph.edges()))
    # add back edges from global_edges_in_superstructure
    target_graph.add_edges_from(target_edges_in_superstructure, weight=weights)
    return target_graph

def no_partition_postprocess(
    ss: np.ndarray,
    est_adj_mat: np.ndarray,
    ss_subset: bool =True,
) -> np.ndarray:
    """Method to postprocess the graph when there is no partition. 
    
    If the ss_subset flag is set then remove all edges that are not in 
    the provided supserstructure, otherwise return the estimated adjacency
    matrix as is. 

    Args:
        ss (np.ndarray): adjacency matrix for superstructure
        est_adj_mat (np.ndarray): estimated adjacency matrix
        ss_subset (bool, optional): Flag to filter out edges not in the superstructure. Defaults to True.

    Returns:
        np.ndarray: The resultant graph as an adjancency matrix
    """
    if ss_subset:
        ss_graph = nx.from_numpy_array(ss, create_using=nx.DiGraph)
        est_DiGraph = nx.from_numpy_array(
            est_adj_mat, create_using=nx.DiGraph
        )
        subselected_serial_DiGraph = remove_edges_not_in_ss(
            est_DiGraph, ss_graph
        )
        # convert back to numpy array
        est_adj_mat = nx.to_numpy_array(
            subselected_serial_DiGraph,
            nodelist=np.arange(len(subselected_serial_DiGraph.nodes())),
        )
    return est_adj_mat

def screen_projections_pag2cpdag(
    ss: np.ndarray,
    partition: dict[Any, Any],
    local_cd_adj_mats: list[np.ndarray],
    ss_subset:bool =True,
    finite_lim : bool = True,
    data:np.ndarray =None,
    full_cand_set: bool = False,
) -> nx.DiGraph:
    
    # The pag represetation has following edge to number mapping
    # pag[i,j] = 0 iff no edge btw i,j
    # pag[i,j] = 1 iff i *-o j
    # pag[i,j] = 2 iff i *-> j
    # pag[i,j] = 3 iff i *-- j
    
    # CPDAG
    # cpdag[i,j] = 0 and cpdag[j,i] = 0 iff no edge between i, j
    # cpdag[i,j] = 1 and cpdag[j,i] = 0 iff i->j 
    # cpdag[i,j] = 1 and cpdag[j,i] = 1 iff i--j 
    
    # Start with a fully connected global CPDAG 
    cpdag = np.ones(ss.shape)
    pag_edges = dict()

    for comm_id, pag in enumerate(local_cd_adj_mats):
        for row, col in itertools.product(np.arange(pag.shape[0]), np.arange(pag.shape[1])):
            global_row = partition[comm_id][row]
            global_col = partition[comm_id][col]
            # If edge does not exist in any PAG, remove it from the global CPDAG
            # This ensures that edges in overlap that agree in the PAG space remain in the global CPDAG
            # while edges that disagree are removed
            if pag[row,col] == 0:
                cpdag[global_row, global_col] = 0
                cpdag[global_col, global_row] = 0
                
            # Add edges to dictionary, create a list for overlapping edges
            else:
                if (global_row, global_col) in pag_edges:
                    pag_edges[(global_row, global_col)] += [pag[row,col]]
                else:
                    pag_edges[(global_row, global_col)] = [pag[row,col]]
    
    for edge, end_marks in pag_edges.items():
        u = edge[0]
        v = edge[1]
        # Tag PAG arrowheads (that agree across all PAGs) in global CDPAG         
        if all(x==2 for x in end_marks) and cpdag[u,v]==1:
            cpdag[u,v]=2
    print(len(np.where(cpdag==2)[0]))
    # Orient unshielded triples in CDPAG adjacency matrix
    all_twos=0
    arrowheads_from = [[i for i in range(cpdag.shape[0]) if cpdag[i,col]==2] for col in range(cpdag.shape[1])]
    for col in range(cpdag.shape[1]):
        all_twos+=len(arrowheads_from[col])
        # Check if there is a triple that is unshielded
        if len(arrowheads_from[col]) == 2:
            u,v = arrowheads_from[col]
            #usheilded
            if cpdag[u,v] == 0:
                cpdag[u,col] = 1
                cpdag[v,col] = 1
                cpdag[col, u] = 0
                cpdag[col, v] = 0
            #shielded
            else:
                cpdag[u,col] = 1
                cpdag[v,col] = 1
                cpdag[col, u] = 1
                cpdag[col, v] = 1
        # For all other cases with a PAG arrowhead, 
        # set to an undirected edge in the CPDAG
        elif len(arrowheads_from[col]) > 0:
            for node in arrowheads_from[col]:
                if cpdag[node,col] == 2:
                    cpdag[node, col] = 1
    print(all_twos)
    print(len(np.where(cpdag==2)[0]))
    assert((cpdag < 2).all())
    cpdag_digraph = nx.from_numpy_array(cpdag, create_using=nx.DiGraph)
    
    # Remove all edges not present in superstructure
    if ss_subset:
        ss_graph = nx.from_numpy_array(ss, create_using=nx.DiGraph)
        cpdag_digraph = remove_edges_not_in_ss(cpdag_digraph, ss_graph)
        
    if finite_lim:
        cpdag_digraph = screen_projections_finite_lim_postprocessing(
            ss_graph,
            cpdag_digraph,
            partition,
            ss_subset,
            data,
        )
    return cpdag_digraph
    
def screen_projections(
    ss: np.ndarray,
    partition: dict[Any, Any],
    local_cd_adj_mats: list[np.ndarray],
    ss_subset:bool =True,
    finite_lim : bool = True,
    data:np.ndarray =None,
    full_cand_set: bool = False,
) -> nx.DiGraph:
    """
    Fuse subgraphs by taking the union and resolving conflicts by favoring no edge over
    directed edge. Leave bidirected edges as is. This is the method used for 'infinite'
    data limit problems.

    Args:
        ss (np.ndarray): adjacency matrix for the super structure
        partition (dict[Any, Any]): the partition as a dictionary {comm_id : [nodes]}
        local_cd_adj_mats (list[np.ndarray]): list of adjacency matrices for each local subgraph
        ss_subset (bool): whether to only include edges in global_graph which are in ss
        finite_lim (bool): whether to include adaptations to finite limit setting, including
        resolving bidirected edges using RIC score and cycle detection/deletion.
        data (None or np.ndarray): if finite_lim==True, we need data to use RIC score
        full_cand_set (bool): unused flag
    Returns:
        nx.DiGraph: the final global directed graph with all nodes and edges
    """
    # Take the union over graphs
    local_cd_graphs = _convert_local_adj_mat_to_graph(partition, local_cd_adj_mats)
    global_graph = _union_with_overlaps(local_cd_graphs)
    print(f"SCREEN PROJECTIONS AFTER UNION {len(nx.get_edge_attributes(global_graph, 'weight'))}")

    # Remove all edges not present in superstructure
    ss_graph = nx.from_numpy_array(ss, create_using=nx.DiGraph)
    if ss_subset:
        global_graph = remove_edges_not_in_ss(global_graph, ss_graph)

    # global_graph = no edge if (no edge in comm1) or (no edge in comm2)
    k = list(partition.keys())
    k.sort()
    for i, adj_comm in zip(k, local_cd_adj_mats):
        comm = partition[i]
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
    print(f"SCREEN PROJECTIONS BEFORE FINITE LIM {len(nx.get_edge_attributes(global_graph, 'weight'))}")

    # resolve bidirected edges and delete cycles using RIC score
    if finite_lim:
        global_graph = screen_projections_finite_lim_postprocessing(
            ss_graph,
            global_graph,
            partition,
            ss_subset,
            data,
        )
    print(f"SCREEN PROJECTIONS END {len(nx.get_edge_attributes(global_graph, 'weight'))}")
    return global_graph


def screen_projections_finite_lim_postprocessing(
    ss_graph: nx.DiGraph,
    global_graph: nx.DiGraph,
    partition: dict[Any, Any],
    ss_subset=True,
    data=None,
) -> nx.DiGraph:
    """
    Adapts results of screen_projections to finite limit setting by resolving bidirected
    edges using RIC score and cycle detection/deletion.

    Args:
        ss_graph (nx.DiGraph): directed graph for super structure
        global_graph (nx.DiGraph): estimated directed graph
        partition (dict[Any, Any]): the partition as a dictionary {comm_id : [nodes]}
        ss_subset (bool): whether to only include edges in global_graph which are in ss
        data (None or np.ndarray): we need data to use RIC score

    Returns:
        nx.DiGraph: the final estimated global directed graph
    """
    # We'll need correlation from data to compute RIC score
    cor = np.corrcoef(data.T)

    # Remove trivial cycles (self-loops)
    global_graph.remove_edges_from(nx.selfloop_edges(global_graph))

    # If no cycles remain, return graph
    # nx.find_cycle will throw an error nx.exception.NetworkXNoCycle if no cycles contained
    try:
        cycle_list = nx.find_cycle(global_graph, orientation="original")
    except:
        return global_graph

    # Otherwise, find and eliminate cycles by deleting edges between nodes in overlap
    # Start by making a dictionary (node_id: list_of_communities_containing_node)
    nodes = list(global_graph.nodes())
    node_to_partition = dict(zip(nodes, [[] for _ in np.arange(len(nodes))]))
    for key, value in partition.items():
        for node in value:
            node_to_partition[node] += [key]

    # Find nodes in the overlap based on this dictionary
    def _find_overlaps(partition):
        overlaps = []
        for node, comm in partition.items():
            if len(comm) > 1:
                overlaps.append(node)

        return overlaps

    overlaps = _find_overlaps(node_to_partition)
    # While the graph contains cycles,
    while len(cycle_list) > 0:
        # If we've found a trivial cycle, i.e (i,j) and (j,i) both exist
        if len(cycle_list) == 2:
            # find the endpoints
            i = cycle_list[0][0]
            j = cycle_list[0][1]
            # remove the edges in question so that predecessors don't include i and j
            global_graph.remove_edge(i, j)
            global_graph.remove_edge(j, i)
            # find parents and use RIC score method
            pa_i = list(global_graph.predecessors(i))
            pa_j = list(global_graph.predecessors(j))
            edge = _resolve_w_ric_score(global_graph, data, cor, i, j, pa_i, pa_j)

            # subset_check is true if either we're NOT restricting our estimate to
            # edges present in the superstructure, or if we are restricting and the
            # candidate edge does appear in the superstructure
            subset_check = (not ss_subset) or (edge in list(ss_graph.edges()))
            if edge and subset_check:
                global_graph.add_edge(edge[0], edge[1])
        # If we've found a longer, nontrivial cycle
        else:
            # Find edges that exist in overlap
            edges_in_overlap = []
            for edge in cycle_list:
                # if both endpoints live in overlap
                if edge[0] in overlaps and edge[1] in overlaps:
                    edges_in_overlap.append(edge[:2])
            # Haven't implemented "select edge from overlap with worst METRIC"
            if False:
                # Compute log-likelihood for each edge, and discard the edge with the
                # largest log-likelihood score, i.e. the lowest likelihood score.
                loglikelihood_scores = []
                for edge in edges_in_overlap:
                    i = edge[0]
                    j = edge[1]
                    pa_i = list(global_graph.predecessors(i))
                    pa_j = list(global_graph.predecessors(j))
                    loglikelihood_scores.append(
                        _loglikelihood(data, j, pa_j + [i], cor)
                    )
            # Currently: select arbitrary edge in cycle that's in overlap
            if True:
                if len(edges_in_overlap) == 0:
                    print(
                        "WARNING: CYCLE OCCURS NOT IN OVERLAP. Removing arbitrary edge."
                    )
                    edge_data = cycle_list[0]
                else:
                    edge_data = edges_in_overlap[0]
            global_graph.remove_edge(edge_data[0], edge_data[1])
        # nx.find_cycle will throw an error nx.exception.NetworkXNoCycle when all cycles have been removed
        try:
            cycle_list = nx.find_cycle(global_graph, orientation="original")
        except:
            break
    return global_graph


def fusion(
    ss: np.ndarray,
    partition: dict[Any, Any],
    local_cd_adj_mats: list[np.ndarray],
    data: np.ndarray,
    ss_subset=False,
    finite_lim: bool=False,
    full_cand_set: bool = False,
):
    """
    Fuse subgraphs by taking the union and resolving conflicts by taking the lower
    scoring edge. Ensure that the edge added does not create a cycle
    
    Resolving bidirected edges using RIC score and cycle detection/deletion.

    Args:
        ss (np.ndarray): adjacency matrix for the super structure
        partition (dict[Any, Any]): the partition as a dictionary {comm_id : [nodes]}
        local_cd_adj_mats (list[np.ndarray]): list of adjacency matrices for each local subgraph
        ss_subset (bool): whether to only include edges in global_graph which are in ss
        finite_lim (bool): unused flag
        data (None or np.ndarray): if finite_lim==True, we need data to use RIC score
        full_cand_set (bool): Flag to condition on the all nodes in the graph when determining
                             if an edge exists between two nodes in different subsets. 
                              If False, will only condition on nddes in overlapping sets.
                              Default to False.

    Returns:
        nx.DiGraph: the final global directed graph with all nodes and edges
    """
    # Convert adjacency matrices to nx.DiGraphs, make sure to label nodes using the partition
    local_cd_graphs = _convert_local_adj_mat_to_graph(partition, local_cd_adj_mats)

    # Take the union over graphs
    global_graph = _union_with_overlaps(local_cd_graphs)

    ss_graph = nx.from_numpy_array(ss, create_using=nx.DiGraph)
    cor = np.corrcoef(data.T)
    suffstat = {"n": data.shape[0], "C": cor}

    def _find_overlaps(partition):
        overlaps = []
        for node, comm in partition.items():
            if len(comm) > 1:
                overlaps.append(node)
        return overlaps

    nodes = list(global_graph.nodes())
    node_to_partition = dict(zip(nodes, [[] for _ in np.arange(len(nodes))]))
    for key, value in partition.items():
        for node in value:
            node_to_partition[node] += [key]

    overlaps = _find_overlaps(node_to_partition)

    if full_cand_set:
        all_comms = itertools.combinations(partition.values(), 2)
        candidate_edges = []
        for pair_comm in all_comms:
            candidate_edges += list(itertools.product(pair_comm[0], pair_comm[1]))
            candidate_edges += list(itertools.product(pair_comm[1], pair_comm[0]))

    else:
        candidate_edges = list(itertools.permutations(overlaps, 2))
    # First remove all candidate edges in overlap from the global graph so this does not interfere with correlation tests
    # print("Global graph edges before discarding edges {}".format(len(global_graph.edges())))
    for i, j in candidate_edges:
        if global_graph.has_edge(j, i):
            global_graph.remove_edge(j, i)
        if global_graph.has_edge(i, j):
            global_graph.remove_edge(i, j)

    # print("Global graph edges after discarding edges {}".format(len(global_graph.edges())))
    # print("Number of candidate edges before partial correlation {}".format(len(candidate_edges)))
    alpha = 1 / np.square(len(global_graph.nodes()))
    # print("Alpha for testing is {}".format(alpha))

    candidate_edges = _partial_correlation_cand_edges(
        candidate_edges, global_graph, suffstat, alpha=1e-3
    )
    # print("Number of candidate edges before sequential CI tests {}".format(len(candidate_edges)))
    candidate_edges = _candidate_edge_filter_w_CI_test(
        candidate_edges, global_graph, suffstat, alpha=alpha
    )
    # print("Final number of candidate edges {}".format(len(candidate_edges)))
    # print("Global graph edges before fusion {}".format(len(global_graph.edges())))

    # Loop through the edge options and favor lower ric_score
    for i, j in candidate_edges:
        pa_i = list(global_graph.predecessors(i))
        pa_j = list(global_graph.predecessors(j))
        edge = _resolve_w_ric_score(global_graph, data, cor, i, j, pa_i, pa_j)

        if edge and edge in list(ss_graph.edges()):
            global_graph.add_edge(edge[0], edge[1])
    # print("Global graph edges after fusion {}".format(len(global_graph.edges())))
    if ss_subset:
        global_graph = remove_edges_not_in_ss(global_graph, ss_graph)
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
    k = list(partition.keys())
    k.sort()
    for i, adj in zip(k, local_cd_adj_mats):
        node_ids = partition[i]
        if len(node_ids) == 1:
            subgraph = nx.DiGraph()
            subgraph.add_nodes_from(node_ids)
        else:
            subgraph = nx.from_numpy_array(adj,edge_attr="weight", create_using=nx.DiGraph)
            nx.relabel_nodes(
                subgraph,
                mapping=dict(zip(np.arange(len(node_ids)), node_ids)),
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


# Use partial correlation to filter a candidate set of edges and rank by their p_value
# Return a list sorted by p_value
def _partial_correlation_cand_edges(candidate_edges, global_graph, suffstat, alpha):
    A = []
    p_values = []
    if len(candidate_edges) > 0:
        for edge in candidate_edges:
            i = edge[0]
            j = edge[1]
            if i != j:  # ignore self loops
                N_i = set(global_graph.predecessors(i))
                N_j = set(global_graph.predecessors(j))
                rho = partial_correlation_test(
                    suffstat, i, j, cond_set=N_i.union(N_j), alpha=alpha
                )
                if (
                    rho["reject"] and (j, i) not in A
                ):  # do not add if the opposite direction is already included to save time
                    A.append((i, j))
                    p_values.append(rho["p_value"])

    if len(A) > 0:
        _, A = zip(*sorted(zip(p_values, A)))
    return A


# Use a sequential set of conditional independence tests to find a final candidate set of edges
# Add any parents from the current candidate edges to the conditioning set
def _candidate_edge_filter_w_CI_test(candidate_edges, global_graph, suffstat, alpha):
    def get_parents_in_cand_set(cand_set, i, j):
        parents = []
        for a, b in cand_set:
            if a == i or a == j:
                parents.append(b)
            elif a == j or b == j:
                parents.append(a)
        return parents

    # Sequential CI tests
    A_star = []
    if len(candidate_edges) > 0:
        for i, j in candidate_edges:
            conditioning_set = set.union(
                *[
                    set(global_graph.predecessors(i)),
                    set(global_graph.predecessors(j)),
                    get_parents_in_cand_set(A_star, i, j),
                ]
            )

            rho = partial_correlation_test(
                suffstat, i, j, conditioning_set, alpha=alpha
            )
            if rho["reject"]:
                A_star.append((i, j))
    return A_star
