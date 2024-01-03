import networkx as nx
import numpy as np
import itertools

# Final fusion step to merge subgraphs
# In infinite data limit this is done by screening for conflicting edges during union over subgraphs


def screen_projections(partition, local_cd_adj_mats):
    """Fuse subgraphs by taking the union and resolving conflicts by favoring no edge over
    directed edge. Leave bidirected edges as is. This is the method used for 'infinite' data limit problems

    Args:
        partition (dict): the partition as a dictionary {comm_id : [nodes]}
        local_cd_adj_mat (list[np.ndarray]): list of adjacency matrices for each local subgraph

    Returns:
        nx.DiGraph: the final global directed graph with all nodes and edges
    """
    # Take the union over graphs
    local_cd_graphs = _convert_local_adj_mat_to_graph(partition, local_cd_adj_mats)
    global_graph = _union_with_overlaps(local_cd_graphs)

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


def fusion(partition, local_cd_adj_mats, data, cov):
    """Fuse subgraphs by taking the union and resolving conflicts by taking the lower
    scoring edge. Ensure that the edge added does not create a cycle

    Args:
        partition (dict): the partition as a dictionary {comm_id : [nodes]}
        local_cd_adj_mat (list[np.ndarray]): list of adjacency matrices for each local subgraph

    Returns:
        nx.DiGraph: the final global directed graph with all nodes and edges
    """
    # Convert adjacency matrices to nx.DiGraphs, make sure to label nodes using the partition
    local_cd_graphs = _convert_local_adj_mat_to_graph(partition, local_cd_adj_mats)

    # Take the union over graphs
    global_graph = _union_with_overlaps(local_cd_graphs)

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
            edge = _resolve_w_ric_score(
                global_graph_resolved, data, cov, i, j, pa_i, pa_j
            )

            if edge:
                global_graph_resolved.add_edge(edge[0], edge[1])
    return global_graph_resolved


def fusion_basic(partition, local_cd_adj_mats):
    """Fuse subgraphs by taking the union and resolving conflicts by taking the higher
    weighted edge (for now). Eventually we want to the proof to inform how the merge happens here
    and we also want to consider finite data affects.

    Args:
        partition (dict): the partition as a dictionary {comm_id : [nodes]}
        local_cd_adj_mat (list[np.ndarray]): list of adjacency matrices for each local subgraph

    Returns:
        nx.DiGraph: the final global directed graph with all nodes and edges
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
    """
    local_cd_graphs = []
    for part, adj in zip(partition.items(), local_cd_adj_mats):
        _, node_ids = part
        subgraph = nx.from_numpy_array(adj, create_using=nx.DiGraph)
        nx.relabel_nodes(
            subgraph, mapping=dict(zip(np.arange(len(node_ids)), node_ids)), copy=False
        )
        local_cd_graphs.append(subgraph)
    return local_cd_graphs


def _union_with_overlaps(graphs):
    """
    Helper function that reimplements networkx.union_all, except remove the
    requirement that the node sets be disjoint ie we allow for overlapping nodes/edges
    between graphs
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
        R.add_nodes_from(G.nodes(data=True))
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
    l_0i = _fast_logpdf(data, i, pa_i, cov)
    l_0j = _fast_logpdf(data, j, pa_j, cov)
    l_ij = _fast_logpdf(data, j, pa_j + [i], cov)
    l_ji = _fast_logpdf(data, i, pa_i + [j], cov)
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


def _fast_logpdf(samples, node, parents, covariance):
    """Calculate the likelihood of a set of nodes and candidate parents for Gaussian variables.
    TODO where does this eq come from. The code is ported from graphical_models.GaussDAG.fast_logpdf

    Args:
        samples (np.ndarray): data matrix where each column corresponds to a random variable
        node (int): the variable (column in data matrix) to calculate the likelhood of
        parents (list of ints): the list of parent ids for the node
        covariance (np.ndarray): the covariance matrix for the data matrix
    Returns:
        (float) log likelihood value
    """
    cov_nn = covariance[np.ix_([node], [node])]
    cov_pn = covariance[np.ix_(parents, [node])]
    cov_pp = covariance[np.ix_(parents, parents)]
    rss = cov_nn - cov_pn.T.dot(np.linalg.inv(cov_pp)).dot(cov_pn)

    # vals, vecs = np.linalg.eigh(rss)
    # logdet     = np.sum(np.log(vals))
    # valsinv    = 1./vals
    # U          = vecs * np.sqrt(valsinv)
    # dim        = len(vals)
    # mean = np.mean(samples[:, node])
    # dev        = samples[:, node] - mean
    # maha       = np.square(np.dot(dev, U)).sum(axis=1)
    # log2pi     = np.log(2 * np.pi)
    # return -0.5 * (dim * log2pi + maha + logdet)
    N = samples.shape[0]
    return 0.5 * (-N * (np.log(2 * np.pi) + 1 - np.log(N) + np.log(rss * (N - 1))))
