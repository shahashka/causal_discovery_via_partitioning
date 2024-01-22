import networkx as nx
import numpy as np


def convert_local_adj_mat_to_graph(partition, local_cd_adj_mats):
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
        subgraph = nx.from_numpy_array(adj, create_using=nx.DiGraph)
        subgraph = nx.relabel_nodes(
            subgraph, mapping=dict(zip(np.arange(len(node_ids)), node_ids)), copy=True
        )
        local_cd_graphs.append(subgraph)
    return local_cd_graphs


def union_with_overlaps(graphs):
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


def detect_cycle(G, edge):
    """
    For a candidate added edge (u,v)
    if the path v->u exists is the graph, then adding the edge will create a cycle

    Args:
        G ():
        edge ():

    Returns:

    """
    has_path = nx.has_path(G, edge[1], edge[0])
    return has_path


def resolve_w_ric_score(G, data, cov, i, j, pa_i, pa_j):
    """
    From paper https://www.jmlr.org/papers/v21/19-318.html
    Choose no edge, i->j, or j->i based on local RIC score
    Only add an edge if the RIC score for i->j and j->i both are greater than the score with no edge
    max(RIC(i->j), RIC(j->i)) < RIC(i,j))

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
    l_0i = loglikelihood(data, i, pa_i, cov)
    l_0j = loglikelihood(data, j, pa_j, cov)
    l_ij = loglikelihood(data, j, pa_j + [i], cov)
    l_ji = loglikelihood(data, i, pa_i + [j], cov)
    p = data.shape[1]
    n = data.shape[0]
    lam = np.log(n) if p <= np.sqrt(n) else 2 * np.log(p)

    add_edge = 2 * np.min([l_ij - l_0j, l_ji - l_0i]) > lam

    # Choose the edge that does not result in a cycle, otherwise choose the
    # minimal scoring edge
    if add_edge:
        if detect_cycle(G, (i, j)):
            return (j, i)
        elif detect_cycle(G, (j, i)):
            return (i, j)
        elif l_ji > l_ij:
            return (i, j)
        else:
            return (j, i)
    else:
        return None


def loglikelihood(samples, node, parents, correlation):
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
