from __future__ import annotations

# Relevant graph operations, metrics, and data generation
import itertools
import math
import os
from pathlib import Path
import random 

import causaldag as cd
import cdt
import networkx as nx
import numpy as np
import pandas as pd
from graphical_models import rand, GaussDAG
from numpy.random import RandomState
from sklearn.metrics import roc_curve


def load_random_state(random_state: RandomState | int | None = None) -> RandomState:
    if random_state is None:
        return RandomState()
    elif isinstance(random_state, RandomState):
        return random_state
    elif isinstance(random_state, int):
        return RandomState(random_state)
    else:
        raise ValueError(
            "Illegal value for `load_random_state()` Must be either an instance of "
            "`RandomState`, an integer to seed with, or None."
        )


def adj_to_edge(adj: np.ndarray, nodes: list[str], ignore_weights: bool = False):
    r"""
    Helper function to convert an adjacency matrix into an edge list. Optionally include weights so that
    the edge tuple is (i, j, weight).

    Args:
        adj (np.ndarray): Adjacency  matrix of dimensionality $p \times p$.
        nodes (list[str]): List of node names—in order corresponding to rows/cols of `adj`.
        ignore_weights (bool): Ignore the weights if `True`; include them if `False`. Defaults to `False`.

    Returns:
        Edge list (of nonzero values) from the given adjacency matrix.
    """
    edges = []
    for row, col in itertools.product(np.arange(adj.shape[0]), np.arange(adj.shape[1])):
        if adj[row, col] != 0:
            if ignore_weights:
                edges.append((nodes[row], nodes[col]))
            else:
                edges.append((nodes[row], nodes[col], {"weight": adj[row, col]}))
    return edges


def adj_to_dag(adj: np.ndarray) -> nx.DiGraph:
    r"""
    Helper function to convert an adjacency matrix into a directed graph.

    Args:
        adj (np.ndarray): Adjacency matrix of dimensionality $p \times p$.
        nodes (list[str]): List of node names—in order corresponding to rows/cols of `adj`.

    Returns:
        Directed acyclic graph from the given adjacency matrix.
    """
    dag = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    return dag


def edge_to_adj(edges: list[tuple], nodes: list[str]) -> np.ndarray:
    r"""
    Helper function to convert an edge list into an adjacency matrix.

    Args:
        edges (list[tuple]): List of (i,j) tuples corresponding to directed edges.
        nodes (list[str]): List of node names in order corresponding to rows/cols of adj.

    Returns:
        Adjacency matrix.
    """
    adj_mat = np.zeros((len(nodes), len(nodes)))
    for e in edges:
        start = nodes.index(e[0])
        end = nodes.index(e[1])
        adj_mat[start, end] = 1
    return adj_mat


def edge_to_dag(edges) -> nx.DiGraph:
    """
    Helper function to convert a list of edges into a Networkx DiGraph

    Args:
        edges (list[tuple]): Edge list of directed edges.

    Returns:
        Directed acyclic graph.
    """
    dag = nx.DiGraph()
    dag.add_edges_from(edges)
    return dag


def tpr_fpr_score(
    y_true: np.ndarray | nx.DiGraph, y_pred: np.ndarray | nx.DiGraph
) -> tuple[float, float]:
    """
    Calculate the true positive rate and false positive scores between a true graph and predicted graph using
    sklearn.roc_curve. We choose the point correponding to the maximum threshold i.e if the adjacency matrix
    only has 1s or 0s, the max threshold is 1 and the tpr corresponds to the number of correct 1s.


    Args:
        y_true (np.ndarray | nx.DiGraph): Ground truth topology (either adjacency matrix or directed graph).
        y_pred (np.ndarray | nx.DiGraph): Estimated topology (either adjacency matrix or directed graph).

    Returns:
        Tuple of floats corresponding to true positive rate and false positive rate in ROC curve
        at the maximum threshold value
    """
    if type(y_pred) == nx.DiGraph:
        y_pred = nx.adjacency_matrix(y_pred)
        y_pred = y_pred.todense()
    if type(y_true) == nx.DiGraph:
        y_true = nx.adjacency_matrix(y_true)
        y_true = y_true.todense()
    y_pred = np.array(y_pred != 0, dtype=int).flatten()
    y_true = np.array(y_true != 0, dtype=int).flatten()
    fpr, tpr, _ = roc_curve(y_true.flatten(), y_pred.flatten())
    return tpr[1], fpr[1]


def get_scores(
    alg_names: list[str],
    networks: list[np.ndarray] | list[nx.DiGraph],
    ground_truth: np.ndarray | nx.DiGraph,
    get_sid: bool = False,
) -> tuple[float, float, float, float, float]:
    """
    Calculate metrics Structural Hamming Distance (SHD), Structural Interventional Distance
    (SID), AUC, TPR,FPR for a set of algorithms and networks. Also handles averaging over several
    sets of networks (e.g the random comparison averages over several different generated graphs)
    Default turn off sid, since it is computationally expensive.

    Args:
        alg_names (list[str]): list of algorithm names.
        networks (list[np.ndarray] | list[nx.DiGraph]): list of estimated graphs corresponding to algorithm names.
        ground_truth (np.ndarray | nx.DiGraph): the true graph to compare to.
        get_sid (bool) : flag to calculate SID which is computationally expensive, returned SID is 0 if this is `False`.

    Returns:
        floats corresponding to SHD, SID, AUC, (TPR, FPR)
    """
    if type(ground_truth) == nx.DiGraph:
        ground_truth = nx.adjacency_matrix(
            ground_truth, nodelist=np.arange(len(ground_truth.nodes()))
        ).todense()
    for name, net in zip(alg_names, networks):
        if type(net) == nx.DiGraph:
            net = nx.adjacency_matrix(
                net, nodelist=np.arange(len(net.nodes()))
            ).todense()
        sid = cdt.metrics.SID(ground_truth, net) if get_sid else 0
        if name != "NULL":
            tpr_fpr = tpr_fpr_score(ground_truth, net)
        else:
            tpr_fpr = [0, 0]
        # Precision/recall, SHD requires 0,1 array
        ground_truth = (ground_truth != 0).astype(int)
        net = (net != 0).astype(int)
        auc, pr = cdt.metrics.precision_recall(ground_truth, net)
        shd = cdt.metrics.SHD(ground_truth, net, False)

        # print(
        #     "{} SHD: {} SID: {} AUC: {}, TPR,FPR: {}".format(
        #         name, shd, sid, auc, tpr_fpr
        #     )
        # )
        return shd, sid, auc, tpr_fpr[0], tpr_fpr[1]


def get_random_graph_data(
    graph_type: str,
    num_nodes: int,
    nsamples: int,
    iv_samples: int,
    p: float,
    m: int,
    seed: int | RandomState = 42,
    save: bool = False,
    outdir: Path | str = None,
    # TODO: Add return type,  -> tuple[tuple[set[tuple], list[int], float, float], pd.DataFrame]:
):
    """
    Create a random Gaussian DAG and corresponding observational and interventional dataset.
    Note that the generated topology with Networkx undirected, using graphical_models a causal ordering
    is imposed on this graph which makes it a DAG. Each node has a randomly sampled
    bias and variance from which Gaussian data is generated (children are sums of their parents)
    Save a data.csv file containing the observational and interventional data samples and target column
    Save a ground.txt file containing the edge list of the generated graph.

    Parameters:
        graph_type (str): erdos_renyi, scale_free (Barabasi-Albert) or small_world (Watts-Strogatz)
        num_nodes (int): number of nodes in the generated graph
        nsamples (int): number of observational samples to generate
        iv_samples (int) : number of interventional samples to generate
        p (float): probability of edge creation (erdos_renyi) or rewiring (small_world)
        m (int): number of edges to attach from a new node to existing nodes (scale_free) or number of
            nearest neighbors connected in ring (small_world)
        seed (int): random seed
        save (bool): flag to save the dataset (data.csv) and graph (ground.txt)
        outdir (str): directory to save the data to if save is True
    Returns:
        arcs (list of edges), nodes (list of node indices), bias (bias terms for Gaussian generative model),
            var (variance terms for Gaussian generative model) , df (pandas DataFrame containing sampled
            observational, interventional data and target indices)
    """
    random_state = load_random_state(seed)
    if graph_type == "erdos_renyi":
        random_graph_model = lambda nnodes: nx.erdos_renyi_graph(nnodes, p=p, seed=seed)
    elif "scale_free":
        random_graph_model = lambda nnodes: nx.barabasi_albert_graph(
            nnodes, m=m, seed=random_state
        )
    elif "small_world":
        random_graph_model = lambda nnodes: nx.watts_strogatz_graph(
            nnodes, k=m, p=p, seed=random_state
        )
    elif graph_type == "hierarchical":
        random_graph_model = lambda nnodes: nx.Graph(
            nx.scale_free_graph(
                nnodes,
                alpha=0.2,
                gamma=0.5,
                beta=0.3,
                delta_in=0.0,
                delta_out=0.0,
                seed=random_state,
            ).to_undirected()
        )
    else:
        raise ValueError("Unsupported random graph")

    # TODO this should take a random state
    dag = rand.directed_random_graph(num_nodes, random_graph_model)
    nodes_inds = list(dag.nodes)
    bias = random_state.normal(0, 1, size=len(nodes_inds))
    var = np.abs(random_state.normal(0, 1, size=len(nodes_inds)))

    bn = GaussDAG(nodes=nodes_inds, arcs=dag.arcs, biases=bias, variances=var)
    data = bn.sample(nsamples)

    df = pd.DataFrame(data=data, columns=nodes_inds)
    df["target"] = np.zeros(data.shape[0])

    if iv_samples > 0:
        i_data = []
        for ind, i in enumerate(nodes_inds):
            samples = bn.sample_interventional(
                cd.Intervention({i: cd.ConstantIntervention(val=0)}), iv_samples
            )
            samples = pd.DataFrame(samples, columns=nodes_inds)
            samples["target"] = ind + 1
            i_data.append(samples)
        df_int = pd.concat(i_data)
        df = pd.concat([df, df_int])
    if save:
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        df.to_csv("{}/data.csv".format(outdir), index=False)
        with open("{}/ground.txt".format(outdir), "w") as f:
            for edge in dag.arcs:
                f.write(str(edge) + "\n")
    return (dag.arcs, nodes_inds, bias, var), df


def get_data_from_graph(
    nodes: list[str],
    edges: list[tuple],
    nsamples: int,
    iv_samples: int,
    bias: np.ndarray = None,
    var: np.ndarray = None,
    save: bool = False,
    outdir: Path | str = None,
    random_state: int | RandomState = 42
):
    """
    Get data set from a predefined graph using the Gaussian DAG generative model (same as get_random_graph_data)
    Save a data.csv file containing the observational and interventional data samples and target vector
    Save a ground.txt file containing the edge list of the generated graph
    Save a data.csv file containing the observational and interventional data samples and target column
    Save a ground.txt file containing the edge list of the generated graph.

    Args:
        nodes (list[str]): list of node names
        edges (list[tuple]): list of directed edge tuples (i,j) where i and j are in nodes
        nsamples (int): number of observational samples to generate
        iv_samples (int) : number of interventional samples to generate
        save (bool): flag to save the dataset (data.csv) and graph (ground.txt)
        outdir (Path | str): directory to save the data to if save is True

    Returns:
        edges (list of edges), nodes (list of node indices), bias (bias terms for Gaussian generative model),
            var (variance terms for Gaussian generative model) , df (pandas DataFrame containing sampled
            observational, interventional data and target indices)
    """
    random_state = load_random_state(random_state)
    if bias is None or var is None:
        bias = random_state.normal(0, 1, size=len(nodes))
        var = np.abs(random_state.normal(0, 1, size=len(nodes)))
    bn = GaussDAG(nodes=nodes, arcs=edges, biases=bias, variances=var)
    data = bn.sample(nsamples)

    df = pd.DataFrame(data=data, columns=nodes)
    df["target"] = np.zeros(data.shape[0])

    if iv_samples > 0:
        i_data = []
        for ind, i in enumerate(nodes):
            samples = bn.sample_interventional(
                cd.Intervention({i: cd.ConstantIntervention(val=0)}), iv_samples
            )
            samples = pd.DataFrame(samples, columns=nodes)
            samples["target"] = ind + 1
            i_data.append(samples)
        df_int = pd.concat(i_data)
        df = pd.concat([df, df_int])
    if save:
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        df.to_csv("{}/data.csv".format(outdir), index=False)
        with open("{}/ground.txt".format(outdir), "w") as f:
            for edge in edges:
                f.write(str(edge) + "\n")
    return (edges, nodes, bias, var), df


# Partitions is a dictionary with key,values <partition id>: <list of nodes in partition>
# From this paper https://arxiv.org/abs/0910.5072
# Ranges from 1 to -1. Positive values are better, 1 indicates fully connected graph
def _modularity_overlapping(partitions, nodes, A):
    A = A.todense()  # Sparse matrix indexing support is poor

    def mod_cluster(part, nodes, A, S, D, n_edges, n_nodes):
        part_nodes = part[1]
        node_mod = np.zeros(len(part_nodes))
        for ind, i in enumerate(part_nodes):
            within_cluster = np.sum([A[i, j] for j in part_nodes if j != i])
            intra_cluster = np.sum([A[i, j] for j in nodes if j not in part_nodes])
            d_i = D[i]
            s_i = S[i]
            node_mod[ind] += (
                (within_cluster - intra_cluster) / (d_i * s_i) if d_i != 0 else 0
            )
        return n_edges / (math.comb(n_nodes, 2) * n_nodes) * sum(node_mod)

    K = len(partitions)
    # S is the number of clusters each node i belongs to
    S = np.zeros(len(nodes))

    # D is the degree of each node in the graph
    D = np.zeros(len(nodes))
    for i in nodes:
        S[i] = sum([1 for p in partitions.values() if i in p])
        D[i] = np.sum(A[i]) + np.sum(A[:, [i]]) - 2
    n_nodes = [len(p) for p in partitions.values()]
    p_inds = [np.array(list(p), dtype=int) for p in partitions.values()]
    n_edges = [np.sum(A[p][:, p]) for p in p_inds]

    mod_by_cluster = np.zeros(K)
    for i, part in enumerate(partitions.items()):
        mod_by_cluster[i] = mod_cluster(part, nodes, A, S, D, n_edges[i], n_nodes[i])

    return 1 / K * sum(mod_by_cluster)


def evaluate_partition(partition, G, nodes):
    """Evaluate the partition over a graph with the edge coverage and overlapping
    modularity scores

    Args:
        partition (dict): keys are community ids, values are lists of nodes
        G (nx.DiGraph): the original structure that is being partitioned
        nodes (list): list of nodes in order of adjacency matrix
    """
    # Edge coverage
    covered = 0
    for e in list(G.edges):
        is_covered = False
        for _, p in partition.items():
            if e[0] in p and e[1] in p:
                is_covered = True
        covered += 1 if is_covered else 0
    print("Percent of edges covered by partition: {}".format(covered / len(G.edges)))

    # Modularity of partitions
    mod_overlap = _modularity_overlapping(
        partition, nodes, nx.adjacency_matrix(G, nodelist=nodes)
    )
    print("Modularity for Overlapping partitions: {}".format(mod_overlap))


def delta_causality(est_graph_serial, est_graph_partition, true_graph):
    """Calculate the difference in scores (SHD, AUC, SID, TPR_FPR) between
    the serial estimated grpah and the partitioned estimated graph. The difference
    is calculated as serial_score - partition_score.

    Args:
        est_graph_serial (np.ndarray or nx.DiGraph): the estimated graph from running the causal
            discovery algorithm on the entire data and node set
        est_graph_partition (np.ndarray or nx.DiGraph): the estimated graph from running the causal
            discovery algorithm on the partitioned data and node sets
        true_graph (np.ndarray or nx.DiGraph): the ground truth graph to compare to

    Returns:
        list (float, float, float, float, float): Delta SHD, AUC, SID, TPR, FPR. Note that the sign
            here is relative to the serial implmentation (we do not take the aboslute value)
    """
    scores_s = get_scores(["CD serial"], [est_graph_serial], true_graph)
    scores_p = get_scores(["CD partition"], [est_graph_partition], true_graph)
    delta = [s - p for (s, p) in zip(scores_s, scores_p)]
    return delta

# TODO implement random_state
def create_k_comms(graph_type: str, n: int, m_list: list[int], p_list: list[int], k: int, 
                   rho: int = 0.01, random_state: RandomState | int = 0):
    """Create a random network with k communities with the specified graph type and parameters. Create this by
    generating k disjoint communities adn using preferential attachment. Remove any cycles to 
    make this a DAG

    Args:
        graph_type (str): erdos_renyi, scale_free (Barabasi-Albert) or small_world (Watts-Strogatz)
        n (int): number of nodes per community 
        m_list (list[int]): number of edges to attach from a new node to existing nodes (scale_free) or number of
                            nearest neighbors connected in ring (small_world)
        p_list (list[float]): probability of edge creation (erdos_renyi) or rewiring (small_world)
        k (int): number of communities
        rho (int, optional): Parameter to tune the strength of community structure. This is the fraction of total possible edges
                            between communities. Defaults to 0.01

    Returns:
        tuple(dict, nx.DiGraph): a dictionary storing the community partitions, the graph of the connected communities
    """
    random_state = load_random_state(random_state)
    comms = []
    for i in np.arange(k):
        if type(m_list) == int:
            m = m_list
        else:
            m = m_list[i]
        if type(p_list) == int:
            p = p_list
        else:
            p = p_list[i]
        comm_k = get_random_graph_data(
            graph_type=graph_type, num_nodes=n, nsamples=0, iv_samples=0, p=p, m=m, seed=random_state
        )[0][0]

        comms.append(nx.DiGraph(comm_k))

    # connect the communities using preferential attachment
    degree_sequence = sorted((d for _, d in comms[0].in_degree()), reverse=True)
    dmax = max(degree_sequence)

    # First add all communities as disjoint graphs
    comm_graph = nx.disjoint_union_all(comms)

    # Each node is preferentially attached to other nodes
    # The number of attached nodes is given by a probability distribution over
    # A = 1, 2 ... min(dmax,4) where the probability is equal to the in_degree=A/number of nodes
    # in the community
    A = np.min([dmax, 4])
    in_degree_a = [sum(np.array(degree_sequence) == a) for a in range(A)]
    leftover = n - sum(in_degree_a)
    in_degree_a[-1] += leftover
    probs = np.array(in_degree_a) / (n)

    # Add connections from one community to the previous communities based on probability distribution
    num_edges = rho * n**2 * k
    while num_edges > 0:
        for t in range(1, k):
            for i in range(n):
                node_label = t * n + i
                if len(list(comm_graph.predecessors(node_label))) == 0:
                    num_connected = random_state.choice(np.arange(A), size=1, p=probs)
                    dest = random_state.choice(np.arange(t * n), size=num_connected)
                    connections = [(node_label, d) for d in dest]
                    comm_graph.add_edges_from(connections)
                    num_edges -= num_connected

    init_partition = dict()
    for i in np.arange(k):
        init_partition[i] = list(np.arange(i * n, (i + 1) * n))
    comm_graph = _remove_cycles(comm_graph)
    return init_partition, comm_graph


def _remove_cycles(G):
    # find and remove cycles
    G.remove_edges_from(nx.selfloop_edges(G))
    try:
        cycle_list = nx.find_cycle(G, orientation="original")
        print("Number of cycles found is {}".format(len(cycle_list)))
        while len(cycle_list) > 0:
            edge_data = cycle_list[-1]
            G.remove_edge(edge_data[0], edge_data[1])
            # nx.find_cycle will throw an error nx.exception.NetworkXNoCycle when all cycles have been removed
            try:
                cycle_list = nx.find_cycle(G, orientation="original")
            except:
                break
        return G
    except:
        return G

def artificial_superstructure(
    G_star_adj_mat, frac_retain_direction=0.1, frac_extraneous=0.5
):
    """Creates a superstructure by discarding some of the directions in edges of G_star and adding
    extraneous edges.

    Args:
        G_star_adj_mat (np.ndarray): the adjacency matrix for the target graph
        frac_retain_direction (float): what percentage of edges will retain their direction information
        frac_extraneous (float): adds frac_extraneous*m many additional edges, for m the number of edges in G_star

    Returns:
        super_adj_mat (np.ndarray): an adjacency matrix for the superstructure we've created
    """
    G_star = nx.from_numpy_array(G_star_adj_mat, create_using=nx.DiGraph())

    # returns a deepcopy
    G_super = G_star.to_undirected()
    # add extraneous edges
    m = G_star.number_of_edges()
    nodes = list(G_star.nodes())
    G_super.add_edges_from(pick_k_random_edges(k=int(frac_extraneous * m), nodes=nodes))

    return nx.adjacency_matrix(G_super).toarray()

def pick_k_random_edges(k, nodes):
    return list(zip(random.choices(nodes, k=k), random.choices(nodes, k=k)))


def directed_heirarchical_graph(num_nodes, seed):
    G = nx.DiGraph(
        nx.scale_free_graph(
            num_nodes,
            alpha=0.2,
            gamma=0.5,
            beta=0.3,
            delta_in=0.0,
            delta_out=0.0,
            seed=seed
        )
    )
    # find and remove cycles
    G.remove_edges_from(nx.selfloop_edges(G))
    cycle_list = nx.find_cycle(G, orientation="original")
    while len(cycle_list) > 0:
        edge_data = cycle_list[-1]
        G.remove_edge(edge_data[0], edge_data[1])
        # nx.find_cycle will throw an error nx.exception.NetworkXNoCycle when all cycles have been removed
        try:
            cycle_list = nx.find_cycle(G, orientation="original")
        except:
            break
    return G