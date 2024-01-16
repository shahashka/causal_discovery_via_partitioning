import networkx as nx
import numpy as np
from cd_v_partition.vis_partition import create_partition_plot
from cd_v_partition.overlapping_partition import partition_problem
from cd_v_partition.utils import (
    get_random_graph_data,
    get_data_from_graph,
    delta_causality,
    edge_to_adj,
)
from cd_v_partition.causal_discovery import sp_gies, pc
from cd_v_partition.fusion import screen_projections, fusion, fusion_basic
import functools
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def apply_causal_order(undirected_graph):
    """Apply a causal order to an undirected graph and return a DAG. The order is determined
    by the degree distribution. A highly connected node is more likely to be a source node and
    upstream in the topological order

    Args:
        undirected_graph (nx.Graph): undirected skeleton of the graph

    Returns:
        nx.DiGraph: a directed graph without cycles
    """
    deg_dist = np.array(list(undirected_graph.degree()), dtype=int)[:, 1]
    num_nodes = len(deg_dist)
    normalize = np.sum(np.array(list(undirected_graph.degree()), dtype=int)[:, 1])
    prob = [deg_dist[i] / normalize for i in np.arange(num_nodes)]
    # Higher degree is chosen first
    causal_order = list(
        np.random.choice(np.arange(num_nodes), size=num_nodes, p=prob, replace=False)
    )

    # Loop through the undirected edges, reverse any edges that are not consistent with
    # the topological order
    undirected_edges = undirected_graph.edges()
    directed_edges = []
    for e in undirected_edges:
        if causal_order.index(e[0]) > causal_order.index(e[1]):
            directed_edge = e[::-1]
        else:
            directed_edge = e

        # Make sure there are no self loops
        if directed_edge[::-1] not in directed_edges:
            directed_edges.append(directed_edge)
    directed_graph = nx.DiGraph()
    directed_graph.add_edges_from(directed_edges)
    return directed_graph


def create_two_comms(graph_type, n, m1, m2, p1, p2, vis=True):
    """Create a graph with two communities for the specified graph type and parameters. Use
    preferential attachment to connect two disjoint subgraphs. Also apply a causal order based on the degree distribution so that the
    resulting graph is a DAG.

    Args:
        graph_type (str): erdos_renyi, scale_free, small_world
        n (int): number of nodes in one community
        m1 (int): number of edges to attach from a new node to existing nodes (scale_free) or number of nearest neighbors connected in ring (small_world) for community 1
        m2 (int): number of edges to attach from a new node to existing nodes (scale_free) or number of nearest neighbors connected in ring (small_world) for community 2
        p1 (float): probability of edge creation (erdos_renyi) or rewiring (small_world) for community 1
        p2 (float): probability of edge creation (erdos_renyi) or rewiring (small_world) form community 2

    Returns:
        dict, nx.DiGraph: a dictionary for the partition into two communities, DAG for the two community network
    """
    # generate the edges set
    comm_1 = get_random_graph_data(
        graph_type=graph_type, num_nodes=n, nsamples=0, iv_samples=0, p=p1, k=m1
    )[0][0]
    comm_2 = get_random_graph_data(
        graph_type=graph_type, num_nodes=n, nsamples=0, iv_samples=0, p=p2, k=m2
    )[0][0]

    comm_1 = nx.DiGraph(comm_1)
    comm_2 = nx.DiGraph(comm_2)

    # connect the two communities using preferential attachment
    num_tiles = 2
    degree_sequence = sorted((d for _, d in comm_1.in_degree()), reverse=True)
    dmax = max(degree_sequence)
    tiles = [comm_1, comm_2]

    # First add all communities as disjoint graphs
    tiled_graph = nx.disjoint_union_all(tiles)

    # Each node is preferentially attached to other nodes
    # The number of attached nodes is given by a probability distribution over
    # A = 1, 2 ... min(dmax,4) where the probability is equal to the in_degree=A/number of nodes
    # in the community
    A = np.min([dmax, 4])
    in_degree_a = [sum(np.array(degree_sequence) == a) for a in range(A)]
    leftover = n - sum(in_degree_a)
    in_degree_a[-1] += leftover
    probs = np.array(in_degree_a) / (n)

    # Add connections based on random choice over probability distribution
    for t in range(1, num_tiles):
        for i in range(n):
            node_label = t * n + i
            if len(list(tiled_graph.predecessors(node_label))) == 0:
                num_connected = np.random.choice(np.arange(A), size=1, p=probs)
                dest = np.random.choice(np.arange(t * n), size=num_connected)
                connections = [(node_label, d) for d in dest]
                tiled_graph.add_edges_from(connections)

    causal_tiled_graph = apply_causal_order(tiled_graph)

    init_partition = {0: list(np.arange(n)), 1: list(np.arange(n, 2 * n))}
    if vis:
        create_partition_plot(
            causal_tiled_graph,
            list(causal_tiled_graph.nodes()),
            init_partition,
            "{}/two_comm.png".format("./tests/empirical_tests"),
        )
    return init_partition, causal_tiled_graph

def _local_structure_learn(subproblem):
    skel, data = subproblem
    adj_mat = sp_gies(data, skel=skel, outdir=None)
    return adj_mat
    
def run_causal_discovery(partition, df, G_star):
    # Find superstructure
    df_obs = df.drop(columns=["target"])
    data_obs = df_obs.to_numpy()
    superstructure, _ = pc(data_obs, alpha=0.5, outdir=None)

    # Break up problem according to provided partition
    subproblems = partition_problem(partition, superstructure, df)

    # Local learn
    func_partial = functools.partial(_local_structure_learn)
    results = []
    num_partitions = len(partition)
    nthreads = 2
    chunksize = max(1, num_partitions // nthreads)
    print("Launching processes")

    with ProcessPoolExecutor(max_workers=nthreads) as executor:
        for result in executor.map(func_partial, subproblems, chunksize=chunksize):
            results.append(result)

    # Merge globally
    est_graph_partition = fusion(partition, results, data_obs)
    #est_graph_partition = screen_projections(partition, results)

    # Call serial method
    est_graph_serial = _local_structure_learn([superstructure, df])

    # Compare causal metrics
    d_scores = delta_causality(est_graph_serial, est_graph_partition, G_star)
    return d_scores[-2]  # this is the delta true positive rate


def expansive_causal_partition(partition, graph):
    """Creates a causal partition by adding the outer-boundary of each cluster to that cluster.

    Args:
        adj_mat (np.ndarray): the adjacency matrix for the superstructure
        partition (dict): the estimated partition as a dictionary {comm_id : [nodes]}

    Returns:
        dict: the causal partition as a dictionary {comm_id : [nodes]}
    """
    causal_partition = dict()
    for idx, c in enumerate(list(partition.values())):
        outer_node_boundary = nx.node_boundary(graph, c)
        expanded_cluster = set(c).union(outer_node_boundary)
        causal_partition[idx] = list(expanded_cluster)
    create_partition_plot(
        graph,
        list(graph.nodes()),
        partition,
        "./tests/empirical_tests/expansive_causal_cover.png",
    )

    return causal_partition


def define_rand_edge_coverage(partition, graph, vis=True):
    num_nodes = len(graph.nodes)
    unmarked_nodes = list(np.arange(num_nodes))
    for n in graph.nodes():
        comm_n = int(n >= num_nodes / 2)
        for m in nx.neighbors(graph, n):
            n_unmarked = n in unmarked_nodes
            m_unmarked = m in unmarked_nodes
            if n_unmarked or m_unmarked:
                comm_m = int(m >= num_nodes / 2)
                if comm_n != comm_m:
                    if (
                        m % 2
                    ):  # Randomly assign the cut nodes to one or the other partition to ensure edge coverage
                        partition[comm_n] += [m]
                    else:
                        partition[comm_m] += [n]
                    if m_unmarked:
                        unmarked_nodes.remove(m)
                    if n_unmarked:
                        unmarked_nodes.remove(n)
    partition[0] = list(set(partition[0]))
    partition[1] = list(set(partition[1]))
    if vis:
        create_partition_plot(
            graph,
            list(graph.nodes()),
            partition,
            "./tests/empirical_tests/edge_cover.png",
        )

    return partition

def run():
    num_repeats = 30
    sample_range = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
    scores_edge_cover = np.zeros((num_repeats, len(sample_range)))
    scores_hard_partition = np.zeros((num_repeats, len(sample_range)))
    scores_causal_partition = np.zeros((num_repeats, len(sample_range)))
    for i in range(num_repeats):
        init_partition, graph = create_two_comms(
            "scale_free", n=25, m1=1, m2=2, p1=0.5, p2=0.5, vis=True
        )
        num_nodes = len(graph.nodes())
        bias = np.random.normal(0, 1, size=num_nodes)
        var = np.abs(np.random.normal(0, 1, size=num_nodes))
        for j,ns in enumerate(sample_range):
            # Generate data
            (edges, nodes, _, _), df = get_data_from_graph(
                list(np.arange(num_nodes)),
                list(graph.edges()),
                nsamples=int(ns),
                iv_samples=0,bias=bias, var=var
            )
            G_star = edge_to_adj(edges, nodes)
            d_tpr_hard = run_causal_discovery(init_partition, df, G_star)
            scores_hard_partition[i][j] = d_tpr_hard

            partition = define_rand_edge_coverage(init_partition, graph, vis=False)
            d_tpr_ec = run_causal_discovery(partition, df, G_star)
            scores_edge_cover[i][j] = d_tpr_ec

            partition = expansive_causal_partition(init_partition, graph)
            d_tpr_cp = run_causal_discovery(partition, df, G_star)
            scores_causal_partition[i][j] = d_tpr_cp


    labels = []

    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))

    plt.clf()
    _, ax = plt.subplots()
    add_label(
        ax.violinplot(scores_edge_cover, showmeans=True, showmedians=False),
        label="edge_cover",
    )
    add_label(
        ax.violinplot(scores_hard_partition, showmeans=True, showmedians=False),
        label="hard_partition",
    )
    add_label(
        ax.violinplot(scores_causal_partition, showmeans=True, showmedians=False),
        label="expansive_causal_partition",
    )

    ax.set_xticks(
        np.arange(1, len(sample_range) + 1),
        labels=["1e{}".format(i) for i in range(2,len(sample_range)+2)],
        rotation=45,
    )
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Delta TPR (Serial - Partition)")
    ax.set_title("Comparison of partition types for 2 community scale free networks")
    plt.legend(*zip(*labels), loc=2)
    plt.savefig(
        "./tests/empirical_tests/causal_part_test_sparse_w_expansive_fusion_same_dist.png"
    )


if __name__ == "__main__":
    run()
