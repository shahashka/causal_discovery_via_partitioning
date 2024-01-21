import networkx as nx
import numpy as np

from cd_v_partition.utils import get_scores, get_data_from_graph
from cd_v_partition.overlapping_partition import partition_problem
from cd_v_partition.causal_discovery import sp_gies, pc
import matplotlib.pyplot as plt
from test_causal_partition import create_two_comms


def get_wrong_edges(est_adj, true_adj):
    """Return the edges that differ between two adjacency matrices. Use the row/col index
    as the ndoe name

    Args:
        est_adj (np.ndarray): Estimated graph adjacency matrix
        true_adj (np.ndarray): True graph adjacency matrix

    Returns:
        list: list of pairs of row,col indices
    """
    wrong = np.argwhere(est_adj - true_adj)
    return list(wrong)


def localize_errors(wrong_edges, G_star, partition):
    """Given a list of incorrect edges, the true DAG and a two community partition,
    calculate the normalized distance from the endpoints of the errors to the edge of the
    partition. This is the shortest path from any of the endpoints to any of the nodes in
    the other community normalized by the diameter of the true DAG

    Args:
        wrong_edges (list): list of edges that are incorrect
        G_star (np.ndarray): the adjacency matrix of the true graph
        partition (dict): Keys are community indices, values are lists of nodes that
        belong to that community

    Returns:
        list(float): a list of distances to boundaries for each error
    """
    node_to_comm = dict()
    for comm, node in partition.items():
        for n in node:
            node_to_comm[n] = comm

    G_star = nx.from_numpy_array(G_star)
    diam = nx.diameter(G_star)
    dist = []
    for edge in wrong_edges:
        source = edge[0]
        target = edge[1]
        comm = node_to_comm[source]
        other_comm = 1 - comm
        sp = 100
        for node in partition[other_comm]:
            try:
                from_source = len(nx.shortest_path(G_star, source, node))
                from_target = len(nx.shortest_path(G_star, target, node))
                from_source_adjust = from_source - 1 if from_source > 1 else sp
                from_target_adjust = from_target - 1 if from_target > 1 else sp
                sp = np.min([sp, from_source_adjust, from_target_adjust])
            except:  # ignore if no path was found
                continue
        dist.append(sp / diam)
    return dist


def run():
    num_repeats = 30
    nsamples = int(1e6)
    num_nodes = 50
    alpha = 5e-1  # large alpha -> more denser superstructure

    dist_A_S1_S2 = []
    dist_A_S1 = []
    dist_A_S2 = []
    for _ in range(num_repeats):
        partition, sf_graph = create_two_comms(
            graph_type="scale_free", n=int(num_nodes / 2), m1=2, m2=2, p1=0.5, p2=0.5
        )
        df = get_data_from_graph(
            list(np.arange(num_nodes)),
            list(sf_graph.edges()),
            nsamples=nsamples,
            iv_samples=0,
        )[-1]
        G_star = nx.adjacency_matrix(sf_graph, nodelist=np.arange(num_nodes)).todense()

        superstructure, _ = pc(df, alpha=alpha, outdir=None)
        full_adj = sp_gies(
            df, outdir=None, skel=superstructure, use_pc=True, alpha=alpha
        )
        full_adj[full_adj != 0] = 1  # Ignore parameters

        subproblems = partition_problem(partition, superstructure, df)
        subgraphs = []
        for skel, data in subproblems:
            comm_adj = sp_gies(data, outdir=None, skel=skel, use_pc=True, alpha=alpha)
            comm_adj[comm_adj != 0] = 1  # ignore parameters
            subgraphs.append(comm_adj)

        wrong_edges = get_wrong_edges(full_adj, G_star)
        dist = localize_errors(wrong_edges, G_star, partition)
        dist_A_S1_S2 += dist

        wrong_edges = get_wrong_edges(
            subgraphs[0], G_star[partition[0]][:, partition[0]]
        )
        dist = localize_errors(wrong_edges, G_star, partition)
        dist_A_S1 += dist

        wrong_edges = get_wrong_edges(
            subgraphs[1], G_star[partition[1]][:, partition[1]]
        )
        dist = localize_errors(wrong_edges, G_star, partition)
        dist_A_S2 += dist

        get_scores(["full"], [full_adj], G_star)

    _, ax = plt.subplots()

    ax.violinplot(
        [dist_A_S1_S2, dist_A_S1, dist_A_S2], showmeans=True, showmedians=False
    )

    ax.set_xticks(np.arange(1, 4), labels=["A(S1 U S2)", "A(S1)", "A(S2)"], rotation=45)

    ax.set_xlabel("Edge errors")
    ax.set_ylabel(
        "Normalized shortest path length from error endpoints to opposite community"
    )
    ax.set_title("Distance to boundary for errors for two comm scale free network")
    plt.savefig("./tests/empirical_tests/localize_errors.png")


run()
