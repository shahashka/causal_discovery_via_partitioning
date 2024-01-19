# Run superstructure creation, partition, local discovery and screening
# for a base case network with assumed community structure
from cd_v_partition.utils import (
    get_random_graph_data,
    get_data_from_graph,
    evaluate_partition,
)
from cd_v_partition.causal_discovery import pc, weight_colliders
from cd_v_partition.overlapping_partition import oslom_algorithm
from cd_v_partition.vis_partition import create_partition_plot
import networkx as nx
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt


def sample_and_check_sanity(
    graph_type, n, p, k, ncommunities, alpha, collider_weight, nsamples, outdir
):
    # Create a random 'base' network
    (arcs, nodes, _, _), _ = get_random_graph_data(
        graph_type, num_nodes=n, nsamples=0, iv_samples=0, p=p, m=k, save=False
    )
    net = nx.DiGraph()
    net.add_edges_from(arcs)
    net.add_nodes_from(nodes)

    # Create a tiled network with community structure, save to dataset directory
    print("Creating tiled net")
    nodes = np.arange(n * ncommunities)
    tiled_net = _construct_tiling(net, num_tiles=ncommunities)

    print("Generate data")
    # Generate data from the tiled network
    (arcs, nodes, _, _), df = get_data_from_graph(
        nodes,
        list(tiled_net.edges()),
        nsamples=nsamples,
        iv_samples=0,
        save=False,
        outdir=outdir,
    )

    # Use the data to generate a superstructure using the pc algorithm
    superstructure, p_values = pc(df, alpha=alpha, outdir=outdir)
    superstructure = weight_colliders(superstructure, weight=collider_weight)

    # Checks for sanity
    percent_colliders = _count_colliders(tiled_net)
    num_wrong = _check_superstructure(
        superstructure, nx.adjacency_matrix(tiled_net, nodelist=np.arange(len(nodes)))
    )

    return num_wrong, percent_colliders


def _construct_tiling(net, num_tiles):
    """Helper function to construct the tiled/community network from a base net.
    The tiling is done so that nodes in one community are preferentially attached
    (proportional to degree) to nodes in other communities.

    Args:
        net (nx.DiGraph): the directed graph for one community
        num_tiles (int): the number of tiles or communities to create

    Returns:
        nx.DiGraph: the final directed graph with community structure
    """
    num_nodes = len(list(net.nodes()))
    degree_sequence = sorted((d for _, d in net.in_degree()), reverse=True)
    dmax = max(degree_sequence)
    tiles = [net for _ in range(num_tiles)]

    # First add all communities as disjoint graphs
    tiled_graph = nx.disjoint_union_all(tiles)

    # Each node is preferentially attached to other nodes
    # The number of attached nodes is given by a probability distribution over
    # A = 1, 2 ... min(dmax,4) where the probability is equal to the in_degree=A/number of nodes
    # in the community
    A = np.min([dmax, 4])
    in_degree_a = [sum(np.array(degree_sequence) == a) for a in range(A)]
    leftover = num_nodes - sum(in_degree_a)
    in_degree_a[-1] += leftover
    probs = np.array(in_degree_a) / (num_nodes)

    # Add connections based on random choice over probability distribution
    for t in range(1, num_tiles):
        for i in range(num_nodes):
            node_label = t * num_nodes + i
            if len(list(tiled_graph.predecessors(node_label))) == 0:
                num_connected = np.random.choice(np.arange(A), size=1, p=probs)
                dest = np.random.choice(np.arange(t * num_nodes), size=num_connected)
                connections = [(node_label, d) for d in dest]
                tiled_graph.add_edges_from(connections)
    return tiled_graph


def _count_colliders(G):
    """Helper function to count the number of colliders in the graph G. For every
    triple x-y-z determine if the edges are in a collider orientation. This counts
    as one collider set.

    Args:
        G (nx.DiGraph): Directed graph

    Returns:
        int: number of collider sets in the graph
    """
    num_colliders = 0
    non_colliders = 0

    # Find all triples x-y-z
    for x, y, z in itertools.permutations(G.nodes, 3):
        if G.has_edge(x, y) and G.has_edge(z, y):
            num_colliders += 1
        elif G.has_edge(x, y) and G.has_edge(y, z):
            non_colliders += 1
        elif G.has_edge(y, x) and G.has_edge(z, y):
            non_colliders += 1
        elif G.has_edge(y, x) and G.has_edge(y, z):
            non_colliders += 1
    return num_colliders / (num_colliders + non_colliders)


# Check that this is a superstructure
def _check_superstructure(S, G):
    """Make sure that S is a superstructure of G. This means all edges in G are constrained
       by S.


    Args:
        S (np.ndarray): adjacency matrix for the superstructure
        G (np.ndattay): adjacency matrix for the DAG
    """
    num_wrong = 0
    for row in np.arange(S.shape[0]):
        for col in np.arange(S.shape[1]):
            if G[row, col] == 1:
                if S[row, col] == 0:
                    num_wrong += 1
    # assert(np.sum(G>0) < np.sum(S>0))
    return num_wrong / np.sum(G > 0)


if __name__ == "__main__":
    graph_types = ["erdos_renyi", "scale_free", "small_world"]
    niters = 10
    p_range = np.arange(0.2, 1, 0.2)
    k_range = np.arange(2, 10, 2)
    fig, axs = plt.subplots(
        len(p_range), len(k_range), figsize=(10, 10), sharex=True, sharey=True
    )

    for i, p in enumerate(p_range):
        for j, k in enumerate(k_range):
            collider_metrics_by_graph_type = []
            ss_metrics_by_graph_type = []
            for g in graph_types:
                print(g, p, k)
                metrics = np.zeros(niters)
                for n in range(niters):
                    ss_metric, collider_metric = sample_and_check_sanity(
                        g,
                        n=10,
                        p=p,
                        k=k,
                        ncommunities=5,
                        alpha=0.1,
                        collider_weight=10,
                        nsamples=int(1e6),
                        outdir="./datasets/base_case_sweep/",
                    )
                    metrics[n] = ss_metric
                ss_metrics_by_graph_type.append(metrics)

                axs[i][j].violinplot(
                    ss_metrics_by_graph_type, showmeans=True, showmedians=False
                )
                axs[i][j].set_xticks(
                    np.arange(1, len(graph_types) + 1), labels=graph_types, rotation=45
                )
                axs[i][j].set_title("p={:.1f}, k={:.1f}".format(p, k))
    fig.suptitle("Fraction of wrong superstructure edges compared to edges in G")
    plt.savefig("./tests/sample_ss_by_graph.png")
