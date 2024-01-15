# Imports
import numpy as np
import networkx as nx

from cd_v_partition.utils import get_random_graph_data
from cd_v_partition.vis_partition import create_partition_plot

"""Functions contained elsewhere in cd_v_partition that I was struggling to import to test_alternative_partitions.
"""


# copied verbatim from test_causal_partition to avoid having to figure out how to import atm
def create_two_comms(graph_type, n, m1, m2, p1, p2, nsamples):
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
    create_partition_plot(
        causal_tiled_graph,
        list(causal_tiled_graph.nodes()),
        init_partition,
        "{}/two_comm.png".format("./"),
    )
    return init_partition, causal_tiled_graph


# Impose a causal ordering according to degree distribution, return a directed graph
def apply_causal_order(undirected_graph):
    deg_dist = np.array(list(undirected_graph.degree()), dtype=int)[:, 1]
    num_nodes = len(deg_dist)
    normalize = np.sum(np.array(list(undirected_graph.degree()), dtype=int)[:, 1])
    prob = [deg_dist[i] / normalize for i in np.arange(num_nodes)]
    causal_order = list(
        np.random.choice(np.arange(num_nodes), size=num_nodes, p=prob, replace=False)
    )

    undirected_edges = undirected_graph.edges()
    directed_edges = []
    for e in undirected_edges:
        if causal_order.index(e[0]) > causal_order.index(e[1]):
            directed_edges.append(e[::-1])
        else:
            directed_edges.append(e)
    directed_graph = nx.DiGraph()
    directed_graph.add_edges_from(directed_edges)
    return directed_graph
