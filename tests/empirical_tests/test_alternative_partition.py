# Imports
import numpy as np
import functools
from concurrent.futures import ProcessPoolExecutor

from diagnostics import assess_superstructure
from helpers import artificial_superstructure

from cd_v_partition.utils import (
    get_random_graph_data,
    get_data_from_graph,
    delta_causality,
    edge_to_adj,
    adj_to_dag,
    evaluate_partition,
)
from cd_v_partition.causal_discovery import pc, sp_gies
from cd_v_partition.overlapping_partition import partition_problem
from cd_v_partition.vis_partition import create_partition_plot
from cd_v_partition.fusion import fusion

import pdb
import networkx as nx
from tqdm import tqdm


def modularity_partition(adj_mat, resolution=1, cutoff=2, best_n=2):
    """Creates disjoint partition by greedily maximizing modularity. Using networkx built-in implementaiton.

    Args:
        adj_mat (np.ndarray): the adjacency matrix for the superstructure
        resolution (float): resolution parameter, trading off intra- versus inter-group edges.
        cutoff (int): lower limit on number of communities before termination
        best_n (int): upper limit on number of communities before termination
        See networkx documentation for more.

    Returns:
        dict: the estimated partition as a dictionary {comm_id : [nodes]}
    """
    G = nx.from_numpy_array(adj_mat)
    community_lists = nx.community.greedy_modularity_communities(
        G, cutoff=cutoff, best_n=best_n
    )

    partition = dict()
    for idx, c in enumerate(community_lists):
        partition[idx] = list(c)
    return partition


def expansive_causal_partition(adj_mat, partition):
    """Creates a causal partition by adding the outer-boundary of each cluster to that cluster.

    Args:
        adj_mat (np.ndarray): the adjacency matrix for the superstructure
        partition (dict): the estimated partition as a dictionary {comm_id : [nodes]}

    Returns:
        dict: the causal partition as a dictionary {comm_id : [nodes]}
    """
    G = nx.from_numpy_array(adj_mat)

    causal_partition = dict()
    for idx, c in enumerate(list(partition.values())):
        outer_node_boundary = nx.node_boundary(G, c)
        expanded_cluster = set(c).union(outer_node_boundary)
        causal_partition[idx] = list(expanded_cluster)
    return causal_partition


# copied verbatim from test_causal_partition to avoid having to figure out how to import atm
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
        "{}/two_comm.png".format(outdir),
    )
    return init_partition, causal_tiled_graph


outdir = "./"

## Generate a random network and corresponding dataset

# create graph without inserting "community" structure
if False:
    (edges, nodes, _, _), df = get_random_graph_data(
        graph_type="scale_free",
        iv_samples=0,
        num_nodes=50,
        nsamples=int(1e4),
        p=0.5,
        k=2,
    )
    G_star_adj = edge_to_adj(list(edges), nodes=nodes)
    G_star_graph = nx.from_numpy_array(G_star_adj)

# create two-community groundtruth graph
if True:
    init_partition, G_star_graph = create_two_comms(
        "scale_free", n=25, m1=2, m2=1, p1=0.5, p2=0.5, nsamples=0
    )
    G_star_adj = nx.adjacency_matrix(G_star_graph)
    # make sure set of ordered nodes [0, n-1] is identical to the graph nodes
    # i.e. graph doesn't skip any indices etc.
    assert set(G_star_graph.nodes()) == set(range(G_star_graph.number_of_nodes()))
    nodes = list(range(G_star_graph.number_of_nodes()))

## Generate data
df = get_data_from_graph(
    nodes,
    list(G_star_graph.edges()),
    nsamples=int(1e4),
    iv_samples=0,
)[1]
df_obs = df.drop(columns=["target"])
data_obs = df_obs.to_numpy()

## Find the 'superstructure'
# use PC algorithm to learn from data
if False:
    superstructure, p_values = pc(data_obs, alpha=0.1, outdir=None)
    print("Found superstructure")

# artificially create superstructure
if True:
    superstructure = artificial_superstructure(G_star_adj, frac_extraneous=0.1)

assess_superstructure(G_star_adj, superstructure)

## Partition
# create causal partition by expanding a modularity-based disjoint partition
partition = modularity_partition(superstructure)
causal_partition = expansive_causal_partition(superstructure, partition)

# visualize the partition
superstructure_net = adj_to_dag(
    superstructure
)  # undirected edges in superstructure adjacency become bidirected
evaluate_partition(partition, superstructure_net, nodes)
evaluate_partition(causal_partition, superstructure_net, nodes)
create_partition_plot(
    superstructure_net,
    nodes,
    init_partition,
    "{}/trial_communities_in_superstructure.png".format(outdir),
)
create_partition_plot(
    superstructure_net,
    nodes,
    partition,
    "{}/trial_disjoint.png".format(outdir),
)
create_partition_plot(
    superstructure_net,
    nodes,
    causal_partition,
    "{}/trial_causal.png".format(outdir),
)

## Learning Globally
# call the causal learner on the full data A(X_v) and superstructure
print("Beginning global learning.")
A_X_v = sp_gies(df, skel=superstructure, outdir=None)

## Learning Locally
# Call the causal learner on subsets of the data F({A(X_s)}) and sub-structures
subproblems = partition_problem(causal_partition, superstructure, df)
num_partitions = 2
results = []


def _local_structure_learn(subproblem):
    skel, data = subproblem
    adj_mat = sp_gies(data, skel=skel, outdir=None)
    return adj_mat


# non-parallelized version
print("Beginning local learning.")
for subproblem in tqdm(subproblems):
    results.append(_local_structure_learn(subproblem))

# Merge the subset learned graphs
fused_A_X_s = fusion(causal_partition, results, data_obs)
print("Successfully fused partition output.")
pdb.set_trace()

# Compare the results of the A(X_v) and F({A(X_s)})
# You see the following printed for 'CD-serial' and 'CD-partition'
# SHD: 'number of wrong edges'
# SID: 'ignore this one'
# AUC: 'auroc where edge is 1, no edge is 0',
# TPR,FPR: ('true positive rate', 'false positive rate')
delta_causality(A_X_v, nx.to_numpy_array(fused_A_X_s), nx.to_numpy_array(G_star_graph))
