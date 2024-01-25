# Imports
import numpy as np
import networkx as nx
import functools

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Imports for code development
import matplotlib.pyplot as plt
import pdb
from diagnostics import assess_superstructure
from helpers import artificial_superstructure
from partitioning_methods import (
    modularity_partition,
    heirarchical_partition,
    expansive_causal_partition,
)
from duplicate_functions import create_two_comms
from build_heirarchical_random_graphs import (
    duplicate_get_random_graph_data,
    directed_heirarchical_graph,
)

from cd_v_partition.utils import (
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

outdir = "./"

## Generate a random network and corresponding dataset

# create graph without inserting "community" structure
if False:
    (edges, nodes, _, _), df = duplicate_get_random_graph_data(
        graph_type="scale_free",
        iv_samples=0,
        num_nodes=50,
        nsamples=int(1e4),
        p=0.5,
        k=2,
    )
    G_star_adj = edge_to_adj(list(edges), nodes=nodes)
    G_star_graph = nx.from_numpy_array(G_star_adj)

# Generate graph with heirarchical structure
if True:
    G_star_graph = directed_heirarchical_graph(num_nodes=50)
    G_star_adj = nx.adjacency_matrix(G_star_graph)
    assert set(G_star_graph.nodes()) == set(range(G_star_graph.number_of_nodes()))
    nodes = list(range(G_star_graph.number_of_nodes()))

# create two-community groundtruth graph
if False:
    init_partition, G_star_graph = create_two_comms(
        "scale_free", n=25, m1=2, m2=1, p1=0.5, p2=0.5, nsamples=0
    )
    G_star_adj = nx.adjacency_matrix(G_star_graph)
    # make sure set of ordered nodes [0, n-1] is identical to the graph nodes
    # i.e. graph doesn't skip any indices etc.
    assert set(G_star_graph.nodes()) == set(range(G_star_graph.number_of_nodes()))
    nodes = list(range(G_star_graph.number_of_nodes()))

    create_partition_plot(
        superstructure_net,
        nodes,
        init_partition,
        "{}/trial_communities_in_superstructure.png".format(outdir),
    )

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

# plt.figure()
# nx.draw(G_star_graph)
# plt.show()
# pdb.set_trace()

# plt.figure()
# nx.draw(nx.from_numpy_array(superstructure))
# plt.show()
# pdb.set_trace()

assess_superstructure(G_star_adj, superstructure)

## Partition
if False:
    # create causal partition by expanding a modularity-based disjoint partition
    partition = modularity_partition(superstructure)
if True:
    # create causal partition using heirarchical methods
    partition = heirarchical_partition(superstructure)


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
    partition,
    "{}/trial_disjoint.png".format(outdir),
)
create_partition_plot(
    superstructure_net,
    nodes,
    causal_partition,
    "{}/trial_causal.png".format(outdir),
)
# visualize partition with respect to original graph
create_partition_plot(
    G_star_graph,
    nodes,
    partition,
    "{}/G_star_disjoint.png".format(outdir),
)
create_partition_plot(
    G_star_graph,
    nodes,
    causal_partition,
    "{}/G_star_causal.png".format(outdir),
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

# Compare the results of the A(X_v) and F({A(X_s)})
# You see the following printed for 'CD-serial' and 'CD-partition'
# SHD: 'number of wrong edges'
# SID: 'ignore this one'
# AUC: 'auroc where edge is 1, no edge is 0',
# TPR,FPR: ('true positive rate', 'false positive rate')
delta_causality(A_X_v, nx.to_numpy_array(fused_A_X_s), nx.to_numpy_array(G_star_graph))
