# Imports
import numpy as np
import functools
from concurrent.futures import ProcessPoolExecutor
from cd_v_partition.utils import (
    get_random_graph_data,
    delta_causality,
    edge_to_adj,
    adj_to_dag,
    evaluate_partition,
)
from cd_v_partition.causal_discovery import pc, sp_gies
from cd_v_partition.overlapping_partition import partition_problem
from cd_v_partition.vis_partition import create_partition_plot
from cd_v_partition.fusion import fusion


outdir = "./examples/"

# Generate a random network and corresponding dataset
(edges, nodes, _, _), df = get_random_graph_data(
    graph_type="scale_free", iv_samples=0, num_nodes=50, nsamples=int(1e4), p=0.5, m=1
)
G_star = edge_to_adj(list(edges), nodes=nodes)

# Find the 'superstructure'
superstructure, p_values = pc(df, alpha=0.5, outdir=None)
print("Found superstructure")

# Call the causal learner on the full data A(X_v) and superstructure
A_X_v = sp_gies(df, skel=superstructure, outdir=None)

# Partition the superstructure and the dataset
rand_partition = {0: np.arange(25), 1: np.arange(25, 50)}
subproblems = partition_problem(rand_partition, superstructure, df)

# Visualize the partition
superstructure_net = adj_to_dag(
    superstructure
)  # undirected edges in superstructure adjacency become bidirected
evaluate_partition(rand_partition, superstructure_net, nodes)
create_partition_plot(
    superstructure_net,
    nodes,
    rand_partition,
    "{}/tutorial_partition.png".format(outdir),
)

# Call the causal learner on subsets of the data F({A(X_s)}) and sub-structures
num_partitions = 2
nthreads = 2  # each thread handles one partition


def _local_structure_learn(subproblem):
    skel, data = subproblem
    adj_mat = sp_gies(data, skel=skel, outdir=None)
    return adj_mat


func_partial = functools.partial(_local_structure_learn)
results = []

chunksize = max(1, num_partitions // nthreads)

with ProcessPoolExecutor(max_workers=nthreads) as executor:
    for result in executor.map(func_partial, subproblems, chunksize=chunksize):
        results.append(result)

# Merge the subset learned graphs
df_obs = df.drop(columns=["target"])
data_obs = df_obs.to_numpy()
fused_A_X_s = fusion(rand_partition, results, data_obs)

# Compare the results of the A(X_v) and F({A(X_s)})
# You see the following printed for 'CD-serial' and 'CD-partition'
# SHD: 'number of wrong edges'
# SID: 'ignore this one'
# AUC: 'auroc where edge is 1, no edge is 0',
# TPR,FPR: ('true positive rate', 'false positive rate')
delta_causality(A_X_v, fused_A_X_s, G_star)
