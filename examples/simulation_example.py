# Imports
import numpy as np
import functools
from concurrent.futures import ProcessPoolExecutor
from cd_v_partition.utils import (
    get_data_from_graph,
    delta_causality,
    edge_to_adj,
    adj_to_dag,
    evaluate_partition,
    create_k_comms
)
from cd_v_partition.causal_discovery import pc, sp_gies
from cd_v_partition.overlapping_partition import modularity_partition, rand_edge_cover_partition, expansive_causal_partition, partition_problem
from cd_v_partition.vis_partition import create_partition_plot
from cd_v_partition.fusion import fusion


num_nodes=50
num_samples=1e4
alpha=0.5
num_comms=2
hard_partition, comm_graph = create_k_comms(graph_type="scale_free", n=int(num_nodes/num_comms), 
                                                        m_list=[1,2], 
                                                        p_list=num_comms*[0.2], 
                                                        k=num_comms,tune_mod=1)
# Generate a random network and corresponding dataset
(edges, nodes, _, _), df = get_data_from_graph(
                list(np.arange(num_nodes)),
                list(comm_graph.edges()),
                nsamples=int(num_samples),
                iv_samples=0,bias=None, var=None
            )
G_star = edge_to_adj(list(edges), nodes=nodes)

# Find the 'superstructure'
df_obs = df.drop(columns=["target"])
data_obs = df_obs.to_numpy()
superstructure, p_values = pc(data_obs, alpha=alpha, outdir=None)
print("Found superstructure")

# Call the causal learner on the full data A(X_v) and superstructure
A_X_v = sp_gies(df, skel=superstructure, outdir=None)

# Partition the superstructure and the dataset
mod_partition = modularity_partition(superstructure)
causal_partition =expansive_causal_partition(superstructure, mod_partition) # adapt the modularity partition
edge_cover_partition = rand_edge_cover_partition(superstructure, mod_partition) # adapt the modularity partition randomly 

partition_schemes = {"default":hard_partition, "modularity":mod_partition, "causal":causal_partition, "edge_cover":edge_cover_partition}

# For each partition scheme run parallel causal discovery
for name, partition in partition_schemes.items():
    subproblems = partition_problem(partition, superstructure, df)

    # Visualize the partition
    superstructure_net = adj_to_dag(
        superstructure
    )  # undirected edges in superstructure adjacency become bidirected
    evaluate_partition(partition, superstructure_net, nodes)
    create_partition_plot(
        superstructure_net,
        nodes,
        partition,
        "./examples/{}_partition.png".format(name),
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
    fused_A_X_s = fusion(partition, results, data_obs)

    delta_causality(A_X_v, fused_A_X_s, G_star)
    
    # Save the partition adjacency matrix
    np.savetxt("./examples/adj_{}_partition.txt".format(name), fused_A_X_s)
    # np.savetxt("./examples/{}_partition.txt".format(name), partition)

# Save 
np.savetxt("./examples/adj_serial.txt", A_X_v)
np.savetxt("./examples/adj_true.txt", G_star)