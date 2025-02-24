# Imports
import numpy as np
import functools
from concurrent.futures import ProcessPoolExecutor
from cd_v_partition.utils import (
    create_k_comms,
    get_data_from_graph,
    edge_to_adj,
    adj_to_dag,
    evaluate_partition,
    get_scores,
)
from cd_v_partition.causal_discovery import pc, ges_local_learn
from cd_v_partition.overlapping_partition import (
    partition_problem,
    expansive_causal_partition,
)
from cd_v_partition.vis_partition import create_partition_plot
from cd_v_partition.fusion import screen_projections, no_partition_postprocess

# This tutorial walks you through setting up a divide-and-conquer framework for estimating
# a causal graph given data using a causal parition. The code is defined by the following 6 steps.
#
# (I) We generate the data from a known causal graph following a linear Gaussian additive noise model.
# (II) From the generated data, we estimate a superstructure over the full node set using the PC algorithm. 
# (III) The superstructure is used to partition the vertex set into a causal partition according to the properties outlined in the paper. 
# (IV) The data set is then split according to the partition, and local graphs are estimated using the GES causal learner.
# (V) The local graphs are merged into a final estimated graph over the full vertex set using the Screen method outlined in the paper.
# (VI) The estimated graph is evaluated using the SHD, AUC, TPR and FPR compared to the known causal graph.
#
# For setting up larger simulations runs, where you want to compare different partitioning algorithms, causal learning algorithms, graph_types, etc..
# over several trials see our experiments in the simulations/ folder. 


# (I)
# Generate a random network and corresponding dataset
# In this toy example we create a 50 node network comprised of 2 communities
# m is the number of edges to attach from a new node to an existing node. The m_list contains the m for each community
# The p_list is ignored here as it only applies to erdos renyi and small world networks
#
# rho controls the interconnectivity of the two communities.
# This is the fraction of total possible edges between the communities to add between the communitites.
# Here we are setting it to 0.1% to promote strong community structure.
num_nodes = 50
num_samples = int(1e5)
n_comms = 2
_, graph = create_k_comms(
    graph_type="scale_free",
    n=int(num_nodes / n_comms),
    m_list=[1, 2],
    p_list=[0.5, 0.5],
    k=n_comms,
    rho=0.001,
)
# Generate the data following the linear Gaussian additive model
(edges, nodes, bias, var), df = get_data_from_graph(
    list(np.arange(num_nodes)),
    list(graph.edges()),
    nsamples=int(num_samples),
    iv_samples=0,
    bias=None,  # defaults random sample [0,1)
    var=None,  # defaults random sample [0,1)
)
G_star = edge_to_adj(edges, nodes)
print(f"Ground truth DAG has {len(edges)} edges")

# Get the data as a numpy array, drop the target column which is not used when all data is observational
data_obs = df.drop(columns="target")
data_obs = data_obs.to_numpy()

# (II)
# Find the superstructure using the PC algorithm.
# Set the signficance threshold to 0.5, this is intentionally high to ensure to our best ability
# that the edges in the superstructure constrain the true causal graph.
#
# In the paper we often use utils.artificial_superstructure for our experiments which assumes access
# to the true causal graph (unrealistic)
superstructure, p_values = pc(
    df, skel=np.ones((num_nodes, num_nodes)), alpha=0.5, outdir=None
)
print(f"Superstructure has {np.sum(superstructure)} edges")

# (III)
# Partition the superstructure and the dataset.
#  Let's use the expansive causal partition from the paper. Other options include:
#  cd_v_partition.overlapping_partition.modularity_partition, rand_edge_cover_partition, PEF_partition
# For this toy example we specify the number of communities in best_n, although this is not needed.
ec_partition = expansive_causal_partition(
    adj_mat=superstructure, data=df, resolution=1, cutoff=1, best_n=2
)
# Visualize the partition over the superstructure
superstructure_net = adj_to_dag(
    superstructure
)  # undirected edges in superstructure adjacency become bidirected
evaluate_partition(ec_partition, superstructure_net, nodes)
create_partition_plot(
    superstructure_net, nodes, ec_partition, "examples/tutorial_partition.png"
)

# (IV)
# Call the causal learner on subsets of the data F({A(X_s)}).
# For this example we will use the GES local learner with outputs a DAG.
#
# Other options are cd_v_partition.causal_discovery.dagma_local_learn, pc_local_learn, rfci_pag_local_learn.
# Note that from our paper we find that empirically the dagma_local_learn and ges_local_learn perform the best.
# Howeover, the theory in the paper only supports consistency for rfci_pag_local_learn.
subproblems = partition_problem(ec_partition, superstructure, df)
# Setup thread level parallelism
func_partial = functools.partial(
    ges_local_learn,
    use_skel=True,  # Use the superstructure as a skeleton for GES learner
)
nthreads = 2  # each thread handles one partition
results = []
chunksize = max(1, n_comms // nthreads)
with ProcessPoolExecutor(max_workers=nthreads) as executor:
    for result in executor.map(func_partial, subproblems, chunksize=chunksize):
        results.append(result)

# (V)
# Merge the subset learned graphs using the Screen algorithm from the paper
#
# Since we are using GES in this tutorial, the merge algorithm is over the DAG space.
# When using a PAG learner like RFCI use the screen_projectections_pag2cpdag method.
#
# In essence the two algorithms are the same (they discard edges that do not agree across subsets),
# but they output different graph.
# screen_projectections_pag2cpdag is implemented exactly as Algorithm 1 in the paper
#
# Screen takes in two flags
# ss_subset = True will drop all edges in the estimated graph that are not in the superstructure
# finite_lim = True will run a post processing algorithm that will resolve any cycles that may be introduced
# during the merge because we are not in the theroretical infinite sample limit.
#
# Also because we are using  a DAG learner, given only observational data, the DAG subgraph is any element
# of the MEC rather than the exact causal subgraph. This means merging can very likely introduce cycles.
merged_A_X_s = screen_projections(
    ss=superstructure,
    partition=ec_partition,
    local_cd_adj_mats=results,
    data=data_obs,
    ss_subset=True,
    finite_lim=True,
)

# (VI)
# Call the causal learner on the full data A(X_v) and superstructure. 
# This is the "No Partition" result from the paper.
A_X_v = ges_local_learn(
    (superstructure, df), use_skel=True
)  # Use the superstructure as a skeleton for GES learner
A_X_v = no_partition_postprocess(
    ss=superstructure, est_adj_mat=A_X_v, ss_subset=True
)  # drop edges outside superstructure

# Get the SHD, AUC, TPR, FPR for each graph compared to the true causal graph 
scores = get_scores(
    ["Expansive Causal partition"], [merged_A_X_s], G_star, get_sid=False
)
print(
    f"Expansive Causal SHD: {scores[0]}, AUC {scores[2]}, TPR: {scores[3]}, FPR {scores[4]}"
)

scores = get_scores(["No partition"], [A_X_v], G_star, get_sid=False)
print(
    f"No partition SHD: {scores[0]}, AUC {scores[2]}, TPR: {scores[3]}, FPR {scores[4]}"
)
