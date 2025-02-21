# Argue the following point:
# Causal partitions strike a balance between better learning and fast learning
# We can always increase the sizes of the partitions to improve accuracy but then we lose the benefits of scaling
# We can always decrease the sizes of the partitions, but then we are likely to loose out on accuracy
# A causal partition allows us to choose an optimize parition size for our domain -- depending on
# whatever scaling/timing specification it requires -- adapt it, and boost learning accuracy dramatically without incurring a significant compute cost
# To show this, we set up an experiment with a large graph - 1000 nodes. We create several disjoint partitions of varying max sizes...between 10 and 500 nodes
# For each disjoint partition we create a complementary causal partition. We measure accuracy and time to solution for all partitions and for serial runs
# and show the causal partitions has advantegoues in both directions
# Of course we note that the topology of the network is going to impact how big the causal partitions end up being

from cd_v_partition.utils import (
    create_k_comms,
    directed_heirarchical_graph,
    get_data_from_graph,
    artificial_superstructure,
)
from cd_v_partition.overlapping_partition import (
    hierarchical_partition,
    modularity_partition,
    expansive_causal_partition,
    rand_edge_cover_partition,
)
import numpy as np
import networkx as nx
from common_funcs import run_causal_discovery_partition, run_causal_discovery_serial
import matplotlib.pyplot as plt

# Generate data
nnodes = 1000
nsamples = 1e3
num_repeats = 5
# G_dir = directed_heirarchical_graph(nnodes)
k = 10
init_partition, G_dir = create_k_comms(
    "scale_free", int(nnodes / k), k * [2], k * [0.5], k=k
)
(edges, nodes, _, _), df = get_data_from_graph(
    list(np.arange(len(G_dir.nodes()))),
    list(G_dir.edges()),
    nsamples=int(nsamples),
    iv_samples=0,
    bias=None,
    var=None,
)
print(len(G_dir.edges()))
G_star = nx.adjacency_matrix(G_dir, nodes).todense()
ss = artificial_superstructure(G_star, frac_extraneous=0.1)
num_comms = [100, 20, 10, 4, 2]
size_mod = np.zeros((num_repeats, len(num_comms)))
scores_mod = np.zeros((num_repeats, len(num_comms), 5))

time_mod = np.zeros((num_repeats, len(num_comms)))

size_ec = np.zeros((num_repeats, len(num_comms)))
scores_ec = np.zeros((num_repeats, len(num_comms), 5))
time_ec = np.zeros((num_repeats, len(num_comms)))

size_causal = np.zeros((num_repeats, len(num_comms)))
scores_causal = np.zeros((num_repeats, len(num_comms), 5))
time_causal = np.zeros((num_repeats, len(num_comms)))

scores_serial = np.zeros((num_repeats, 5))
time_serial = np.zeros(num_repeats)

for i in np.arange(num_repeats):
    for j, nc in enumerate(num_comms):
        print(num_comms)
        disjoint_partition = modularity_partition(
            ss, resolution=5, cutoff=nc, best_n=nc
        )
        causal_partition = expansive_causal_partition(ss, disjoint_partition)
        edge_coverage_partition = rand_edge_cover_partition(ss, disjoint_partition)

        biggest_partition = max(len(p) for p in disjoint_partition.values())
        print("Biggest disjoint partition is {}".format(biggest_partition))
        size_mod[i][j] = biggest_partition

        biggest_partition = max(len(p) for p in edge_coverage_partition.values())
        print("Biggest edge coverage partition is {}".format(biggest_partition))
        size_ec[i][j] = biggest_partition

        biggest_partition = max(len(p) for p in causal_partition.values())
        print("Biggest causal partition is {}".format(biggest_partition))
        size_causal[i][j] = biggest_partition

        pscore, ptime = run_causal_discovery_partition(
            "./simulations",
            "mod",
            ss,
            disjoint_partition,
            df,
            G_star,
            nthreads=16,
            full_cand_set=False,
            screen=True,
        )
        scores_mod[i][j] = pscore
        time_mod[i][j] = ptime

        pscore, ptime = run_causal_discovery_partition(
            "./simulations",
            "edge_cover",
            ss,
            edge_coverage_partition,
            df,
            G_star,
            nthreads=16,
            full_cand_set=False,
            screen=True,
        )
        scores_ec[i][j] = pscore
        time_ec[i][j] = ptime

        pscore, ptime = run_causal_discovery_partition(
            "./simulations",
            "expansive_causal",
            ss,
            causal_partition,
            df,
            G_star,
            nthreads=16,
            full_cand_set=False,
            screen=True,
        )
        scores_causal[i][j] = pscore
        time_causal[i][j] = ptime

    sscore, sstime = run_causal_discovery_serial("./simulations", ss, df, G_star)
    scores_serial[i] = sscore
    time_serial[i] = sstime
    print("Serial score {}, Serial time {}".format(sscore, sstime))

fig, axs = plt.subplots(2, sharex=True)
tpr_ind = -2
new_dim = num_repeats * len(num_comms)
axs[0].scatter(
    size_mod.reshape(new_dim), scores_mod[:, :, tpr_ind].reshape(new_dim), label="mod"
)
axs[0].scatter(
    size_ec.reshape(new_dim),
    scores_ec[:, :, tpr_ind].reshape(new_dim),
    label="edge_cover",
)
axs[0].scatter(
    size_causal.reshape(new_dim),
    scores_causal[:, :, tpr_ind].reshape(new_dim),
    label="expansive_causal",
)
axs[0].scatter(
    [nnodes for _ in np.arange(num_repeats)], scores_serial[:, tpr_ind], label="serial"
)
axs[0].set_ylabel("TPR")

axs[1].scatter(size_mod.reshape(new_dim), time_mod.reshape(new_dim), label="mod")
axs[1].scatter(size_ec.reshape(new_dim), time_ec.reshape(new_dim), label="edge_cover")
axs[1].scatter(
    size_causal.reshape(new_dim), time_causal.reshape(new_dim), label="expansive_causal"
)
axs[1].scatter([nnodes for _ in np.arange(num_repeats)], time_serial, label="serial")
axs[1].set_ylabel("Time (s)")
axs[1].set_xlabel("Size of largest partition")
plt.legend()
plt.savefig("./simulations/acc_scaling_tradoff.png")

save_arrays_scores = [scores_serial, scores_mod, scores_ec, scores_causal]
save_arrays_time = [time_serial, time_mod, time_ec, time_causal]
save_arrays_size = [size_mod, size_ec, size_causal]

labels = ["serial", "mod", "ec", "causal"]
for s, t, l in zip(save_arrays_scores, save_arrays_time, labels):
    if l == "serial":
        np.savetxt("acc_tradeoff_scores_{}".format(l), s)
    else:
        np.savetxt("acc_tradeoff_scores_{}".format(l), s.reshape(s.shape[0], -1))
    np.savetxt("acc_tradeoff_time_{}".format(l), t)

for size, l in zip(save_arrays_size, labels[1:]):
    np.savetxt("acc_tradeoff_size_{}".format(l), size)
