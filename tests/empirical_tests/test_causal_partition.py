import networkx as nx
import numpy as np
from cd_v_partition.vis_partition import create_partition_plot
from cd_v_partition.overlapping_partition import partition_problem
from cd_v_partition.utils import (
    adj_to_dag,
    get_data_from_graph,
    delta_causality,
    edge_to_adj,
    create_k_comms
)
from cd_v_partition.causal_discovery import sp_gies, pc
from cd_v_partition.fusion import screen_projections, fusion, fusion_basic
from cd_v_partition.overlapping_partition import rand_edge_cover_partition, expansive_causal_partition, modularity_partition
import functools
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random 

def _local_structure_learn(subproblem):
    skel, data = subproblem
    adj_mat = sp_gies(data, skel=skel, outdir=None)
    return adj_mat
    
def run_causal_discovery(superstructure, partition, df, G_star):

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
    data_obs = df.drop(columns=["target"]).to_numpy()
    est_graph_partition = fusion(partition, results, data_obs)
    #est_graph_partition = screen_projections(partition, results)

    # Call serial method
    est_graph_serial = _local_structure_learn([superstructure, df])

    # Compare causal metrics
    d_scores = delta_causality(est_graph_serial, est_graph_partition, G_star)
    return d_scores[-2]  # this is the delta true positive rate

def vis(name, partition, superstructure):
    superstructure = adj_to_dag(superstructure)
    create_partition_plot(superstructure, nodes=np.arange(len(superstructure.nodes())),
                          partition=partition, save_name="./tests/empirical_tests/{}_partition.png".format(name))


def artificial_superstructure(
    G_star_adj_mat, frac_retain_direction=0.1, frac_extraneous=0.5
):
    """Creates a superstructure by discarding some of the directions in edges of G_star and adding
    extraneous edges.

    Args:
        G_star_adj_mat (np.ndarray): the adjacency matrix for the target graph
        frac_retain_direction (float): what percentage of edges will retain their direction information
        frac_extraneous (float): adds frac_extraneous*m many additional edges, for m the number of edges in G_star

    Returns:
        super_adj_mat (np.ndarray): an adjacency matrix for the superstructure we've created
    """
    G_star = nx.from_numpy_array(G_star_adj_mat, create_using=nx.DiGraph())
    true_edge_set = set(G_star.edges())

    # returns a deepcopy
    G_super = G_star.to_undirected()
    # add extraneous edges
    m = G_star.number_of_edges()
    nodes = list(G_star.nodes())
    G_super.add_edges_from(pick_k_random_edges(k=int(frac_extraneous * m), nodes=nodes))

    return nx.adjacency_matrix(G_super).toarray()

def pick_k_random_edges(k, nodes):
    return list(zip(random.choices(nodes, k=k), random.choices(nodes, k=k)))


def run():
    num_repeats = 30
    sample_range = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
    alpha=0.5
    scores_edge_cover = np.zeros((num_repeats, len(sample_range)))
    scores_hard_partition = np.zeros((num_repeats, len(sample_range)))
    scores_causal_partition = np.zeros((num_repeats, len(sample_range)))
    scores_mod_partition = np.zeros((num_repeats, len(sample_range)))

    for i in range(num_repeats):
        init_partition, graph = create_k_comms(
            "scale_free", n=25, m_list=[1,2], p_list=[0.5,0.5], k=2
        )
        num_nodes = len(graph.nodes())
        bias = np.random.normal(0, 1, size=num_nodes)
        var = np.abs(np.random.normal(0, 1, size=num_nodes))
        for j,ns in enumerate(sample_range):
            print("Number of samples {}".format(ns))
            # Generate data
            (edges, nodes, _, _), df = get_data_from_graph(
                list(np.arange(num_nodes)),
                list(graph.edges()),
                nsamples=int(ns),
                iv_samples=0,bias=bias, var=var
            )
            G_star = edge_to_adj(edges, nodes)
            #superstructure = artificial_superstructure(G_star, frac_extraneous=0.1)
            data_obs = df.drop(columns=["target"]).to_numpy()
            superstructure, _ = pc(data_obs, alpha=alpha, outdir=None)


            d_tpr_hard = run_causal_discovery(superstructure, init_partition, df, G_star)
            vis("init", init_partition, G_star)
            scores_hard_partition[i][j] = d_tpr_hard
            print(init_partition)

            partition = rand_edge_cover_partition(superstructure, init_partition)
            vis("edge_cover", partition, G_star)
            d_tpr_ec = run_causal_discovery(superstructure, partition, df, G_star)
            scores_edge_cover[i][j] = d_tpr_ec
            print(partition)
            
            partition = expansive_causal_partition(superstructure, init_partition)
            vis("causal", partition, G_star)
            d_tpr_cp = run_causal_discovery(superstructure, partition, df, G_star)
            scores_causal_partition[i][j] = d_tpr_cp
            print(partition)
            
            partition = modularity_partition(superstructure, cutoff=1, best_n=2)
            vis("mod", partition, G_star)
            d_tpr_mod = run_causal_discovery(superstructure, partition, df, G_star)
            scores_mod_partition[i][j] = d_tpr_mod
            print(partition)

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
    add_label(
        ax.violinplot(scores_mod_partition, showmeans=True, showmedians=False),
        label="modularity_partition",
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
        "./tests/empirical_tests/causal_part_test_ss.png"
    )


if __name__ == "__main__":
    run()
