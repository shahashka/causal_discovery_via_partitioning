import networkx as nx
import numpy as np
from cd_v_partition.vis_partition import create_partition_plot
from cd_v_partition.overlapping_partition import partition_problem
import seaborn as sns
import pandas as pd
from cd_v_partition.utils import (
    adj_to_dag,
    get_data_from_graph,
    delta_causality,
    edge_to_adj,
    create_k_comms,
    artificial_superstructure,
    get_scores
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
    # d_scores = delta_causality(est_graph_serial, est_graph_partition, G_star)
    scores_serial = get_scores(["CD-serial"], [est_graph_serial], G_star)
    scores_part = get_scores(["CD-partition"], [est_graph_partition], G_star)

    return scores_serial[-2], scores_part[-2]  # this is the  true positive rate

def vis(name, partition, superstructure):
    superstructure = adj_to_dag(superstructure)
    create_partition_plot(superstructure, nodes=np.arange(len(superstructure.nodes())),
                          partition=partition, save_name="./tests/empirical_tests/{}_partition.png".format(name))


def vis_violin_plot(ax, scores, score_names, samples):
    sns.violinplot(data=scores, x='samples', y='TPR', hue='variable',
               order=samples, hue_order=score_names,
               inner='point', common_norm=False, ax=ax)

def run():
    num_repeats = 10
    sample_range = [1e2, 1e3, 1e4, 1e5]#, 1e6, 1e7]
    alpha=0.5
    scores_serial = np.zeros((num_repeats, len(sample_range)))
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
            superstructure = artificial_superstructure(G_star, frac_extraneous=0.1)
            #data_obs = df.drop(columns=["target"]).to_numpy()
            #superstructure, _ = pc(data_obs, alpha=alpha, outdir=None)


            ss, sp = run_causal_discovery(superstructure, init_partition, df, G_star)
            vis("init", init_partition, G_star)
            scores_serial[i][j] = ss
            scores_hard_partition[i][j] = sp

            partition = rand_edge_cover_partition(superstructure, init_partition)
            vis("edge_cover", partition, G_star)
            _, sp = run_causal_discovery(superstructure, partition, df, G_star)
            scores_edge_cover[i][j] = sp
            
            partition = expansive_causal_partition(superstructure, init_partition)
            vis("causal", partition, G_star)
            _, sp = run_causal_discovery(superstructure, partition, df, G_star)
            scores_causal_partition[i][j] = sp
            
            partition = modularity_partition(superstructure, cutoff=1, best_n=2)
            vis("mod", partition, G_star)
            _, sp = run_causal_discovery(superstructure, partition, df, G_star)
            scores_mod_partition[i][j] = sp

    # labels = []

    # def add_label(violin, label):
    #     color = violin["bodies"][0].get_facecolor().flatten()
    #     labels.append((mpatches.Patch(color=color), label))

    plt.clf()
    _, ax = plt.subplots()
    # add_label(
    #     ax.violinplot(scores_serial, showmeans=True, showmedians=False),
    #     label="serial",
    # )
    # add_label(
    #     ax.violinplot(scores_edge_cover, showmeans=True, showmedians=False),
    #     label="edge_cover",
    # )
    # add_label(
    #     ax.violinplot(scores_hard_partition, showmeans=True, showmedians=False),
    #     label="hard_partition",
    # )
    # add_label(
    #     ax.violinplot(scores_causal_partition, showmeans=True, showmedians=False),
    #     label="expansive_causal_partition",
    # )
    # add_label(
    #     ax.violinplot(scores_mod_partition, showmeans=True, showmedians=False),
    #     label="modularity_partition",
    # )
    # ax.set_xticks(
    #     np.arange(1, len(sample_range) + 1),
    #     labels=["1e{}".format(i) for i in range(2,len(sample_range)+2)],
    #     rotation=45,
    # )

    data = [scores_serial, scores_edge_cover, scores_causal_partition, scores_hard_partition, scores_mod_partition]
    data = [np.reshape(d, num_repeats*len(sample_range)) for d in data]
    print(data)
    labels = [ 'serial', 'edge_cover' ,'expansive_causal','hard', 'mod']
    df = pd.DataFrame(data=np.column_stack(data), columns=labels)
    df['samples'] = np.repeat(sample_range, num_repeats)
    print(df.head)
    df = df.melt(id_vars='samples', value_vars=labels)
    print(df.head)
    x_order = np.unique(df['samples'])
    sns.violinplot(data=df, x='samples', y='value', hue='variable', order=x_order, hue_order=labels, ax=ax)
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("TPR)")
    ax.set_title("Comparison of partition types for 2 community scale free networks")
    #plt.legend(*zip(*labels), loc=2)
    plt.tight_layout()
    plt.savefig(
        "./tests/empirical_tests/causal_part_test_artificial_ss.png"
    )
    
    np.savetxt("./tests/empirical_tests/scores_serial.txt", scores_serial)
    np.savetxt( "./tests/empirical_tests/scores_edge_cover.txt", scores_edge_cover)
    np.savetxt( "./tests/empirical_tests/scores_hard_partition.txt", scores_hard_partition)
    np.savetxt( "./tests/empirical_tests/scores_causal_partition.txt", scores_causal_partition) 
    np.savetxt( "./tests/empirical_tests/scores_mod_partition.txt", scores_mod_partition)


if __name__ == "__main__":
    run()
