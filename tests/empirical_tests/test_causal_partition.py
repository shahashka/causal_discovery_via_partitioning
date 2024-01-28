import networkx as nx
import numpy as np
from cd_v_partition.vis_partition import create_partition_plot
from cd_v_partition.overlapping_partition import partition_problem, PEF_partition, rand_edge_cover_partition, expansive_causal_partition, modularity_partition
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
import functools
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random 
import os

def _local_structure_learn(subproblem):
    skel, data = subproblem
    adj_mat = sp_gies(data, skel=skel, outdir=None)
    return adj_mat
    
def run_causal_discovery(superstructure, partition, df, G_star, full_cand_set=False):

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
    #est_graph_partition = fusion(partition, results, data_obs, full_cand_set=full_cand_set)
    est_graph_partition = screen_projections(partition, results)

    # Call serial method
    est_graph_serial = _local_structure_learn([superstructure, df])

    # Compare causal metrics
    # d_scores = delta_causality(est_graph_serial, est_graph_partition, G_star)
    scores_serial = get_scores(["CD-serial"], [est_graph_serial], G_star)
    scores_part = get_scores(["CD-partition"], [est_graph_partition], G_star)

    return scores_serial[-2:], scores_part[-2:]  # this is the  true positive rate

def vis(name, partition, superstructure):
    superstructure = adj_to_dag(superstructure)
    create_partition_plot(superstructure, nodes=np.arange(len(superstructure.nodes())),
                          partition=partition, save_name="./tests/empirical_tests/{}_partition.png".format(name))


def vis_box_plot(ax, scores, score_names, samples):
    sns.boxplot(data=scores, x='samples', y='TPR', hue='variable',
               order=samples, hue_order=score_names,
               inner='point', common_norm=False, ax=ax)


def run_tune_mod():
    num_repeats = 10
    ns=1e5
    alpha=0.5
    tune_mod = list(np.arange(0,0.1,0.01))
    scores_serial = np.zeros((num_repeats, len(tune_mod), 2))
    scores_edge_cover = np.zeros((num_repeats, len(tune_mod), 2))
    scores_hard_partition = np.zeros((num_repeats, len(tune_mod), 2))
    scores_causal_partition = np.zeros((num_repeats, len(tune_mod), 2))
    scores_mod_partition = np.zeros((num_repeats, len(tune_mod), 2))
    scores_pef = np.zeros((num_repeats, len(tune_mod), 2))

    for i in range(num_repeats):
        for j,mod in enumerate(tune_mod):

            init_partition, graph = create_k_comms(
                "scale_free", n=25, m_list=[1,2], p_list=[0.5,0.5], k=2, rho=mod
            )
            num_nodes = len(graph.nodes())
            bias = np.random.normal(0, 1, size=num_nodes)
            var = np.abs(np.random.normal(0, 1, size=num_nodes))
            print("Rho {}".format(mod))
            # Generate data
            (edges, nodes, _, _), df = get_data_from_graph(
                list(np.arange(num_nodes)),
                list(graph.edges()),
                nsamples=int(ns),
                iv_samples=0,bias=bias, var=var
            )
            G_star = edge_to_adj(edges, nodes)
            superstructure = artificial_superstructure(G_star, frac_extraneous=0.1)
            #superstructure, _ = pc(df, alpha=alpha, outdir=None)

            print("Modularity is {}".format(nx.community.modularity(nx.from_numpy_array(G_star), init_partition.values())))

            # ss, sp = run_causal_discovery(superstructure, init_partition, df, G_star)
            # vis("init", init_partition, G_star)
            # scores_serial[i][j] = ss
            # scores_hard_partition[i][j] = sp


            
            mod_partition = modularity_partition(superstructure, cutoff=1, best_n=None)
            
            vis("mod", mod_partition, G_star)
            ss, sp = run_causal_discovery(superstructure, mod_partition, df, G_star)
            scores_serial[i][j] = ss
            scores_mod_partition[i][j] = sp
            
            partition = rand_edge_cover_partition(superstructure, mod_partition)
            vis("edge_cover", partition, G_star)
            _, sp = run_causal_discovery(superstructure, partition, df, G_star)
            scores_edge_cover[i][j] = sp
            
            partition = expansive_causal_partition(superstructure, mod_partition)
            vis("causal", partition, G_star)
            _, sp = run_causal_discovery(superstructure, partition, df, G_star)
            scores_causal_partition[i][j] = sp
            
            partition = PEF_partition(df)
            vis("pef", partition, G_star)
            _, sp = run_causal_discovery(superstructure, partition, df, G_star, full_cand_set=True)
            scores_pef[i][j] = sp
        

    plt.clf()
    fig, axs = plt.subplots(2, figsize=(10,8),sharex=True)
    plt.title("2 community scale free modularity sweep")
    
    data = [scores_serial[:,:,0], scores_pef[:,:,0], scores_edge_cover[:,:,0], scores_causal_partition[:,:,0], scores_mod_partition[:,:,0]] # scores_hard_partition
    data = [np.reshape(d, num_repeats*len(tune_mod)) for d in data]
    print(data)
    labels = [ 'serial', 'pef' , 'edge_cover', 'expansive_causal', 'mod'] # 'hard'
    df = pd.DataFrame(data=np.column_stack(data), columns=labels)
    df['samples'] = np.repeat(tune_mod, num_repeats)
    df = df.melt(id_vars='samples', value_vars=labels)
    x_order = np.unique(df['samples'])
    g = sns.boxplot(data=df, x='samples', y='value', hue='variable', order=x_order, hue_order=labels, ax=axs[0])
    axs[0].set_xlabel("Rho")
    axs[0].set_ylabel("TPR")
    
    
    data = [scores_serial[:,:,1], scores_pef[:,:,1], scores_edge_cover[:,:,1], scores_causal_partition[:,:,1], scores_mod_partition[:,:,1]] # scores_hard_partition
    data = [np.reshape(d, num_repeats*len(tune_mod)) for d in data]
    labels = [ 'serial', 'pef' , 'edge_cover', 'expansive_causal', 'mod'] # 'hard'
    df = pd.DataFrame(data=np.column_stack(data), columns=labels)
    df['samples'] = np.repeat(tune_mod, num_repeats)
    df = df.melt(id_vars='samples', value_vars=labels)
    x_order = np.unique(df['samples'])
    sns.boxplot(data=df, x='samples', y='value', hue='variable', order=x_order, hue_order=labels, ax=axs[1], legend=None)
    axs[1].set_xlabel("Rho")
    axs[1].set_ylabel("FPR")
    sns.move_legend(g, "center left", bbox_to_anchor=(1, .5), title='Algorithm')

    plt.savefig(
        "./tests/empirical_tests/causal_part_test_tune_mod.png"
    )

    
def run_samples():
    num_repeats = 1
    sample_range = [1e6]#[1e1,1e2, 1e3, 1e4, 1e5]#, 1e6, 1e7]
    alpha=0.5
    scores_serial = np.zeros((num_repeats, len(sample_range), 2))
    scores_edge_cover = np.zeros((num_repeats, len(sample_range), 2))
    scores_hard_partition = np.zeros((num_repeats, len(sample_range), 2))
    scores_causal_partition = np.zeros((num_repeats, len(sample_range), 2))
    scores_mod_partition = np.zeros((num_repeats, len(sample_range), 2))
    scores_pef = np.zeros((num_repeats, len(sample_range), 2))

    for i in range(num_repeats):
        init_partition, graph = create_k_comms(
            "scale_free", n=25, m_list=[1,2], p_list=[0.5,0.5], k=2
        )
        num_nodes = len(graph.nodes())
        bias = np.random.normal(0, 1, size=num_nodes)
        var = np.abs(np.random.normal(0, 1, size=num_nodes))
        for j,ns in enumerate(sample_range):
            if not os.path.exists("./datasets/test_causal_partition_by_sample_{}".format(ns)):
                os.makedirs("./datasets/test_causal_partition_by_sample_{}".format(ns))
                
            print("Number of samples {}".format(ns))
            # Generate data
            (edges, nodes, _, _), df = get_data_from_graph(
                list(np.arange(num_nodes)),
                list(graph.edges()),
                nsamples=int(ns),
                iv_samples=0,bias=bias, var=var
            )
            df.to_csv("./datasets/test_causal_partition_by_sample_{}/data_{}.csv".format(ns, i), header=True, index=False)
            pd.DataFrame(data=np.array(edges), columns=['node1', 'node2']).to_csv("./datasets/test_causal_partition_by_sample_{}/edges_true_{}.csv".format(ns, i), index=False)

            G_star = edge_to_adj(edges, nodes)
            superstructure = artificial_superstructure(G_star, frac_extraneous=0.1)
            #superstructure, _ = pc(df, alpha=alpha, outdir=None)


            # ss, sp = run_causal_discovery(superstructure, init_partition, df, G_star)
            vis("init", init_partition, G_star)
            # scores_serial[i][j] = ss
            # scores_hard_partition[i][j] = sp


            
            mod_partition = modularity_partition(superstructure, cutoff=1, best_n=None)
            vis("mod", mod_partition, G_star)
            ss, sp = run_causal_discovery(superstructure, mod_partition, df, G_star)
            scores_serial[i][j] = ss
            scores_mod_partition[i][j] = sp
            
            partition = rand_edge_cover_partition(superstructure, mod_partition)
            vis("edge_cover", partition, G_star)
            _, sp = run_causal_discovery(superstructure, partition, df, G_star)
            scores_edge_cover[i][j] = sp
            
            partition = expansive_causal_partition(superstructure, mod_partition)
            vis("causal", partition, G_star)
            _, sp = run_causal_discovery(superstructure, partition, df, G_star)
            scores_causal_partition[i][j] = sp
            
            partition = PEF_partition(df)
            vis("pef", partition, G_star)
            _, sp = run_causal_discovery(superstructure, partition, df, G_star, full_cand_set=True)
            scores_pef[i][j] = sp
            


    plt.clf()
    fig, axs = plt.subplots(2, figsize=(10,8),sharex=True)
    plt.title("Comparison of partition types for 2 community scale free networks")

    data = [scores_serial[:,:,0], scores_pef[:,:,0], scores_edge_cover[:,:,0], scores_causal_partition[:,:,0], scores_mod_partition[:,:,0]] # scores_hard_partition
    data = [np.reshape(d, num_repeats*len(sample_range)) for d in data]
    print(data)
    labels = [ 'serial', 'pef' , 'edge_cover', 'expansive_causal', 'mod'] # 'hard'
    df = pd.DataFrame(data=np.column_stack(data), columns=labels)
    df['samples'] = np.repeat(sample_range, num_repeats)
    df = df.melt(id_vars='samples', value_vars=labels)
    x_order = np.unique(df['samples'])
    g = sns.boxplot(data=df, x='samples', y='value', hue='variable', order=x_order, hue_order=labels, ax=axs[0])
    axs[0].set_xlabel("Number of samples")
    axs[0].set_ylabel("TPR")
    
    
    data = [scores_serial[:,:,1], scores_pef[:,:,1], scores_edge_cover[:,:,1], scores_causal_partition[:,:,1], scores_mod_partition[:,:,1]] # scores_hard_partition
    data = [np.reshape(d, num_repeats*len(sample_range)) for d in data]
    labels = [ 'serial', 'pef' , 'edge_cover', 'expansive_causal', 'mod'] # 'hard'
    df = pd.DataFrame(data=np.column_stack(data), columns=labels)
    df['samples'] = np.repeat(sample_range, num_repeats)
    df = df.melt(id_vars='samples', value_vars=labels)
    x_order = np.unique(df['samples'])
    sns.boxplot(data=df, x='samples', y='value', hue='variable', order=x_order, hue_order=labels, ax=axs[1], legend=False)
    axs[1].set_xlabel("Number of samples")
    axs[1].set_ylabel("FPR")
    
    sns.move_legend(g, "center left", bbox_to_anchor=(1, .5), title='Algorithm')

    plt.tight_layout()
    plt.savefig(
        "./tests/empirical_tests/causal_part_test_artificial_ss_pef_10.png"
    )


if __name__ == "__main__":
    #run_tune_mod()
    run_samples()
