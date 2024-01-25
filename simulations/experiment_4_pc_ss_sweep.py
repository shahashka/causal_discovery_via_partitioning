# Experiment 4: two community, scale free, num samples 1e5, num_nodes=50, 
# num_trials=30, default modularity (rho=0.01), fusion + screen projections
# Sweep PC alpha parameter which controls 
# the density of the supsertructure 

import networkx as nx
import numpy as np
from cd_v_partition.vis_partition import create_partition_plot
from cd_v_partition.overlapping_partition import partition_problem, PEF_partition, rand_edge_cover_partition, expansive_causal_partition, modularity_partition
import seaborn as sns
import pandas as pd
from cd_v_partition.utils import (
    get_data_from_graph,
    edge_to_adj,
    create_k_comms,
    artificial_superstructure,
    get_scores, adj_to_edge
)
from cd_v_partition.causal_discovery import sp_gies, pc
from cd_v_partition.fusion import screen_projections, fusion
import functools
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import os
import time

def _local_structure_learn(subproblem):
    skel, data = subproblem
    adj_mat = sp_gies(data, skel=skel, outdir=None)
    return adj_mat
    
def run_causal_discovery(superstructure, partition, df, G_star, nthreads=16, run_serial=False, full_cand_set=False, screen=False):

    start = time.time()
    # Break up problem according to provided partition
    subproblems = partition_problem(partition, superstructure, df)

    # Local learn
    func_partial = functools.partial(_local_structure_learn)
    results = []
    num_partitions = len(partition)
    chunksize = max(1, num_partitions // nthreads)
    print("Launching processes")

    with ProcessPoolExecutor(max_workers=nthreads) as executor:
        for result in executor.map(func_partial, subproblems, chunksize=chunksize):
            results.append(result)

    # Merge globally
    data_obs = df.drop(columns=["target"]).to_numpy()
    if screen:
        est_graph_partition = screen_projections(partition, results)
    else:
        est_graph_partition = fusion(partition, results, data_obs, full_cand_set=full_cand_set)
    time_partition = time.time() - start

    # Call serial method
    scores_serial = np.zeros(5)
    time_serial = 0
    if run_serial:
        start = time.time()
        est_graph_serial = _local_structure_learn([superstructure, df])
        time_serial = time.time() - start
        scores_serial = get_scores(["CD-serial"], [est_graph_serial], G_star)

    scores_part = get_scores(["CD-partition"], [est_graph_partition], G_star)

    return scores_serial, scores_part, time_serial, time_partition
    
def run_ss_pc(experiment_dir, num_repeats, alpha_range, nthreads=16, screen=False):
    nsamples = 1e5
    scores_serial = np.zeros((num_repeats, len(alpha_range), 5))
    scores_edge_cover = np.zeros((num_repeats, len(alpha_range), 5))
    scores_causal_partition = np.zeros((num_repeats, len(alpha_range), 5))
    scores_mod_partition = np.zeros((num_repeats, len(alpha_range), 5))
    scores_pef = np.zeros((num_repeats, len(alpha_range), 5))

    for i in range(num_repeats):

        init_partition, graph = create_k_comms(
            "scale_free", n=25, m_list=[1,2], p_list=[0.5,0.5], k=2
        )
            
        num_nodes = len(graph.nodes())
        bias = np.random.normal(0, 1, size=num_nodes)
        var = np.abs(np.random.normal(0, 1, size=num_nodes))
            
        # Generate data
        (edges, nodes, _, _), df = get_data_from_graph(
            list(np.arange(num_nodes)),
            list(graph.edges()),
            nsamples=int(nsamples),
            iv_samples=0,bias=bias, var=var
        )
        for j,alpha in enumerate(alpha_range):
            
            dir_name = "./{}/screen_projections/alpha_{}/{}/".format(experiment_dir, alpha, i) if screen else "./{}/fusion/alpha_{}/{}/".format(experiment_dir, alpha, i)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            print("Alpha {}".format(alpha))
            
            # Save true graph and data 
            df.to_csv("{}/data.csv".format(dir_name), header=True, index=False)
            pd.DataFrame(data=np.array(edges), columns=['node1', 'node2']).to_csv("{}/edges_true.csv".format(dir_name), index=False)
            G_star = edge_to_adj(edges, nodes)
        
            # Find superstructure 
            superstructure, _ = pc(df, alpha=alpha, num_cores=nthreads, outdir=None)
            superstructure_edges = adj_to_edge(superstructure, nodes, ignore_weights=True)
            pd.DataFrame(data=np.array(superstructure_edges), columns=['node1', 'node2']).to_csv("{}/edges_ss.csv".format(dir_name), index=False)

            
            # Run each partition and get scores 
            mod_partition = modularity_partition(superstructure, cutoff=1, best_n=None)
            ss, sp, ts, tp = run_causal_discovery(superstructure, mod_partition, df, G_star, nthreads=nthreads, screen=screen, run_serial=True)
            scores_serial[i][j] = ss
            scores_mod_partition[i][j] = sp
            
            partition = rand_edge_cover_partition(superstructure, mod_partition)
            _, sp, _ , tp = run_causal_discovery(superstructure, partition, df, G_star, nthreads=nthreads,screen=screen)
            scores_edge_cover[i][j] = sp
            
            partition = expansive_causal_partition(superstructure, mod_partition)
            _, sp, _, tp = run_causal_discovery(superstructure, partition, df, G_star, nthreads=nthreads,screen=screen)
            scores_causal_partition[i][j] = sp
            
            partition = PEF_partition(df)
            _, sp, _, tp = run_causal_discovery(superstructure, partition, df, G_star, nthreads=nthreads,screen=screen, full_cand_set=True)
            scores_pef[i][j] = sp
            


    plt.clf()
    fig, axs = plt.subplots(3, figsize=(10,12),sharex=True)

    tpr_ind = -2
    data = [scores_serial[:,:,tpr_ind], scores_pef[:,:,tpr_ind], scores_edge_cover[:,:,tpr_ind], scores_causal_partition[:,:,tpr_ind], scores_mod_partition[:,:,tpr_ind]] 
    data = [np.reshape(d, num_repeats*len(alpha_range)) for d in data]
    labels = [ 'serial', 'pef' , 'edge_cover', 'expansive_causal', 'mod'] 
    df = pd.DataFrame(data=np.column_stack(data), columns=labels)
    df['samples'] = np.repeat([alpha_range], num_repeats, axis=0).flatten() 
    df = df.melt(id_vars='samples', value_vars=labels)
    x_order = np.unique(df['samples'])
    g = sns.boxplot(data=df, x='samples', y='value', hue='variable', order=x_order, hue_order=labels, ax=axs[0])
    axs[0].set_xlabel("Alpha")
    axs[0].set_ylabel("TPR")
    
    fpr_ind = -1
    data = [scores_serial[:,:,fpr_ind], scores_pef[:,:,fpr_ind], scores_edge_cover[:,:,fpr_ind], scores_causal_partition[:,:,fpr_ind], scores_mod_partition[:,:,fpr_ind]] 
    data = [np.reshape(d, num_repeats*len(alpha_range)) for d in data]
    labels = [ 'serial', 'pef' , 'edge_cover', 'expansive_causal', 'mod'] 
    df = pd.DataFrame(data=np.column_stack(data), columns=labels)
    df['samples'] = np.repeat([alpha_range], num_repeats, axis=0).flatten() 
    df = df.melt(id_vars='samples', value_vars=labels)
    x_order = np.unique(df['samples'])
    sns.boxplot(data=df, x='samples', y='value', hue='variable', order=x_order, hue_order=labels, ax=axs[1], legend=False)
    axs[1].set_xlabel("Alpha")
    axs[1].set_ylabel("FPR")
    
    shd_ind = 0
    data = [scores_serial[:,:,shd_ind], scores_pef[:,:,shd_ind], scores_edge_cover[:,:,shd_ind], scores_causal_partition[:,:,shd_ind], scores_mod_partition[:,:,shd_ind]] 
    data = [np.reshape(d, num_repeats*len(alpha_range)) for d in data]
    labels = [ 'serial', 'pef' , 'edge_cover', 'expansive_causal', 'mod'] 
    df = pd.DataFrame(data=np.column_stack(data), columns=labels)
    df['samples'] = np.repeat([alpha_range], num_repeats, axis=0).flatten() 
    df = df.melt(id_vars='samples', value_vars=labels)
    x_order = np.unique(df['samples'])
    sns.boxplot(data=df, x='samples', y='value', hue='variable', order=x_order, hue_order=labels, ax=axs[2], legend=False)
    axs[2].set_xlabel("Alpha")
    axs[2].set_ylabel("SHD")
    
    sns.move_legend(g, "center left", bbox_to_anchor=(1, .5), title='Algorithm')

    plt.tight_layout()
    plot_dir = "./{}/screen_projections/".format(experiment_dir) if screen else "./{}/fusion/".format(experiment_dir)
    plt.savefig("{}/fig.png".format(plot_dir))
    
    # Save score matrices
    np.savetxt("{}/scores_serial.txt".format(plot_dir), scores_serial.reshape(num_repeats, -1))
    np.savetxt("{}/scores_pef.txt".format(plot_dir), scores_pef.reshape(num_repeats, -1))
    np.savetxt("{}/scores_edge_cover.txt".format(plot_dir), scores_edge_cover.reshape(num_repeats, -1))
    np.savetxt("{}/scores_causal_partition.txt".format(plot_dir), scores_causal_partition.reshape(num_repeats, -1))
    np.savetxt("{}/scores_mod.txt".format(plot_dir), scores_mod_partition.reshape(num_repeats, -1))



if __name__ == "__main__":
    # Simple case for debugging
    # run_ss_pc("./simulations/experiment_4/", nthreads=16, num_repeats=2, alpha_range=np.arange(0.1,0.5,0.1), screen=False)
    # run_ss_pc("./simulations/experiment_4/", nthreads=16, num_repeats=2, alpha_range=np.arange(0.1,0.5,0.1), screen=True)

    run_ss_pc("./simulations/experiment_4/", nthreads=16, num_repeats=50, alpha_range=np.arange(0.1,1,0.1), screen=False)
    run_ss_pc("./simulations/experiment_4/", nthreads=16, num_repeats=50, alpha_range=np.arange(0.1,1,0.1), screen=True)