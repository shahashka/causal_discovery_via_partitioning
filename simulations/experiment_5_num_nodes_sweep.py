# Experiment 5: hierarchical networks, num samples 1e5, 
# artificial superstructure with 10% extraneous edges, 
# num_trials=30, default modularity (rho=0.01), fusion + screen projections
# Sweep the number of nodes 50 5e4

import networkx as nx
import numpy as np
from cd_v_partition.vis_partition import create_partition_plot
from cd_v_partition.overlapping_partition import partition_problem, PEF_partition, rand_edge_cover_partition, expansive_causal_partition, modularity_partition
import seaborn as sns
import pandas as pd
from cd_v_partition.utils import (
    get_data_from_graph,
    edge_to_adj,
    artificial_superstructure,
    get_scores, adj_to_edge, get_random_graph_data, directed_heirarchical_graph
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
    
def run_causal_discovery(dir_name, save_name, superstructure, partition, df, G_star, nthreads=16, run_serial=False, full_cand_set=False, screen=False):

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
        est_graph_partition = fusion(superstructure, partition, results, data_obs, full_cand_set=full_cand_set)
    time_partition = time.time() - start
    
    # Save the edge list
    learned_edges = list(est_graph_partition.edges(data=False))
    pd.DataFrame(data=learned_edges, columns=['node1', 'node2']).to_csv("{}/edges_{}.csv".format(dir_name, save_name), index=False)
    
    
    # Call serial method
    scores_serial = np.zeros(5)
    time_serial = 0
    if run_serial:
        start = time.time()
        est_graph_serial = _local_structure_learn([superstructure, df])
        time_serial = time.time() - start
        scores_serial = get_scores(["CD-serial"], [est_graph_serial], G_star)
        learned_edges = adj_to_edge(est_graph_serial, nodes=list(np.arange(est_graph_serial.shape[0])), ignore_weights=True)
        pd.DataFrame(data=learned_edges, columns=['node1', 'node2']).to_csv("{}/edges_serial.csv".format(dir_name), index=False)
        
    scores_part = get_scores(["CD-partition"], [est_graph_partition], G_star)

    return scores_serial, scores_part, time_serial, time_partition
    
def run_nnodes(experiment_dir, num_repeats, nnodes_range, nthreads=16, screen=False):
    nsamples = 1e5
    scores_serial = np.zeros((num_repeats, len(nnodes_range), 6))
    scores_edge_cover = np.zeros((num_repeats, len(nnodes_range), 6))
    scores_causal_partition = np.zeros((num_repeats, len(nnodes_range), 6))
    scores_mod_partition = np.zeros((num_repeats, len(nnodes_range), 6))
    # scores_pef = np.zeros((num_repeats, len(nnodes_range), 6))

    for i in range(num_repeats):
        for j,nnodes in enumerate(nnodes_range):
            print("Number of nodes {}".format(nnodes))

            # Generate data
            G_dir = directed_heirarchical_graph(nnodes)
            (edges, nodes, _, _), df = get_data_from_graph(list(np.arange(len(G_dir.nodes()))), list(G_dir.edges()), nsamples=int(nsamples), iv_samples=0, bias=None, var=None)
            #(edges, nodes, _, _), df = get_random_graph_data("hierarchical", num_nodes=nnodes, nsamples=int(nsamples), iv_samples=0, p=0.5, m=2)
            
            dir_name = "./{}/screen_projections/nnodes_{}/{}/".format(experiment_dir, nnodes, i) if screen else "./{}/fusion/nnodes_{}/{}/".format(experiment_dir, nnodes, i)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            
            # Save true graph and data 
            df.to_csv("{}/data.csv".format(dir_name), header=True, index=False)
            pd.DataFrame(data=np.array(edges), columns=['node1', 'node2']).to_csv("{}/edges_true.csv".format(dir_name), index=False)
            G_star = edge_to_adj(edges, nodes)
        
            # Find superstructure 
            superstructure = artificial_superstructure(G_star, frac_extraneous=0.1)
            superstructure_edges = adj_to_edge(superstructure, nodes, ignore_weights=True)
            pd.DataFrame(data=np.array(superstructure_edges), columns=['node1', 'node2']).to_csv("{}/edges_ss.csv".format(dir_name), index=False)

            
            # Run each partition and get scores 
            start = time.time()
            mod_partition = modularity_partition(superstructure, cutoff=1, best_n=None)
            tm = time.time() - start
            
            ss, sp, ts, tp = run_causal_discovery(dir_name, "mod",superstructure, mod_partition, df, G_star, nthreads=nthreads, screen=screen, run_serial=True)
            scores_serial[i][j][0:5] = ss
            scores_mod_partition[i][j][0:5] = sp
            
            scores_serial[i][j][-1] = ts
            scores_mod_partition[i][j][-1] = tp + tm 

            start = time.time()
            partition = rand_edge_cover_partition(superstructure, mod_partition)
            tec = time.time() - start

            _, sp, _ , tp = run_causal_discovery(dir_name, "edge_cover",superstructure, partition, df, G_star, nthreads=nthreads,screen=screen)
            scores_edge_cover[i][j][0:5] = sp
            scores_edge_cover[i][j][-1] = tp + tec + tm 

            start = time.time()
            partition = expansive_causal_partition(superstructure, mod_partition)
            tca = time.time() - start
            
            _, sp, _, tp = run_causal_discovery(dir_name, "causal",superstructure, partition, df, G_star, nthreads=nthreads,screen=screen)
            scores_causal_partition[i][j][0:5] = sp
            scores_causal_partition[i][j][-1] = tp + tca + tm 

            # start = time.time()
            # partition = PEF_partition(df)
            # tpef = time.time()-start
            # print(tpef)
            
            # _, sp, _, tp = run_causal_discovery(dir_name, "pef",superstructure, partition, df, G_star, nthreads=nthreads,screen=screen, full_cand_set=True)
            # scores_pef[i][j][0:5] = sp
            # scores_pef[i][j][-1] = tp + tpef 
            # print(tp+tpef)

            


    plt.clf()
    fig, axs = plt.subplots(3, figsize=(10,12),sharex=True)

    tpr_ind = -3
    data = [scores_serial[:,:,tpr_ind], scores_edge_cover[:,:,tpr_ind], scores_causal_partition[:,:,tpr_ind], scores_mod_partition[:,:,tpr_ind]] 
    data = [np.reshape(d, num_repeats*len(nnodes_range)) for d in data]
    labels = [ 'serial', 'edge_cover', 'expansive_causal', 'mod'] 
    df = pd.DataFrame(data=np.column_stack(data), columns=labels)
    df['samples'] = np.repeat([nnodes_range], num_repeats, axis=0).flatten() 
    df = df.melt(id_vars='samples', value_vars=labels)
    x_order = np.unique(df['samples'])
    g = sns.boxplot(data=df, x='samples', y='value', hue='variable', order=x_order, hue_order=labels, ax=axs[0])
    axs[0].set_xlabel("Number of nodes")
    axs[0].set_ylabel("TPR")
    
    fpr_ind = -2
    data = [scores_serial[:,:,fpr_ind], scores_edge_cover[:,:,fpr_ind], scores_causal_partition[:,:,fpr_ind], scores_mod_partition[:,:,fpr_ind]] 
    data = [np.reshape(d, num_repeats*len(nnodes_range)) for d in data]
    labels = [ 'serial', 'edge_cover', 'expansive_causal', 'mod'] 
    df = pd.DataFrame(data=np.column_stack(data), columns=labels)
    df['samples'] = np.repeat([nnodes_range], num_repeats, axis=0).flatten() 
    df = df.melt(id_vars='samples', value_vars=labels)
    x_order = np.unique(df['samples'])
    sns.boxplot(data=df, x='samples', y='value', hue='variable', order=x_order, hue_order=labels, ax=axs[1], legend=False)
    axs[1].set_xlabel("Number of nodes")
    axs[1].set_ylabel("FPR")
    
    shd_ind = 0
    data = [scores_serial[:,:,shd_ind], scores_edge_cover[:,:,shd_ind], scores_causal_partition[:,:,shd_ind], scores_mod_partition[:,:,shd_ind]] 
    data = [np.reshape(d, num_repeats*len(nnodes_range)) for d in data]
    labels = [ 'serial', 'edge_cover', 'expansive_causal', 'mod'] 
    df = pd.DataFrame(data=np.column_stack(data), columns=labels)
    df['samples'] = np.repeat([nnodes_range], num_repeats, axis=0).flatten() 
    df = df.melt(id_vars='samples', value_vars=labels)
    x_order = np.unique(df['samples'])
    sns.boxplot(data=df, x='samples', y='value', hue='variable', order=x_order, hue_order=labels, ax=axs[2], legend=False)
    axs[2].set_xlabel("Number of nodes")
    axs[2].set_ylabel("SHD")
    
    sns.move_legend(g, "center left", bbox_to_anchor=(1, .5), title='Algorithm')

    plt.tight_layout()
    plot_dir = "./{}/screen_projections/".format(experiment_dir) if screen else "./{}/fusion/".format(experiment_dir)
    plt.savefig("{}/fig.png".format(plot_dir))
    
    
    plt.clf()
    fig, ax = plt.subplots()

    time_ind = -1
    data = [scores_serial[:,:,time_ind], scores_edge_cover[:,:,time_ind], scores_causal_partition[:,:,time_ind], scores_mod_partition[:,:,time_ind]] 
    data = [np.reshape(d, num_repeats*len(nnodes_range)) for d in data]
    labels = [ 'serial', 'edge_cover', 'expansive_causal', 'mod'] 
    df = pd.DataFrame(data=np.column_stack(data), columns=labels)
    df['samples'] = np.repeat([nnodes_range], num_repeats, axis=0).flatten() 
    df = df.melt(id_vars='samples', value_vars=labels)
    x_order = np.unique(df['samples'])
    g = sns.boxplot(data=df, x='samples', y='value', hue='variable', order=x_order, hue_order=labels, ax=ax)
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Time to solution (s)")
    plt.savefig("{}/time.png".format(plot_dir))

    # Save score matrices
    np.savetxt("{}/scores_serial.txt".format(plot_dir), scores_serial.reshape(num_repeats, -1))
    # np.savetxt("{}/scores_pef.txt".format(plot_dir), scores_pef.reshape(num_repeats, -1))
    np.savetxt("{}/scores_edge_cover.txt".format(plot_dir), scores_edge_cover.reshape(num_repeats, -1))
    np.savetxt("{}/scores_causal_partition.txt".format(plot_dir), scores_causal_partition.reshape(num_repeats, -1))
    np.savetxt("{}/scores_mod.txt".format(plot_dir), scores_mod_partition.reshape(num_repeats, -1))

    

if __name__ == "__main__":
    # Simple case for debugging
    # run_nnodes("./simulations/experiment_5_test/", nthreads=16, num_repeats=1, nnodes_range=[10**i for i in np.arange(1,2)], screen=False)
    # run_nnodes("./simulations/experiment_5_test/", nthreads=16, num_repeats=1, nnodes_range=[10**i for i in np.arange(1,2)], screen=True)

    run_nnodes("./simulations/experiment_5_no_pef/", nthreads=16, num_repeats=10, nnodes_range=[10**i for i in np.arange(1,5)], screen=False)
    run_nnodes("./simulations/experiment_5_no_pef/", nthreads=16, num_repeats=10, nnodes_range=[10**i for i in np.arange(1,5)], screen=True)
