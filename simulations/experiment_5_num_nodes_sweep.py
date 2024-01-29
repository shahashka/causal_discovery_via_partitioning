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
    
def run_causal_discovery(dir_name, save_name, superstructure, partition, df, G_star, nthreads=16, full_cand_set=False, screen=False):
    pd.DataFrame(list(zip(partition.keys(), partition.values()))).to_csv("{}/{}_partition.csv".format(dir_name, save_name), header=["comm id", "node list"], index=False)

    start = time.time()
    # Break up problem according to provided partition
    subproblems = partition_problem(partition, superstructure, df)

    # Local learn
    func_partial = functools.partial(_local_structure_learn)
    results = []
    num_partitions = len(partition)
    chunksize = max(1, num_partitions // nthreads)
    # print("Launching processes")

    with ProcessPoolExecutor(max_workers=nthreads) as executor:
        for result in executor.map(func_partial, subproblems, chunksize=chunksize):
            results.append(result)

    # for s in subproblems:
    #     sp = time.time()
    #     r = _local_structure_learn(s)
    #     results.append(r)
    #     tp = time.time() - sp
    #     print("Time for local run with {} nodes is {} (s)".format(s[0].shape[0], tp))
        
    # Merge globally
    data_obs = df.drop(columns=["target"]).to_numpy()
    if screen:
        est_graph_partition = screen_projections(partition, results)
    else:
        est_graph_partition = fusion(superstructure, partition, results, data_obs, full_cand_set=full_cand_set)
    time_partition = time.time() - start
    print("Time for partitioned run with {} nodes is {} (s)".format(len(est_graph_partition.nodes()), time_partition))

    # Save the edge list
    learned_edges = list(est_graph_partition.edges(data=False))
    pd.DataFrame(data=learned_edges, columns=['node1', 'node2']).to_csv("{}/edges_{}.csv".format(dir_name, save_name), index=False)
    
    scores_part = get_scores(["CD-partition"], [est_graph_partition], G_star)

    return scores_part, time_partition
    
def run_nnodes_alg( algorithm,experiment_dir, num_repeats, nnodes_range, nthreads=16, screen=False):
    nsamples = 1e4
    scores = np.zeros((num_repeats, len(nnodes_range), 6))
    print("Algorithm is {}".format(algorithm))
    for i in range(num_repeats):
        for j,nnodes in enumerate(nnodes_range):
            print("Number of nodes {}".format(nnodes))

            # Generate data
            G_dir = directed_heirarchical_graph(nnodes)
            (edges, nodes, _, _), df = get_data_from_graph(list(np.arange(len(G_dir.nodes()))), list(G_dir.edges()), nsamples=int(nsamples), iv_samples=0, bias=None, var=None)
            #(edges, nodes, _, _), df = get_random_graph_data("hierarchical", num_nodes=nnodes, nsamples=int(nsamples), iv_samples=0, p=0.5, m=2)
            
            dir_name = "./{}/screen_projections/{}/nnodes_{}/{}/".format(experiment_dir,algorithm, nnodes, i) if screen else "./{}/fusion/{}/nnodes_{}/{}/".format(experiment_dir, algorithm, nnodes, i)
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

            if algorithm == 'serial':
                start = time.time()
                est_graph_serial = _local_structure_learn([superstructure, df])
                time_serial = time.time() - start
                print("Time for serial run with {} nodes is {} (s)".format(est_graph_serial.shape[0], time_serial))
        
                ss = get_scores(["CD-serial"], [est_graph_serial], G_star)
                learned_edges = adj_to_edge(est_graph_serial, nodes=list(np.arange(est_graph_serial.shape[0])), ignore_weights=True)
                pd.DataFrame(data=learned_edges, columns=['node1', 'node2']).to_csv("{}/edges_serial.csv".format(dir_name), index=False)
                scores[i][j][0:5] = ss
                scores[i][j][-1] = time_serial
                np.savetxt("{}/time.txt".format(dir_name), scores[i][j])            

            else :
                start = time.time()
                mod_partition = modularity_partition(superstructure , cutoff=1, best_n=None)
                tm = time.time() - start if algorithm != 'pef' else 0 
                biggest_partition = max(len(p) for p in mod_partition.values())
                print("Biggest partition is {}".format(biggest_partition))
                
                full_cand_set = algorithm == 'pef' 
                score, tp = run_causal_discovery(dir_name, algorithm ,superstructure, mod_partition, df, G_star, nthreads=nthreads, screen=screen, full_cand_set=full_cand_set)
                scores[i][j][0:5] = score            
                scores[i][j][-1] = tp + tm 
                np.savetxt("{}/time.txt".format(dir_name), scores[i][j])            




    plt.clf()
    fig, axs = plt.subplots(3, figsize=(10,12),sharex=True)

    tpr_ind = -3
    data = [scores[:,:,tpr_ind]]
    data = [np.reshape(d, num_repeats*len(nnodes_range)) for d in data]
    labels = [ algorithm] 
    df = pd.DataFrame(data=np.column_stack(data), columns=labels)
    df['samples'] = np.repeat([nnodes_range], num_repeats, axis=0).flatten() 
    df = df.melt(id_vars='samples', value_vars=labels)
    x_order = np.unique(df['samples'])
    g = sns.boxplot(data=df, x='samples', y='value', hue='variable', order=x_order, hue_order=labels, ax=axs[0])
    axs[0].set_xlabel("Number of nodes")
    axs[0].set_ylabel("TPR")
    
    fpr_ind = -2
    data = [scores[:,:,fpr_ind]] 
    data = [np.reshape(d, num_repeats*len(nnodes_range)) for d in data]
    labels = [ algorithm] 
    df = pd.DataFrame(data=np.column_stack(data), columns=labels)
    df['samples'] = np.repeat([nnodes_range], num_repeats, axis=0).flatten() 
    df = df.melt(id_vars='samples', value_vars=labels)
    x_order = np.unique(df['samples'])
    sns.boxplot(data=df, x='samples', y='value', hue='variable', order=x_order, hue_order=labels, ax=axs[1], legend=False)
    axs[1].set_xlabel("Number of nodes")
    axs[1].set_ylabel("FPR")
    
    shd_ind = 0
    data = [scores[:,:,shd_ind]] 
    data = [np.reshape(d, num_repeats*len(nnodes_range)) for d in data]
    labels = [algorithm] 
    df = pd.DataFrame(data=np.column_stack(data), columns=labels)
    df['samples'] = np.repeat([nnodes_range], num_repeats, axis=0).flatten() 
    df = df.melt(id_vars='samples', value_vars=labels)
    x_order = np.unique(df['samples'])
    sns.boxplot(data=df, x='samples', y='value', hue='variable', order=x_order, hue_order=labels, ax=axs[2], legend=False)
    axs[2].set_xlabel("Number of nodes")
    axs[2].set_ylabel("SHD")
    
    sns.move_legend(g, "center left", bbox_to_anchor=(1, .5), title='Algorithm')

    plt.tight_layout()
    plot_dir = "./{}/screen_projections/{}".format(experiment_dir, algorithm) if screen else "./{}/fusion/{}".format(experiment_dir, algorithm)
    plt.savefig("{}/fig.png".format(plot_dir))
    
    
    plt.clf()
    fig, ax = plt.subplots()

    time_ind = -1
    data = [scores[:,:,time_ind]] 
    data = [np.reshape(d, num_repeats*len(nnodes_range)) for d in data]
    labels = [ algorithm] 
    df = pd.DataFrame(data=np.column_stack(data), columns=labels)
    df['samples'] = np.repeat([nnodes_range], num_repeats, axis=0).flatten() 
    df = df.melt(id_vars='samples', value_vars=labels)
    x_order = np.unique(df['samples'])
    g = sns.boxplot(data=df, x='samples', y='value', hue='variable', order=x_order, hue_order=labels, ax=ax)
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Time to solution (s)")
    plt.savefig("{}/time.png".format(plot_dir))

    # Save score matrices
    np.savetxt("{}/scores.txt".format(plot_dir), scores.reshape(num_repeats, -1))

    

if __name__ == "__main__":
    # Simple case for debugging
    algorithms = ['serial', 'pef', 'edge_cover', 'causal', 'mod']
    #func_partial = functools.partial(run_nnodes_alg, experiment_dir="./simulations/experiment_5_test/", nthreads=16, num_repeats=2, nnodes_range=[10**i for i in np.arange(1,4)], screen=True )
    func_partial = functools.partial(run_nnodes_alg, experiment_dir="./simulations/experiment_5/", nthreads=16, num_repeats=5, nnodes_range=[10**i for i in np.arange(1,5)], screen=True )
    results = []
    with ProcessPoolExecutor(max_workers=len(algorithms)) as executor:
        for result in executor.map(func_partial, algorithms, chunksize=1):
            results.append(result)
