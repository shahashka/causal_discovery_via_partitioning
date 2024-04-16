from cd_v_partition.causal_discovery import sp_gies, pc
from cd_v_partition.fusion import screen_projections, fusion, remove_edges_not_in_ss
from cd_v_partition.overlapping_partition import partition_problem
from cd_v_partition.utils import get_scores, adj_to_edge
import functools
from concurrent.futures import ProcessPoolExecutor
import time
import networkx as nx
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dcd.admg_discovery import Discovery
import torch
from dagma import utils
from dagma.linear import DagmaLinear
def admg_to_adj(admg, shape):
    adj_mat = np.zeros(shape)
    for d in admg.di_edges:
        start, end = d
        adj_mat[start,end] = 1
    for b in admg.bi_edges:
        n1, n2 = b
        adj_mat[n1, n2] = 1
        adj_mat[n2,n1] = 1
    return adj_mat
        
def _local_structure_learn_pc(subproblem):
    skel, data = subproblem
    adj, _ = pc(data, alpha=1e-3, num_cores=4, outdir=None)
    return adj 
    
def _local_structure_learn_dagma(subproblem):
    skel, data = subproblem
    data = data.drop(columns=['target']).to_numpy()
    model = DagmaLinear(loss_type='l2')
    adj = model.fit(data, lambda1=0.02)
    return adj 

def _local_structure_learn_dcd(subproblem):
    skel, data = subproblem
    data = data.drop(columns=['target'])
    learn = Discovery(lamda=0.05)
    best_G = learn.discover_admg(data, admg_class="bowfree", verbose=True)
    return admg_to_adj(best_G, skel.shape)

def _local_structure_learn_ges(subproblem):
    skel, data = subproblem
    adj_mat = sp_gies(data, skel=skel, outdir=None)
    return adj_mat

# ss_subset dictates whether we discard edges not in the superstructure
# as part of post-processing, in both the serial and fusion methods.
def run_causal_discovery_partition(
    dir_name,
    save_name,
    superstructure,
    partition,
    df,
    G_star,
    nthreads=16,
    full_cand_set=False,
    screen=False,
    ss_subset=True,
    finite_sample_limit=True
):

    start = time.time()
    # Break up problem according to provided partition
    subproblems = partition_problem(partition, superstructure, df)

    # Local learn
    func_partial = functools.partial(_local_structure_learn_dagma)
    results = []
    num_partitions = len(partition)
    chunksize = max(1, num_partitions // nthreads)
    print("Launching processes")

    with ProcessPoolExecutor(max_workers=nthreads) as executor:
       for result in executor.map(func_partial, subproblems, chunksize=chunksize):
           results.append(result)
    # Debugging
    # for s in subproblems:
    #      sp = time.time()
    #      r = _local_structure_learn(s)
    #      results.append(r)
    #      tp = time.time() - sp
    #      print("Time for local run with {} nodes is {} (s)".format(s[0].shape[0], tp))

    # Merge globally
    data_obs = df.drop(columns=["target"]).to_numpy()
    if screen:
        est_graph_partition = screen_projections(
            superstructure,
            partition,
            results,
            ss_subset=ss_subset,
            finite_lim=finite_sample_limit,
            data=data_obs,
        )
    else:
        est_graph_partition = fusion(
            superstructure, partition, results, data_obs, full_cand_set=full_cand_set, ss_subset=ss_subset
        )
    time_partition = time.time() - start
    
    scores_part = get_scores(["CD-partition"], [est_graph_partition], G_star)

    return  scores_part, time_partition

def run_causal_discovery_serial(
    dir_name,
    superstructure,
    df,
    G_star,
    ss_subset=True,
):
    # Call serial method
    print('start')
    start = time.time()
    est_graph_serial = _local_structure_learn_dagma([superstructure, df])
    print('end')
    # optional post-processing: discard edges not in superstructure
    if ss_subset:
        ss_graph = nx.from_numpy_array(superstructure, create_using=nx.DiGraph)
        est_graph_serial_DiGraph = nx.from_numpy_array(
            est_graph_serial, create_using=nx.DiGraph
        )
        subselected_serial_DiGraph = remove_edges_not_in_ss(
            est_graph_serial_DiGraph, ss_graph
        )
        # convert back to numpy array
        est_graph_serial = nx.to_numpy_array(
            subselected_serial_DiGraph,
            nodelist=np.arange(len(subselected_serial_DiGraph.nodes())),
        )
    time_serial = time.time() - start
    scores_serial = get_scores(["CD-serial"], [est_graph_serial], G_star)
    return scores_serial, time_serial 
    
    
def save(experiment_dir, scores, labels, num_repeats, sample_range, x_axis_name, screen, plot_dir=None, time=True, remove_incomplete=False):
    ig, axs = plt.subplots(2, figsize=(10, 12), sharex=True)

    tpr_ind = -3 if time else -2
    data = [ s[:,:,tpr_ind] for s in scores]
    data = [np.reshape(d, num_repeats * len(sample_range)) for d in data]
    print(data[0].shape, len(data), np.column_stack(data).shape)
    df = pd.DataFrame(data=np.column_stack(data), columns=labels)
    df["samples"] = np.repeat(
        [sample_range], num_repeats, axis=0
    ).flatten()  # samples go 1e2->1e7 1e2->1e7 etc
    df = df.melt(id_vars="samples", value_vars=labels)
    if remove_incomplete:
        df= df[df['value'] != 0] # Remove incomplete rows 
    x_order = np.unique(df["samples"])
    g = sns.boxplot(
        data=df,
        x="samples",
        y="value",
        hue="variable",
        order=x_order,
        hue_order=labels,
        ax=axs[0],
        showfliers=False,
    )
    axs[0].set_xlabel(x_axis_name)
    axs[0].set_ylabel("TPR")
    

    shd_ind = 0
    data = [ s[:,:,shd_ind] for s in scores]
    data = [np.reshape(d, num_repeats * len(sample_range)) for d in data]
    df = pd.DataFrame(data=np.column_stack(data), columns=labels)
    df["samples"] = np.repeat(
        [sample_range], num_repeats, axis=0
    ).flatten()  # samples go 1e2->1e7 1e2->1e7 etc
    df = df.melt(id_vars="samples", value_vars=labels)
    if remove_incomplete:
        df= df[df['value'] != 0] # Remove incomplete rows 
    x_order = np.unique(df["samples"])
    sns.boxplot(
        data=df,
        x="samples",
        y="value",
        hue="variable",
        order=x_order,
        hue_order=labels,
        ax=axs[1],
        legend=False,
        showfliers=False,
    )
    axs[1].set_xlabel(x_axis_name)
    axs[1].set_ylabel("SHD")

    sns.move_legend(g, "center left", bbox_to_anchor=(1, 0.5), title="Algorithm")

    plt.tight_layout()
    if plot_dir is None:
        plot_dir = (
            "./{}/screen_projections/".format(experiment_dir)
            if screen
            else "./{}/fusion/".format(experiment_dir)
        )
    plt.savefig("{}/fig.png".format(plot_dir))

    plt.clf()
    fig, ax = plt.subplots()

    if time:
        time_ind = -1
        data = [ s[:,:,time_ind] for s in scores]

        data = [np.reshape(d, num_repeats * len(sample_range)) for d in data]
        df = pd.DataFrame(data=np.column_stack(data), columns=labels)
        df["samples"] = np.repeat([sample_range], num_repeats, axis=0).flatten()
        df = df.melt(id_vars="samples", value_vars=labels)
        if remove_incomplete:
            df= df[df['value'] != 0] # Remove incomplete rows 
        x_order = np.unique(df["samples"])
        g = sns.boxplot(
            data=df,
            x="samples",
            y="value",
            hue="variable",
            order=x_order,
            hue_order=labels,
            ax=ax,
        )
        g.set(yscale="log")
        ax.set_xlabel(x_axis_name)
        ax.set_ylabel("Time to solution (s)")
        plt.savefig("{}/time.png".format(plot_dir))

    # Save score matrices
    for s, l in zip(scores, labels):
        np.savetxt(
            "{}/scores_{}.txt".format(plot_dir, l), s.reshape(num_repeats, -1)
        )
