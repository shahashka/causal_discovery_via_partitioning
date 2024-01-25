import networkx as nx
import numpy as np
from cd_v_partition.vis_partition import create_partition_plot
from cd_v_partition.overlapping_partition import partition_problem, hierarchical_partition
import seaborn as sns
import pandas as pd
from cd_v_partition.utils import (
    adj_to_dag,
    get_data_from_graph,
    adj_to_edge,
    get_scores,
    artificial_superstructure
)
from cd_v_partition.causal_discovery import sp_gies, pc
from cd_v_partition.fusion import screen_projections, fusion, fusion_basic
from cd_v_partition.overlapping_partition import rand_edge_cover_partition, expansive_causal_partition, modularity_partition, PEF_partition
import functools
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random 
import time

def _local_structure_learn(subproblem):
    skel, data = subproblem
    adj_mat = sp_gies(data, skel=skel, outdir=None)
    return adj_mat
    
def run_causal_discovery(superstructure, partition, df, G_star, run_serial=True):

    # Break up problem according to provided partition
    subproblems = partition_problem(partition, superstructure, df)

    # Local learn
    func_partial = functools.partial(_local_structure_learn)
    results = []
    num_partitions = len(partition)
    nthreads = 16
    chunksize = max(1, num_partitions // nthreads)
    print("Launching processes")
    start = time.time()
    with ProcessPoolExecutor(max_workers=nthreads) as executor:
        for result in executor.map(func_partial, subproblems, chunksize=chunksize):
            results.append(result)

    # Merge globally
    data_obs = df.drop(columns=["target"]).to_numpy()
    est_graph_partition = fusion(partition, results, data_obs)
    print("Partition run took {} (s)".format(time.time()-start))
    #est_graph_partition = screen_projections(partition, results)

    # Call serial method
    scores_serial = np.zeros(5)
    if run_serial:
        start = time.time()
        est_graph_serial = _local_structure_learn([superstructure, df])
        print("Serial run took {} (s)".format(time.time()-start))
        scores_serial = get_scores(["CD-serial"], [est_graph_serial], G_star)

    # Compare causal metrics
    # d_scores = delta_causality(est_graph_serial, est_graph_partition, G_star)
    scores_part = get_scores(["CD-partition"], [est_graph_partition], G_star)

    return scores_serial[-2:], scores_part[-2:]  # this is the  true positive rate

def vis(name, partition, superstructure):
    superstructure = adj_to_dag(superstructure)
    create_partition_plot(superstructure, nodes=np.arange(len(superstructure.nodes())),
                          partition=partition, save_name="./examples/{}_partition.png".format(name))


def vis_violin_plot(ax, scores, score_names, samples):
    sns.violinplot(data=scores, x='samples', y='TPR', hue='variable',
               order=samples, hue_order=score_names,
               inner='point', common_norm=False, ax=ax)


def run():
    G_star = np.loadtxt("./datasets/bionetworks/ecoli/synthetic_copies/net_0.txt")
    nodes = np.arange(G_star.shape[0])
    edges = adj_to_edge(G_star, nodes, ignore_weights=True)
    df = get_data_from_graph(nodes, edges, nsamples=int(1e4), iv_samples=0, bias=None, var=None)[-1]
    superstructure = artificial_superstructure(G_star, frac_extraneous=0.1)
    
    print("Start Hierarchical partitioning...")
    start = time.time()
    init_partition = modularity_partition(superstructure)
    print("Time for partitioning {}".format(time.time() - start))
    pd.DataFrame(list(zip(init_partition.keys(), init_partition.values()))).to_csv("./examples/ecoli_partition.csv", header=["comm id", "node list"], index=False)
    #vis("modular_part", init_partition, G_star)
    print("Modularity partition")
    ss, sp_h = run_causal_discovery(superstructure, init_partition, df, G_star)
    
    causal_partition = expansive_causal_partition(superstructure, init_partition)
    #vis("causal_part", causal_partition, G_star)
    print("Causal partition")
    _, sp_c = run_causal_discovery(superstructure, causal_partition, df, G_star, run_serial=False)
    
    pef_partition = PEF_partition(df)
   # vis("pef_part", pef_partition, G_star)
    print("PEF partition")
    _, sp_c = run_causal_discovery(superstructure, pef_partition, df, G_star, run_serial=False)

    ec_partition = rand_edge_cover_partition(superstructure, init_partition)
    #vis("edge_cover_part", pef_partition, G_star)
    print("Edge Coverage partition partition")
    _, sp_c = run_causal_discovery(superstructure, ec_partition, df, G_star, run_serial=False)


if __name__ == "__main__":
    run()