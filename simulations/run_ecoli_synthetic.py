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
    artificial_superstructure
)
from cd_v_partition.overlapping_partition import rand_edge_cover_partition, expansive_causal_partition, modularity_partition, PEF_partition
import time
from common_funcs import run_causal_discovery_partition, run_causal_discovery_serial

def vis(name, partition, superstructure):
    superstructure = adj_to_dag(superstructure)
    create_partition_plot(superstructure, nodes=np.arange(len(superstructure.nodes())),
                          partition=partition, save_name="./examples/{}_partition.png".format(name))


def vis_violin_plot(ax, scores, score_names, samples):
    sns.violinplot(data=scores, x='samples', y='TPR', hue='variable',
               order=samples, hue_order=score_names,
               inner='point', common_norm=False, ax=ax)


def run(): 
    G_star = np.loadtxt("./datasets/bionetworks/ecoli/synthetic_copies/net_1.txt")
    #G_star = G_star[0:1000][:,0:1000]
    print(G_star.shape)
    nodes = np.arange(G_star.shape[0])
    edges = adj_to_edge(G_star, nodes, ignore_weights=True)
    df = get_data_from_graph(nodes, edges, nsamples=int(1e4), iv_samples=0, bias=None, var=None)[-1]
    superstructure = artificial_superstructure(G_star, frac_extraneous=0.1)
    dir_name="./simulations/"
    print("Start Mod partitioning...")
    start = time.time()
    init_partition = modularity_partition(superstructure, resolution=5, cutoff=100, best_n=100)
    print("Time for partitioning {}".format(time.time() - start))
    
    biggest_partition = max(len(p) for p in init_partition.values())
    print("Biggest partition is {}".format(biggest_partition))
    pd.DataFrame(list(zip(init_partition.keys(), init_partition.values()))).to_csv("./examples/ecoli_partition.csv", header=["comm id", "node list"], index=False)
    #vis("modular_part", init_partition, G_star)
    print("Modularity partition")
    score, tp = run_causal_discovery_partition(
                    dir_name,
                    "mod",
                    superstructure,
                    init_partition,
                    df,
                    G_star,
                    nthreads=16,
                    screen=True,
                    full_cand_set=False, finite_sample_limit=False
                )    
    causal_partition = expansive_causal_partition(superstructure, init_partition)
    #vis("causal_part", causal_partition, G_star)
    biggest_partition = max(len(p) for p in causal_partition.values())
    print("Biggest partition is {}".format(biggest_partition))
    print("Causal partition")
    score, tp = run_causal_discovery_partition(
                    dir_name,
                    "expansive_causal",
                    superstructure,
                    causal_partition,
                    df,
                    G_star,
                    nthreads=16,
                    screen=True,
                    full_cand_set=False,
                    
                )

    ec_partition = rand_edge_cover_partition(superstructure, init_partition)
    #vis("edge_cover_part", pef_partition, G_star)
    biggest_partition = max(len(p) for p in ec_partition.values())
    print("Biggest partition is {}".format(biggest_partition))
    print("Edge Coverage partition")
    score, tp = run_causal_discovery_partition(
                    dir_name,
                    "edge_cover",
                    superstructure,
                    ec_partition,
                    df,
                    G_star,
                    nthreads=16,
                    screen=False,
                    full_cand_set=False,
                )    
        
#     pef_partition = PEF_partition(df)
#     biggest_partition = max(len(p) for p in pef_partition.values())
#     print("Biggest partition is {}".format(biggest_partition))
#    # vis("pef_part", pef_partition, G_star)
#     print("PEF partition")
#     _, sp_c = run_causal_discovery(superstructure, pef_partition, df, G_star, run_serial=True, full_cand_set=True, screen=False)


if __name__ == "__main__":
    run()