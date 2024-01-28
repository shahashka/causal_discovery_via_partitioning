# Experiment 6: Synthetic e.coli networks, num samples 1e5, artificial ss with frac_extraneoues = 0.5
# Modularity partitioning

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
import networkx as nx
import numpy as np
from cd_v_partition.overlapping_partition import partition_problem, PEF_partition, rand_edge_cover_partition, expansive_causal_partition, modularity_partition
import seaborn as sns
import pandas as pd
from cd_v_partition.utils import (
    get_data_from_graph,
    edge_to_adj,
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
    
def run_causal_discovery(dir_name, save_name, superstructure, partition, df, G_star, nthreads=16, run_serial=False, full_cand_set=False, screen=False):
    pd.DataFrame(list(zip(partition.keys(), partition.values()))).to_csv("{}/{}_partition.csv".format(dir_name, save_name), header=["comm id", "node list"], index=False)

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


def run_ecoli(i): 
    data_dir="./datasets/bionetworks/ecoli/synthetic_copies"
    experiment_dir = "./simulations/experiment_6/"
    screen = True
    nthreads=16
    num_samples = 1e5
    frac_extraneoues=0.1
    scores_by_net = pd.DataFrame(columns=["Algorithm", "SHD", "TPR","FPR", "Time (s)"])
    G_star = np.loadtxt("{}/net_{}.txt".format(data_dir, i))
    #G_star = G_star[0:100][:,0:100] for debugging
    nodes = np.arange(G_star.shape[0])

    G_star_edges = adj_to_edge(G_star, list(nodes), ignore_weights=True)
    df = get_data_from_graph(nodes, G_star_edges, nsamples=int(num_samples), iv_samples=0, bias=None, var=None)[-1]

    dir_name = "./{}/screen_projections/net_{}/".format(experiment_dir, i) if screen else "./{}/fusion/net_{}/".format(experiment_dir, i)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    # Save true graph and data 
    df.to_csv("{}/data.csv".format(dir_name), header=True, index=False)
    pd.DataFrame(data=np.array(G_star_edges), columns=['node1', 'node2']).to_csv("{}/edges_true.csv".format(dir_name), index=False)

    # Find superstructure 
    superstructure = artificial_superstructure(G_star, frac_extraneous=frac_extraneoues)
    superstructure_edges = adj_to_edge(superstructure, nodes, ignore_weights=True)
    pd.DataFrame(data=np.array(superstructure_edges), columns=['node1', 'node2']).to_csv("{}/edges_ss.csv".format(dir_name), index=False)

    
    # Run each partition and get scores 
    start = time.time()
    mod_partition = modularity_partition(superstructure, cutoff=1, best_n=None) 
    tm = time.time() - start
    
    ss, sp, ts, tp = run_causal_discovery(dir_name, "mod",superstructure, mod_partition, df, G_star, nthreads=nthreads, screen=screen, run_serial=True)
    scores_by_net.loc[len(scores_by_net.index)] = ["Serial", ss[0], ss[-2], ss[-1], ts]
    scores_by_net.loc[len(scores_by_net.index)] = ["Modularity Partition", sp[0], sp[-2], sp[-1], tp + tm]

    start = time.time()
    partition = rand_edge_cover_partition(superstructure, mod_partition)
    tec = time.time() - start

    _, sp, _ , tp = run_causal_discovery(dir_name, "edge_cover",superstructure, partition, df, G_star, nthreads=nthreads,screen=screen)
    scores_by_net.loc[len(scores_by_net.index)] = ["Random Edge Cover Partition", sp[0], sp[-2], sp[-1], tp + tec+ tm]

    start = time.time()
    partition = expansive_causal_partition(superstructure, mod_partition)
    tca = time.time() - start
    
    _, sp, _, tp = run_causal_discovery(dir_name, "causal",superstructure, partition, df, G_star, nthreads=nthreads,screen=screen)
    scores_by_net.loc[len(scores_by_net.index)] = ["Causal Partition", sp[0], sp[-2], sp[-1], tp + tca+ tm]

    start = time.time()
    partition = PEF_partition(df)
    tpef = time.time()-start
    
    _, sp, _, tp = run_causal_discovery(dir_name, "pef",superstructure, partition, df, G_star, nthreads=nthreads,screen=screen, full_cand_set=True)
    scores_by_net.loc[len(scores_by_net.index)] = ["PEF", sp[0], sp[-2], sp[-1], tp+tpef]
    
    scores_by_net.to_csv("{}/scores_{}.csv".format(dir_name,i))



if __name__ == "__main__":
    #run_ecoli("./simulations/experiment_6/", nthreads=16,  screen=False)
    nthreads = 16
    num_datasets = 10
    chunksize = max(1, num_datasets // nthreads)
    run_index = np.arange(num_datasets)
    results = []
    with ProcessPoolExecutor(max_workers=nthreads) as executor:
        for result in executor.map(run_ecoli, run_index, chunksize=chunksize):
            results.append(result)
    #run_ecoli("./simulations/experiment_6/", nthreads=16,  screen=True)
