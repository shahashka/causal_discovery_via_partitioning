# Experiment 6: Synthetic e.coli networks, num samples 1e5, artificial ss with frac_extraneoues = 0.5
# Modularity partitioning
import numpy as np
import pandas as pd
from cd_v_partition.utils import (
    get_data_from_graph,
    adj_to_edge,
    artificial_superstructure,
)
from cd_v_partition.overlapping_partition import (
    PEF_partition,
    rand_edge_cover_partition,
    expansive_causal_partition,
    modularity_partition,
)


from concurrent.futures import ProcessPoolExecutor
import os
import time
from common_funcs import run_causal_discovery_partition, run_causal_discovery_serial


def run_ecoli(i):
    data_dir = "./datasets/bionetworks/ecoli/synthetic_copies"
    experiment_dir = "./simulations/experiment_6/"
    screen = True
    nthreads = 16
    num_samples = 1e5
    frac_extraneoues = 0.1
    scores_by_net = pd.DataFrame(columns=["Algorithm", "SHD", "TPR", "FPR", "Time (s)"])
    G_star = np.loadtxt("{}/net_{}.txt".format(data_dir, i))
    # G_star = G_star[0:100][:,0:100] # for debugging
    nodes = np.arange(G_star.shape[0])

    G_star_edges = adj_to_edge(G_star, list(nodes), ignore_weights=True)
    df = get_data_from_graph(
        nodes,
        G_star_edges,
        nsamples=int(num_samples),
        iv_samples=0,
        bias=None,
        var=None,
    )[-1]

    dir_name = (
        "./{}/screen_projections/net_{}/".format(experiment_dir, i)
        if screen
        else "./{}/fusion/net_{}/".format(experiment_dir, i)
    )
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Save true graph and data
    df.to_csv("{}/data.csv".format(dir_name), header=True, index=False)
    pd.DataFrame(data=np.array(G_star_edges), columns=["node1", "node2"]).to_csv(
        "{}/edges_true.csv".format(dir_name), index=False
    )

    # Find superstructure
    superstructure = artificial_superstructure(G_star, frac_extraneous=frac_extraneoues)
    superstructure_edges = adj_to_edge(superstructure, nodes, ignore_weights=True)
    pd.DataFrame(
        data=np.array(superstructure_edges), columns=["node1", "node2"]
    ).to_csv("{}/edges_ss.csv".format(dir_name), index=False)

    # Run serial
    ss, ts = run_causal_discovery_serial(
        dir_name,
        superstructure,
        df,
        G_star,
    )
    scores_by_net.loc[len(scores_by_net.index)] = ["Serial", ss[0], ss[-2], ss[-1], ts]
            
    # Run each partition and get scores
    start = time.time()
    mod_partition = modularity_partition(superstructure, resolution=5, cutoff=20, best_n=20)
    tm = time.time() - start

    sp,tp = run_causal_discovery_partition(
        dir_name,
        "mod",
        superstructure,
        mod_partition,
        df,
        G_star,
        screen=screen,
        nthreads=nthreads
    )
    scores_by_net.loc[len(scores_by_net.index)] = [
        "Modularity Partition",
        sp[0],
        sp[-2],
        sp[-1],
        tp + tm,
    ]

    start = time.time()
    partition = rand_edge_cover_partition(superstructure, mod_partition)
    tec = time.time() - start

    sp, tp = run_causal_discovery_partition(
        dir_name,
        "edge_cover",
        superstructure,
        partition,
        df,
        G_star,
        screen=screen,
        nthreads=nthreads

    )

    scores_by_net.loc[len(scores_by_net.index)] = [
        "Random Edge Cover Partition",
        sp[0],
        sp[-2],
        sp[-1],
        tp + tec + tm,
    ]

    start = time.time()
    partition = expansive_causal_partition(superstructure, mod_partition)
    tca = time.time() - start

    sp, tp = run_causal_discovery_partition(
        dir_name,
        "causal",
        superstructure, 
        partition, 
        df, 
        G_star, 
        screen=screen,
        nthreads=nthreads
    )
    scores_by_net.loc[len(scores_by_net.index)] = [
        "Causal Partition",
        sp[0],
        sp[-2],
        sp[-1],
        tp + tca + tm,
    ]

    start = time.time()
    partition = PEF_partition(df)
    tpef = time.time() - start

    sp, tp = run_causal_discovery_partition(
        dir_name,
        "pef",
        superstructure,
        partition,
        df,
        G_star,
        screen=screen,
        full_cand_set=True,
        nthreads=nthreads

    )
    scores_by_net.loc[len(scores_by_net.index)] = [
        "PEF",
        sp[0],
        sp[-2],
        sp[-1],
        tp + tpef,
    ]

    scores_by_net.to_csv("{}/scores_{}.csv".format(dir_name, i))


if __name__ == "__main__":
    # run_ecoli("./simulations/experiment_6/", nthreads=16,  screen=False)
    nthreads = 16
    num_datasets = 10
    chunksize = max(1, num_datasets // nthreads)
    run_index = np.arange(num_datasets)
    results = []
    with ProcessPoolExecutor(max_workers=nthreads) as executor:
        for result in executor.map(run_ecoli, run_index, chunksize=chunksize):
            results.append(result)
    # run_ecoli("./simulations/experiment_6/", nthreads=16,  screen=True)
