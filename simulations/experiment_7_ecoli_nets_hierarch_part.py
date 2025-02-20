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
    hierarchical_partition,
)

import functools
from concurrent.futures import ProcessPoolExecutor
import os
import time
from common_funcs import run_causal_discovery_partition, run_causal_discovery_serial


def run_ecoli_alg(
    algorithm, experiment_dir, net_id, num_samples, nthreads=16, screen=False
):
    data_dir = "./datasets/bionetworks/ecoli/synthetic_copies"
    frac_extraneoues = 0.1
    G_star = np.loadtxt("{}/net_{}.txt".format(data_dir, net_id))
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
    if algorithm == "pef":
        screen = False
    dir_name = (
        "./{}/{}/screen_projections/net_{}/".format(experiment_dir, algorithm, net_id)
        if screen
        else "./{}/{}/fusion/net_{}/".format(experiment_dir, algorithm, net_id)
    )
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Find superstructure
    superstructure = artificial_superstructure(G_star, frac_extraneous=frac_extraneoues)

    scores = np.zeros(6)
    print("Algorithm is {}".format(algorithm))
    if algorithm == "serial":
        ss, ts = run_causal_discovery_serial(
            dir_name,
            superstructure,
            df,
            G_star,
        )
        scores[0:5] = ss
        scores[-1] = ts
        np.savetxt("{}/time_chkpoint.txt".format(dir_name), scores)

    else:
        start = time.time()
        partition = hierarchical_partition(superstructure, max_community_size=0.1)
        tm = time.time() - start

        if algorithm == "expansive_causal":
            start = time.time()
            partition = expansive_causal_partition(superstructure, partition)
            tm += time.time() - start
        elif algorithm == "edge_cover":
            start = time.time()
            partition = rand_edge_cover_partition(superstructure, partition)
            tm += time.time() - start
        elif algorithm == "pef":
            start = time.time()
            partition = PEF_partition(df)
            tm = time.time() - start

        biggest_partition = max(len(p) for p in partition.values())
        print("Biggest partition is {}".format(biggest_partition))

        full_cand_set = algorithm == "pef"
        score, tp = run_causal_discovery_partition(
            dir_name,
            algorithm,
            superstructure,
            partition,
            df,
            G_star,
            nthreads=nthreads,
            screen=screen,
            full_cand_set=full_cand_set,
        )
        scores[0:5] = score
        scores[-1] = tp + tm
        np.savetxt("{}/time_chkpoint.txt".format(dir_name), scores)


if __name__ == "__main__":
    algorithms = ["serial", "pef", "edge_cover", "expansive_causal", "hierarchical"]
    func_partial = functools.partial(
        run_ecoli_alg,
        experiment_dir="./simulations/experiment_7/",
        nthreads=64,
        net_id=0,
        num_samples=1e4,
        screen=True,
    )
    results = []
    with ProcessPoolExecutor(max_workers=len(algorithms)) as executor:
        for result in executor.map(func_partial, algorithms, chunksize=1):
            results.append(result)
