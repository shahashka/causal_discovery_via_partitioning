# Experiment 5: hierarchical networks, num samples 1e5,
# artificial superstructure with 10% extraneous edges,
# num_trials=30, default modularity (rho=0.01), fusion + screen projections
# Sweep the number of nodes 50 5e4

import networkx as nx
import numpy as np
from cd_v_partition.vis_partition import create_partition_plot
from cd_v_partition.overlapping_partition import (
    partition_problem,
    PEF_partition,
    rand_edge_cover_partition,
    expansive_causal_partition,
    modularity_partition,
)
import seaborn as sns
import pandas as pd
from cd_v_partition.utils import (
    get_data_from_graph,
    edge_to_adj,
    artificial_superstructure,
    get_scores,
    adj_to_edge,
    get_random_graph_data,
    directed_heirarchical_graph,
)
from cd_v_partition.causal_discovery import sp_gies, pc
from cd_v_partition.fusion import screen_projections, fusion
import functools
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import os
import time
from common_funcs import run_causal_discovery_serial, run_causal_discovery_partition, save


def run_nnodes_alg(
    algorithm, experiment_dir, num_repeats, nnodes_range, nthreads=16, screen=False
):
    nsamples = 1e3
    scores = np.zeros((num_repeats, len(nnodes_range), 6))
    print("Algorithm is {}".format(algorithm))
    for i in range(num_repeats):
        for j, nnodes in enumerate(nnodes_range):
            print("Number of nodes {}".format(nnodes))

            # Generate data
            G_dir = directed_heirarchical_graph(nnodes)
            (edges, nodes, _, _), df = get_data_from_graph(
                list(np.arange(len(G_dir.nodes()))),
                list(G_dir.edges()),
                nsamples=int(nsamples),
                iv_samples=0,
                bias=None,
                var=None,
            )
            # (edges, nodes, _, _), df = get_random_graph_data("hierarchical", num_nodes=nnodes, nsamples=int(nsamples), iv_samples=0, p=0.5, m=2)

            dir_name = (
                "./{}/{}/screen_projections/nnodes_{}/{}/".format(
                    experiment_dir, algorithm, nnodes, i
                )
                if screen
                else "./{}/{}/fusion/nnodes_{}/{}/".format(
                    experiment_dir, algorithm, nnodes, i
                )
            )
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            G_star = edge_to_adj(edges, nodes)

            # Find superstructure
            superstructure = artificial_superstructure(G_star, frac_extraneous=0.1)
            superstructure_edges = adj_to_edge(
                superstructure, nodes, ignore_weights=True
            )
            if algorithm == "serial":
                ss, ts = run_causal_discovery_serial(
                    dir_name,
                    superstructure,
                    df,
                    G_star,
                )
                scores[i][j][0:5] = ss
                scores[i][j][-1] = ts
                np.savetxt("{}/time_chkpoint.txt".format(dir_name), scores[i][j])

            else:
                start = time.time()
                nc = int(nnodes/10)
                partition = modularity_partition(
                    superstructure, cutoff=nc, best_n=nc
                )
                tm = time.time() - start

                if algorithm=='expansive_causal':
                    start = time.time()
                    partition = expansive_causal_partition(superstructure, partition)
                    tm += time.time() - start
                elif algorithm=='edge_cover':
                    start = time.time()
                    partition = rand_edge_cover_partition(superstructure, partition)
                    tm += time.time() - start
                else:
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
                    screen=False,
                    full_cand_set=full_cand_set,
                )
                scores[i][j][0:5] = score
                scores[i][j][-1] = tp + tm
                np.savetxt("{}/time_chkpoint.txt".format(dir_name), scores[i][j])
    save("{}/{}".format(experiment_dir, algorithm), [scores], [algorithm], num_repeats, nnodes_range, x_axis_name="Number of nodes", screen=screen)


if __name__ == "__main__":
    # Simple case for debugging
    algorithms = ["serial", "pef", "edge_cover", "expansive_causal", "mod"]
    #func_partial = functools.partial(run_nnodes_alg, experiment_dir="./simulations/experiment_5_test/", nthreads=16, num_repeats=2, nnodes_range=[10**i for i in np.arange(1,3)], screen=True )
    # screen projections 
    func_partial = functools.partial(
        run_nnodes_alg,
        experiment_dir="./simulations/experiment_5/",
        nthreads=16,
        num_repeats=5,
        nnodes_range=[10**i for i in np.arange(1, 5)],
        screen=True,
    )
    results = []
    with ProcessPoolExecutor(max_workers=len(algorithms)) as executor:
        for result in executor.map(func_partial, algorithms, chunksize=1):
            results.append(result)
    
    # #fusion        
    # func_partial = functools.partial(
    #     run_nnodes_alg,
    #     experiment_dir="./simulations/experiment_5/",
    #     nthreads=16,
    #     num_repeats=5,
    #     nnodes_range=[10**i for i in np.arange(1, 5)],
    #     screen=False,
    # )
    # results = []
    # with ProcessPoolExecutor(max_workers=len(algorithms)) as executor:
    #     for result in executor.map(func_partial, algorithms, chunksize=1):
    #         results.append(result)
