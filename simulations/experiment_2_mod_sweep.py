# Experiment 2: two community, scale free, num samples 1e5, num_nodes=50,
# num_trials=30, artificial superstructure with 10% extraneous edges,
# fusion + screen projections
# Sweep rho parameter which controls the number of edges between the
# two communities
import numpy as np
from cd_v_partition.overlapping_partition import (
    PEF_partition,
    rand_edge_cover_partition,
    expansive_causal_partition,
    modularity_partition,
)
import pandas as pd
from cd_v_partition.utils import (
    get_data_from_graph,
    edge_to_adj,
    create_k_comms,
    artificial_superstructure,
    adj_to_edge,
)
import os
import time
from common_funcs import run_causal_discovery_serial, run_causal_discovery_partition, save
import functools
from concurrent.futures import ProcessPoolExecutor
def run_mod_alg(
    algorithm, experiment_dir, num_repeats, rho_range, nthreads=16, screen=False
):
    nsamples = 1e5
    scores = np.zeros((num_repeats, len(rho_range), 6))
    print("Algorithm is {}".format(algorithm))
    for i in range(num_repeats):
        for j, r in enumerate(
            rho_range
        ):  # Because the sweep parameter affects the graph structure we don't need to be careful about where the loop is
            init_partition, graph = create_k_comms(
                "scale_free", n=25, m_list=[1, 2], p_list=[0.5, 0.5], k=2, rho=r
            )
            num_nodes = len(graph.nodes())
            bias = np.random.normal(0, 1, size=num_nodes)
            var = np.abs(np.random.normal(0, 1, size=num_nodes))
            dir_name = (
                "./{}/{}/screen_projections/rho_{}/{}/".format(
                    experiment_dir, algorithm, r, i
                )
                if screen
                else "./{}/{}/fusion/rho_{}/{}/".format(
                    experiment_dir, algorithm, r, i
                )
            )
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            print("Rho {}".format(r))

            # Generate data
            (edges, nodes, _, _), df = get_data_from_graph(
                list(np.arange(num_nodes)),
                list(graph.edges()),
                nsamples=int(nsamples),
                iv_samples=0,
                bias=bias,
                var=var,
            )
            G_star = edge_to_adj(edges, nodes)

            # Find superstructure
            frac_extraneous = 0.1
            superstructure = artificial_superstructure(
                G_star, frac_extraneous=frac_extraneous
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
                partition = modularity_partition(
                    superstructure, cutoff=1, best_n=None
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
                    screen=screen,
                    full_cand_set=full_cand_set,
                )
                scores[i][j][0:5] = score
                scores[i][j][-1] = tp + tm
                np.savetxt("{}/time_chkpoint.txt".format(dir_name), scores[i][j])
                
    save("{}/{}".format(experiment_dir, algorithm), [scores], [algorithm], num_repeats, rho_range, x_axis_name="Rho", screen=screen)



if __name__ == "__main__":
    # Simple case for debugging
    algorithms = ["serial", "pef", "edge_cover", "expansive_causal", "mod"]
    # func_partial = functools.partial(run_mod_alg, experiment_dir="./simulations/experiment_2_test/", nthreads=16, num_repeats=2, rho_range=np.arange(0,0.2,0.1), screen=True )
    # results = []
    # with ProcessPoolExecutor(max_workers=len(algorithms)) as executor:
    #     for result in executor.map(func_partial, algorithms, chunksize=1):
    #         results.append(result)
    
    #screen projections
    func_partial = functools.partial(
        run_mod_alg,
        experiment_dir="./simulations/experiment_2/",
        nthreads=16,
        num_repeats=10,
        rho_range=np.arange(0,0.5,0.1),
        screen=True,
    )
    results = []
    with ProcessPoolExecutor(max_workers=len(algorithms)) as executor:
        for result in executor.map(func_partial, algorithms, chunksize=1):
            results.append(result)
            
    
    # fusion    
    func_partial = functools.partial(
        run_mod_alg,
        experiment_dir="./simulations/experiment_2/",
        nthreads=16,
        num_repeats=10,
        rho_range=[0, 0.01, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5],
        screen=False,
    )
    results = []
    with ProcessPoolExecutor(max_workers=len(algorithms)) as executor:
        for result in executor.map(func_partial, algorithms, chunksize=1):
            results.append(result)
