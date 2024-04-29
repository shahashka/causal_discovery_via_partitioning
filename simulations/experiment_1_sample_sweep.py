# Experiment 1: two community, scale free, default rho modularity (0.01),,
# num_nodes=50, num_trials=30, artificial superstructure with 10% extraneous edges,
# fusion + screen projections
# Sweep number of samples

import numpy as np
from cd_v_partition.config import SimulationConfig
from cd_v_partition.experiment import Experiment
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
import functools
import networkx as nx
from concurrent.futures import ProcessPoolExecutor
from common_funcs import run_causal_discovery_serial, run_causal_discovery_partition, save
import os
import time

from tqdm import tqdm
import pdb


def run_samples_alg(
    algorithm, experiment_dir, num_repeats, sample_range, nthreads=16, screen=False
):
    scores = np.zeros((num_repeats, len(sample_range), 6))
    print("Algorithm is {}".format(algorithm))
    for i in range(num_repeats):
        init_partition, graph = create_k_comms(
            "scale_free", n=25, m_list=[1, 2], p_list=[0.5, 0.5], k=2
        )
        # graph = nx.DiGraph()
        # graph.add_edges_from((0,1), (1,3),(2,4), (1,4))
        num_nodes = len(graph.nodes())
        bias = np.random.normal(0, 1, size=num_nodes)
        var = np.abs(np.random.normal(0, 1, size=num_nodes))
        for j, ns in enumerate(sample_range):
            if algorithm == 'pef':
                screen=False
            dir_name = (
                "./{}/{}/screen_projections/samples_{}/{}/".format(
                    experiment_dir, algorithm, ns, i
                )
                if screen
                else "./{}/{}/fusion/samples_{}/{}/".format(
                    experiment_dir, algorithm, ns, i
                )
            )
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            print("Number of samples {}".format(ns))

            # Generate data
            (edges, nodes, _, _), df = get_data_from_graph(
                list(np.arange(num_nodes)),
                list(graph.edges()),
                nsamples=int(ns),
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
                elif algorithm=='pef':
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
                
    save("{}/{}".format(experiment_dir, algorithm), [scores], [algorithm], num_repeats, sample_range, x_axis_name="Number of samples", screen=screen)


if __name__ == "__main__":
    exp_1 = Experiment(32)
    sim_cfg = SimulationConfig(graph_per_spec=2,
                               experiment_id="simulations/experiment_1_refactor",
                               partition_fn=['no_partition', 'modularity', 'edge_cover', 'expansive_causal'],
                               num_samples=[10**i for i in np.arange(1, 3)],
                               graph_kind="scale_free",
                               num_nodes=25,
                               num_communities=2,                              
                               causal_learn_fn=["GES"], 
                               merge_fn=["screen"],
                               )
    
    sim_cfg_pef = SimulationConfig(graph_per_spec=2,
                               experiment_id="experiment_1/pef",
                               partition_fn=['PEF'],
                               num_samples=[10**i for i in np.arange(1, 3)],
                               graph_kind="scale_free",
                               num_nodes=25,
                               num_communities=2,                              
                               causal_learn_fn=["GES"], 
                               merge_fn=["fusion"],
                               merge_full_cand_set=[True]
                               )
    exp_1.run(sim_cfg, random_state=1)
    exp_1.run(sim_cfg_pef, random_state=1)
    # sim_spec = SimulationSpec( graph_kind="scale_free",
    #                            num_nodes=25,
    #                            num_communities=2,
    #                            inter_edge_prob=0.01,
    #                            edge_prob_alpha=0.5,
    #                            comm_pop_alpha=1, 
    #                            comm_pop_coeff=1, # TODO can i get m1=1 and m2=2 with this? 
    #                            partition_fn="modularity",
    #                            partition_best_n=None,
    #                            partition_cutoff=1,
    #                            partition_resolution=1,
    #                            merge_fn="screen", # PEF needs fusion
    #                            merge_full_cand_set=False, # PEF needs True
    #                            merge_ss_subset_flag=True,
    #                            merge_finite_sample_flag=True,
    #                            causal_learn_fn="GES",
    #                            use_pc_algorithm=False,
    #                            alpha=None,
    #                            frac_extraneous=0.1,
    #                            frac_retain_direction=0.1
    #                            )
    # Simple case for debugging
    # algorithms = ["serial", "pef", "edge_cover", "expansive_causal", "mod"]
    # # func_partial = functools.partial(
    # #     run_samples_alg, 
    # #     experiment_dir="./simulations/experiment_1_test_dagma/", 
    # #     nthreads=16, 
    # #     num_repeats=1, 
    # #     sample_range=[10**i for i in np.arange(3,4)], 
    # #     screen=True )
    # # screen projections
    # func_partial = functools.partial(
    #     run_samples_alg,
    #     experiment_dir="./simulations/experiment_1_dagma/",
    #     nthreads=64,
    #     num_repeats=30,
    #     sample_range=[10**i for i in np.arange(1, 7)],
    #     screen=True,
    # )
    # results = []
    # with ProcessPoolExecutor(max_workers=len(algorithms)) as executor:
    #     for result in executor.map(func_partial, algorithms, chunksize=1):
    #         results.append(result)
        
