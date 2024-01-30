# Experiment 1: two community, scale free, default rho modularity (0.01),,
# num_nodes=50, num_trials=30, artificial superstructure with 10% extraneous edges,
# fusion + screen projections
# Sweep number of samples

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

from common_funcs import run_causal_discovery_serial, run_causal_discovery_partition, save

import os
import time

from tqdm import tqdm
import pdb


def run_samples(experiment_dir, num_repeats, sample_range, nthreads=16, screen=False):
    scores_serial = np.zeros((num_repeats, len(sample_range), 6))
    scores_edge_cover = np.zeros((num_repeats, len(sample_range), 6))
    scores_causal_partition = np.zeros((num_repeats, len(sample_range), 6))
    scores_mod_partition = np.zeros((num_repeats, len(sample_range), 6))
    scores_pef = np.zeros((num_repeats, len(sample_range), 6))

    for i in range(num_repeats):
        init_partition, graph = create_k_comms(
            "scale_free", n=25, m_list=[1, 2], p_list=[0.5, 0.5], k=2
        )
        num_nodes = len(graph.nodes())
        bias = np.random.normal(0, 1, size=num_nodes)
        var = np.abs(np.random.normal(0, 1, size=num_nodes))
        for j, ns in enumerate(sample_range):
            dir_name = (
                "./{}/screen_projections/samples_{}/{}/".format(experiment_dir, ns, i)
                if screen
                else "./{}/fusion/samples_{}/{}/".format(experiment_dir, ns, i)
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
            # Save true graph and data
            df.to_csv("{}/data.csv".format(dir_name), header=True, index=False)
            pd.DataFrame(data=np.array(edges), columns=["node1", "node2"]).to_csv(
                "{}/edges_true.csv".format(dir_name), index=False
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
            scores_serial[i][j][0:5] = ss
            scores_serial[i][j][-1] = ts

            # Run each partition and get scores
            start = time.time()
            mod_partition = modularity_partition(superstructure, cutoff=1, best_n=None)
            tm = time.time() - start

            sp,tp = run_causal_discovery_partition(
                dir_name,
                "mod",
                superstructure,
                mod_partition,
                df,
                G_star,
                screen=screen,
                nthreads=nthreads,
            )
            scores_mod_partition[i][j][0:5] = sp
            scores_mod_partition[i][j][-1] = tp + tm

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
                nthreads=nthreads,
            )

            scores_edge_cover[i][j][0:5] = sp
            scores_edge_cover[i][j][-1] = tp + tec + tm

            start = time.time()
            partition = expansive_causal_partition(superstructure, mod_partition)
            tca = time.time() - start

            sp, tp = run_causal_discovery_partition(
                dir_name, "causal", superstructure, partition, df, G_star, screen=screen
            )
            scores_causal_partition[i][j][0:5] = sp
            scores_causal_partition[i][j][-1] = tp + tca + tm

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
                nthreads=nthreads,

            )
            scores_pef[i][j][0:5] = sp
            scores_pef[i][j][-1] = tp + tpef


    scores = [
        scores_serial,
        scores_pef,
        scores_edge_cover,
        scores_causal_partition,
        scores_mod_partition,
    ]
    labels = ["serial", "pef", "edge_cover", "expansive_causal", "mod"]
    save(experiment_dir, scores, labels, num_repeats, sample_range, "Number of samples", screen)



if __name__ == "__main__":
    # Simple version for debugging
    # run_samples("./simulations/experiment_1/", nthreads=16, num_repeats=1, sample_range=[10**i for i in range(1,3)], screen=False)
    # run_samples(
    #     "./simulations/experiment_1_test/",
    #     nthreads=16,
    #     num_repeats=2,
    #     sample_range=[10**i for i in range(1, 3)],
    #     screen=True,
    # )

    run_samples(
        "./simulations/experiment_1/",
        nthreads=16,
        num_repeats=30,
        sample_range=[10**i for i in range(1, 6)],
        screen=True,
    )
    run_samples(
        "./simulations/experiment_1/",
        nthreads=16,
        num_repeats=30,
        sample_range=[10**i for i in range(1, 6)],
        screen=False,
    )
