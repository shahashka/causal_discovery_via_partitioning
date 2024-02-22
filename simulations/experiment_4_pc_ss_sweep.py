# Experiment 4: two community, scale free, num samples 1e5, num_nodes=50,
# num_trials=30, default modularity (rho=0.01), fusion + screen projections
# Sweep PC alpha parameter which controls
# the density of the supsertructure
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
    adj_to_edge,
)
from cd_v_partition.causal_discovery import  pc
import os
import time
from common_funcs import run_causal_discovery_serial, run_causal_discovery_partition, save


def run_ss_pc(experiment_dir, num_repeats, alpha_range, nthreads=16, screen=False):
    nsamples = 1e5
    scores_serial = np.zeros((num_repeats, len(alpha_range), 6))
    scores_edge_cover = np.zeros((num_repeats, len(alpha_range), 6))
    scores_causal_partition = np.zeros((num_repeats, len(alpha_range), 6))
    scores_mod_partition = np.zeros((num_repeats, len(alpha_range), 6))
    scores_pef = np.zeros((num_repeats, len(alpha_range), 6))

    for i in range(num_repeats):
        init_partition, graph = create_k_comms(
            "scale_free", n=25, m_list=[1, 2], p_list=[0.5, 0.5], k=2
        )

        num_nodes = len(graph.nodes())
        bias = np.random.normal(0, 1, size=num_nodes)
        var = np.abs(np.random.normal(0, 1, size=num_nodes))

        # Generate data
        (edges, nodes, _, _), df = get_data_from_graph(
            list(np.arange(num_nodes)),
            list(graph.edges()),
            nsamples=int(nsamples),
            iv_samples=0,
            bias=bias,
            var=var,
        )
        for j, alpha in enumerate(alpha_range):
            dir_name = (
                "./{}/screen_projections/alpha_{}/{}/".format(experiment_dir, alpha, i)
                if screen
                else "./{}/fusion/alpha_{}/{}/".format(experiment_dir, alpha, i)
            )
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            print("Alpha {}".format(alpha))

            G_star = edge_to_adj(edges, nodes)

            # Find superstructure
            superstructure, _ = pc(df, alpha=alpha, num_cores=nthreads, outdir=None)
            # Run serial
            ss, ts = run_causal_discovery_serial(
                dir_name,
                superstructure,
                df,
                G_star,ss_subset=False
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
                nthreads=nthreads,ss_subset=False
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
                nthreads=nthreads,ss_subset=False

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
                nthreads=nthreads,ss_subset=False

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
    save(experiment_dir, scores, labels, num_repeats, alpha_range, "Alpha", screen)



if __name__ == "__main__":
    # Simple case for debugging
    # run_ss_pc("./simulations/experiment_4_test/", nthreads=16, num_repeats=1, alpha_range=np.arange(0.1,0.2,0.1), screen=False)
    # run_ss_pc("./simulations/experiment_4_test/", nthreads=16, num_repeats=1, alpha_range=np.arange(0.1,0.2,0.1), screen=True)

    run_ss_pc(
        "./simulations/experiment_4/",
        nthreads=16,
        num_repeats=50,
        alpha_range=np.arange(0.1, 1, 0.1),
        screen=True,
    )
    run_ss_pc(
        "./simulations/experiment_4/",
        nthreads=16,
        num_repeats=50,
        alpha_range=np.arange(0.1, 1, 0.1),
        screen=False,
    )
