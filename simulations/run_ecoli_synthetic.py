import networkx as nx
import numpy as np
from cd_v_partition.vis_partition import create_partition_plot
from cd_v_partition.overlapping_partition import (
    partition_problem,
    hierarchical_partition,
)
import seaborn as sns
import pandas as pd
from cd_v_partition.utils import (
    adj_to_dag,
    get_data_from_graph,
    adj_to_edge,
    artificial_superstructure,
    get_scores,
)
from cd_v_partition.overlapping_partition import (
    rand_edge_cover_partition,
    expansive_causal_partition,
    modularity_partition,
    PEF_partition,
)
from cd_v_partition.causal_discovery import sp_gies, pc
from cd_v_partition.fusion import screen_projections, fusion, remove_edges_not_in_ss
import time
import functools
from concurrent.futures import ProcessPoolExecutor


def _local_structure_learn(subproblem):
    skel, data = subproblem
    # adj_mat = sp_gies(data, skel=skel, outdir=None)
    adj_mat, _ = pc(data, outdir=None)
    return adj_mat


# ss_subset dictates whether we discard edges not in the superstructure
# as part of post-processing, in both the serial and fusion methods.
def run_causal_discovery_partition(
    dir_name,
    save_name,
    superstructure,
    partition,
    df,
    G_star,
    nthreads=16,
    full_cand_set=False,
    screen=False,
    ss_subset=True,
    finite_sample_limit=True,
):
    # pd.DataFrame(list(zip(partition.keys(), partition.values()))).to_csv(
    #     "{}/{}_partition.csv".format(dir_name, save_name),
    #     header=["comm id", "node list"],
    #     index=False,
    # )

    start = time.time()
    # Break up problem according to provided partition
    subproblems = partition_problem(partition, superstructure, df)

    # Local learn
    func_partial = functools.partial(_local_structure_learn)
    results = []
    num_partitions = len(partition)
    chunksize = max(1, num_partitions // nthreads)
    # print("Launching processes")

    # with ProcessPoolExecutor(max_workers=nthreads) as executor:
    #    for result in executor.map(func_partial, subproblems, chunksize=chunksize):
    #        results.append(result)

    for s in subproblems:
        sp = time.time()
        r = _local_structure_learn(s)
        results.append(r)
        tp = time.time() - sp
        print("Time for local run with {} nodes is {} (s)".format(s[0].shape[0], tp))

    # Merge globally
    data_obs = df.drop(columns=["target"]).to_numpy()
    if screen:
        est_graph_partition = screen_projections(
            superstructure,
            partition,
            results,
            ss_subset=ss_subset,
            finite_lim=finite_sample_limit,
            data=data_obs,
        )
    else:
        est_graph_partition = fusion(
            superstructure, partition, results, data_obs, full_cand_set=full_cand_set
        )
    time_partition = time.time() - start

    # Save the edge list
    # learned_edges = list(est_graph_partition.edges(data=False))
    # pd.DataFrame(data=learned_edges, columns=["node1", "node2"]).to_csv(
    #     "{}/edges_{}.csv".format(dir_name, save_name), index=False
    # )
    scores_part = get_scores(["CD-partition"], [est_graph_partition], G_star)

    return scores_part, time_partition


def run_causal_discovery_serial(
    dir_name,
    superstructure,
    df,
    G_star,
    ss_subset=True,
):
    # Call serial method

    start = time.time()
    est_graph_serial = _local_structure_learn([superstructure, df])
    # optional post-processing: discard edges not in superstructure
    if ss_subset:
        ss_graph = nx.from_numpy_array(superstructure, create_using=nx.DiGraph)
        est_graph_serial_DiGraph = nx.from_numpy_array(
            est_graph_serial, create_using=nx.DiGraph
        )
        subselected_serial_DiGraph = remove_edges_not_in_ss(
            est_graph_serial_DiGraph, ss_graph
        )
        # convert back to numpy array
        est_graph_serial = nx.to_numpy_array(
            subselected_serial_DiGraph,
            nodelist=np.arange(len(subselected_serial_DiGraph.nodes())),
        )
    time_serial = time.time() - start
    scores_serial = get_scores(["CD-serial"], [est_graph_serial], G_star)
    # learned_edges = adj_to_edge(
    #     est_graph_serial,
    #     nodes=list(np.arange(est_graph_serial.shape[0])),
    #     ignore_weights=True,
    # )
    # pd.DataFrame(data=learned_edges, columns=["node1", "node2"]).to_csv(
    #     "{}/edges_serial.csv".format(dir_name), index=False
    # )
    return scores_serial, time_serial


def run():
    G_star = np.loadtxt("./datasets/bionetworks/ecoli/synthetic_copies/net_1.txt")
    # G_star = G_star[0:1000][:,0:1000]
    print(G_star.shape)
    nodes = np.arange(G_star.shape[0])
    edges = adj_to_edge(G_star, nodes, ignore_weights=True)
    df = get_data_from_graph(
        nodes, edges, nsamples=int(1e3), iv_samples=0, bias=None, var=None
    )[-1]
    superstructure = artificial_superstructure(G_star, frac_extraneous=0.1)
    dir_name = "./simulations/"
    print("Start Mod partitioning...")
    start = time.time()
    init_partition = modularity_partition(superstructure, cutoff=1, best_n=None)
    print("Time for partitioning {}".format(time.time() - start))

    biggest_partition = max(len(p) for p in init_partition.values())
    print("Biggest partition is {}".format(biggest_partition))
    pd.DataFrame(list(zip(init_partition.keys(), init_partition.values()))).to_csv(
        "./examples/ecoli_partition.csv", header=["comm id", "node list"], index=False
    )
    # vis("modular_part", init_partition, G_star)
    print("Modularity partition")
    score, tp = run_causal_discovery_partition(
        dir_name,
        "mod",
        superstructure,
        init_partition,
        df,
        G_star,
        nthreads=1,
        screen=True,
        full_cand_set=False,
        finite_sample_limit=False,
    )
    causal_partition = expansive_causal_partition(superstructure, init_partition)
    # vis("causal_part", causal_partition, G_star)
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
        nthreads=1,
        screen=True,
        full_cand_set=False,
    )

    ec_partition = rand_edge_cover_partition(superstructure, init_partition)
    # vis("edge_cover_part", pef_partition, G_star)
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
        nthreads=1,
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
