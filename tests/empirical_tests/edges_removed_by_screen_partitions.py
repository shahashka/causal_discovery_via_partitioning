import networkx as nx
import numpy as np
from cd_v_partition.overlapping_partition import (
    partition_problem,
    rand_edge_cover_partition,
    expansive_causal_partition,
    heirarchical_partition,
)
import seaborn as sns
import pandas as pd
from cd_v_partition.utils import (
    get_data_from_graph,
    artificial_superstructure,
    get_scores,
    adj_to_dag,
)
from cd_v_partition.causal_discovery import sp_gies, pc
from cd_v_partition.fusion import (
    screen_projections,
    fusion,
    remove_edges_not_in_ss,
    _convert_local_adj_mat_to_graph,
    _union_with_overlaps,
)
from cd_v_partition.vis_partition import create_partition_plot
import matplotlib.pyplot as plt
import time

from build_heirarchical_random_graphs import (
    directed_heirarchical_graph,
)

from tqdm import tqdm
import pdb


def _local_structure_learn(subproblem):
    skel, data = subproblem
    adj_mat = sp_gies(data, skel=skel, outdir=None)
    return adj_mat


def generate_heirarchical_instance(num_nodes=50, superstructure_mode="artificial"):
    ## Generate a random network with heirarchical structure and corresponding dataset
    G_star_graph = directed_heirarchical_graph(num_nodes=num_nodes)
    G_star_adj = nx.adjacency_matrix(G_star_graph)
    assert set(G_star_graph.nodes()) == set(range(G_star_graph.number_of_nodes()))
    nodes = list(range(G_star_graph.number_of_nodes()))

    ## Generate data
    df = get_data_from_graph(
        nodes,
        list(G_star_graph.edges()),
        nsamples=int(1e4),
        iv_samples=0,
    )[1]

    ## Find the 'superstructure'
    # artificially create superstructure
    if superstructure_mode == "artificial":
        superstructure = artificial_superstructure(G_star_adj, frac_extraneous=0.1)
    elif superstructure_mode == "PC":
        df_obs = df.drop(columns=["target"])
        data_obs = df_obs.to_numpy()
        superstructure, _ = pc(data_obs, alpha=0.1, outdir=None)
    return G_star_graph, G_star_adj, superstructure, df


def generate_partitions(superstructure, disjoint_method=heirarchical_partition):
    disjoint_partition = disjoint_method(superstructure)
    causal_partition = expansive_causal_partition(superstructure, disjoint_partition)
    edge_cover_partition = rand_edge_cover_partition(superstructure, disjoint_partition)

    return disjoint_partition, causal_partition, edge_cover_partition


def get_estimated_graphs_unparallelized(
    superstructure,
    partition,
    df,
    G_star,
    run_serial=False,
    full_cand_set=False,
    screen=False,
    ss_subset=True,
):
    start = time.time()
    # Break up problem according to provided partition
    subproblems = partition_problem(partition, superstructure, df)

    # Local learn
    results = []
    print("Beginning local learning.")

    for subproblem in tqdm(subproblems):
        results.append(_local_structure_learn(subproblem))

    # Merge globally
    data_obs = df.drop(columns=["target"]).to_numpy()
    if screen:
        est_graph_partition = screen_projections(
            superstructure,
            partition,
            results,
            ss_subset=ss_subset,
            finite_lim=True,
            data=data_obs,
        )
    else:
        est_graph_partition = fusion(
            superstructure, partition, results, data_obs, full_cand_set=full_cand_set
        )

    # Call serial method
    if run_serial:
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
        _ = get_scores(["CD-serial"], [subselected_serial_DiGraph], G_star)

    _ = get_scores(["CD-partition"], [est_graph_partition], G_star)

    return results, est_graph_partition, subselected_serial_DiGraph


def visualize_partitions(
    disjoint_partition,
    causal_partition,
    G_star_graph,
    superstructure,
    nodes,
    outdir="./",
):
    superstructure_graph = adj_to_dag(superstructure)

    print("PLOTTING DISJOINT PARTITION IN SUPERSTRUCTURE")
    create_partition_plot(
        superstructure_graph,
        nodes,
        disjoint_partition,
        "{}/trial_disjoint.png".format(outdir),
    )
    print("PLOTTING CAUSAL PARTITION IN SUPERSTRUCTURE")
    create_partition_plot(
        superstructure_graph,
        nodes,
        causal_partition,
        "{}/trial_causal.png".format(outdir),
    )
    # visualize partition with respect to original graph
    print("PLOTTING DISJOINT PARTITION IN G_STAR")
    create_partition_plot(
        G_star_graph,
        nodes,
        disjoint_partition,
        "{}/G_star_disjoint.png".format(outdir),
    )
    print("PLOTTING CAUSAL PARTITION IN G_STAR")
    create_partition_plot(
        G_star_graph,
        nodes,
        causal_partition,
        "{}/G_star_causal.png".format(outdir),
    )
    return


def find_edges_in_overlap(graph, nodes_in_overlap):
    edgelist = []
    for edge in graph.edges():
        if edge[0] in nodes_in_overlap and edge[1] in nodes_in_overlap:
            edgelist.append(edge)
    return edgelist


def find_nodes_in_overlap(nodelist, partition):
    node_to_partition = dict(zip(nodelist, [[] for _ in np.arange(len(nodelist))]))
    for key, value in partition.items():
        for node in value:
            node_to_partition[node] += [key]

    # Find nodes in the overlap based on this dictionary
    def _find_overlaps(partition):
        overlaps = []
        for node, comm in partition.items():
            if len(comm) > 1:
                overlaps.append(node)

        return overlaps

    overlaps = _find_overlaps(node_to_partition)
    return overlaps


def run_single_trial(verbose=False):
    outdir = "./"

    ## Generate a random network with heirarchical structure and corresponding dataset
    G_star_graph, G_star_adj, superstructure, df = generate_heirarchical_instance(
        num_nodes=50, superstructure_mode="artificial"
    )
    # post-processing
    assert set(G_star_graph.nodes()) == set(range(G_star_graph.number_of_nodes()))
    nodes = list(range(G_star_graph.number_of_nodes()))
    df_obs = df.drop(columns=["target"])
    data_obs = df_obs.to_numpy()

    ## Partition
    # create a disjoint, expansive causal, and random edge-covering partition using heirarchical methods
    disjoint_partition, causal_partition, edge_cover_partition = generate_partitions(
        superstructure, disjoint_method=heirarchical_partition
    )
    target_partition = causal_partition

    # visualize the partition
    if verbose:
        visualize_partitions(
            disjoint_partition, causal_partition, G_star_graph, superstructure, nodes
        )
    (
        local_cd_adj_mats,
        est_graph_partition,
        est_graph_serial,
    ) = get_estimated_graphs_unparallelized(
        superstructure,
        target_partition,
        df,
        G_star_adj.toarray(),
        screen=True,
        run_serial=True,
    )

    # Find the "union graph" used by screen partitions as an intermediate
    local_cd_graphs = _convert_local_adj_mat_to_graph(
        target_partition, local_cd_adj_mats
    )
    global_graph = _union_with_overlaps(local_cd_graphs)

    if verbose:
        pos = nx.spring_layout(G_star_graph)
        _, axes = plt.subplots(nrows=1, ncols=3)
        axes[0].set_title("G star")
        nx.draw(G_star_graph, ax=axes[0], pos=pos)

        axes[1].set_title("Union graph")
        nx.draw(global_graph, ax=axes[1], pos=pos)

        axes[2].set_title("Estimated graph")
        nx.draw(est_graph_partition, ax=axes[2], pos=pos)
        plt.show()

    # Find edges in global graph not in est_graph_partition
    initial_edge_set = set(global_graph.edges())
    final_edge_set = set(est_graph_partition.edges())
    discarded_edges = initial_edge_set.difference(final_edge_set)

    # Tally true positive and false positive discards
    discard_TP_count = 0
    discard_FP_count = 0
    for edge in discarded_edges:
        if edge in G_star_graph.edges():
            discard_TP_count += 1
        else:
            discard_FP_count += 1
    discard_FP_ratio = np.divide(discard_FP_count, discard_TP_count + discard_FP_count)

    # Compare to set of all candidate discard edges, i.e. all edges in overlap
    nodes_in_overlap = find_nodes_in_overlap(nodes, target_partition)
    edges_in_overlap = find_edges_in_overlap(global_graph, nodes_in_overlap)
    candidate_TP_count = 0
    candidate_FP_count = 0
    for edge in edges_in_overlap:
        if edge in G_star_graph.edges():
            candidate_TP_count += 1
        else:
            candidate_FP_count += 1
    candidate_FP_ratio = np.divide(
        candidate_FP_count, candidate_TP_count + candidate_FP_count
    )

    return discard_FP_ratio, candidate_FP_ratio


def aggregate_multiple_trials(num_trials=50):
    discard_FP_ratio_list = []
    candidate_FP_ratio_list = []
    for _ in range(num_trials):
        discard_FP_ratio, candidate_FP_ratio = run_single_trial(verbose=False)
        discard_FP_ratio_list.append(discard_FP_ratio)
        candidate_FP_ratio_list.append(candidate_FP_ratio)

    pdb.set_trace()
    # plotting
    _, ax = plt.subplots()
    ax.violinplot([discard_FP_ratio_list, candidate_FP_ratio_list])
    ax.set_xticks(
        [y + 1 for y in range(2)],
        labels=[
            "Edges discarded by screen partition",
            "Candidate edges",
        ],
    )
    ax.set_ylabel("Percentage of edges which are False Positives")
    ax.set_title(
        f"Results over {num_trials} independent trials \n based on hierarchical partition of hierarchical network"
    )
    plt.show()


aggregate_multiple_trials()
