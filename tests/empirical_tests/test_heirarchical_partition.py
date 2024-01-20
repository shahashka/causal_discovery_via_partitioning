# Imports
import numpy as np
import networkx as nx
import functools

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Imports for code development
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pdb
from diagnostics import assess_superstructure, find_misclassified_edges
from helpers import artificial_superstructure
from duplicate_functions import create_two_comms
from build_heirarchical_random_graphs import (
    duplicate_get_random_graph_data,
    directed_heirarchical_graph,
)

from cd_v_partition.utils import (
    get_random_graph_data,
    get_data_from_graph,
    delta_causality,
    edge_to_adj,
    adj_to_dag,
    evaluate_partition,
    get_scores,
)
from cd_v_partition.causal_discovery import pc, sp_gies
from cd_v_partition.overlapping_partition import (
    partition_problem,
    heirarchical_partition,
    expansive_causal_partition,
    rand_edge_cover_partition,
)
from cd_v_partition.vis_partition import create_partition_plot
from cd_v_partition.fusion import fusion


def generate_heirarchical_instance(num_nodes=50, superstructure_mode="artificial"):
    ## Generate a random network with heirarchical structure and corresponding dataset
    G_star_graph = directed_heirarchical_graph(num_nodes=50)
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


def _local_structure_learn(subproblem):
    skel, data = subproblem
    adj_mat = sp_gies(data, skel=skel, outdir=None)
    return adj_mat


def visualize_partitions(
    disjoint_partition,
    causal_partition,
    G_star_graph,
    superstructure,
    nodes,
    outdir="./",
):
    superstructure_graph = adj_to_dag(superstructure)

    # undirected edges in superstructure adjacency become bidirected
    evaluate_partition(disjoint_partition, superstructure_graph, nodes)
    evaluate_partition(causal_partition, superstructure_graph, nodes)

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


def heirarchical_vs_serial(verbose=True):
    outdir = "./"

    ## Generate a random network with heirarchical structure and corresponding dataset
    # Generate a problem instance
    # G_star, data, and superstructure
    G_star_graph, G_star_adj, superstructure, df = generate_heirarchical_instance(
        num_nodes=50, superstructure_mode="artificial"
    )
    # post-processing
    assert set(G_star_graph.nodes()) == set(range(G_star_graph.number_of_nodes()))
    nodes = list(range(G_star_graph.number_of_nodes()))
    df_obs = df.drop(columns=["target"])
    data_obs = df_obs.to_numpy()

    if verbose:
        plt.figure()
        nx.draw(G_star_graph)
        plt.show()

        plt.figure()
        nx.draw(nx.from_numpy_array(superstructure))
        plt.show()

        assess_superstructure(G_star_adj, superstructure)

    ## Partition
    # create a disjoint, expansive causal, and random edge-covering partition using heirarchical methods
    disjoint_partition, causal_partition, edge_cover_partition = generate_partitions(
        superstructure, disjoint_method=heirarchical_partition
    )

    # visualize the partition
    if verbose:
        visualize_partitions(
            disjoint_partition, causal_partition, G_star_graph, superstructure, nodes
        )

    ## Learning Globally
    # call the causal learner on the full data A(X_v) and superstructure
    if verbose:
        print("Beginning global learning.")
    A_X_v = sp_gies(df, skel=superstructure, outdir=None)

    ## Learning Locally
    # Call the causal learner on subsets of the data F({A(X_s)}) and sub-structures
    subproblems = partition_problem(causal_partition, superstructure, df)
    results = []

    # non-parallelized version
    if verbose:
        print("Beginning local learning.")
    for subproblem in tqdm(subproblems):
        results.append(_local_structure_learn(subproblem))

    # Merge the subset learned graphs
    fused_A_X_s = fusion(causal_partition, results, data_obs)
    if verbose:
        print("Successfully fused partition output.")

    # ## Assess output
    partition_learning_metrics = {}
    serial_learning_metrics = {}
    partition_learning_normalized_shd = []
    serial_learning_normalized_shd = []
    if verbose:
        # Compare the results of the A(X_v) and F({A(X_s)})
        # You see the following printed for 'CD-serial' and 'CD-partition'
        # SHD: 'number of wrong edges'
        # SID: 'ignore this one'
        # AUC: 'auroc where edge is 1, no edge is 0',
        # TPR,FPR: ('true positive rate', 'false positive rate')
        # delta_causality(
        #     A_X_v, nx.to_numpy_array(fused_A_X_s), nx.to_numpy_array(G_star_graph)
        # )
        delta_causality(A_X_v, fused_A_X_s, G_star_graph)
        shd, _, auc, tpr, fpr = get_scores(["CD serial "], [A_X_v], G_star_graph)
        shd, _, auc, tpr, fpr = get_scores(
            ["CD partition"], [fused_A_X_s], G_star_graph
        )

    ## Asses output globally (SHD/size)
    shd, _, auc, tpr, fpr = get_scores(["CD partition"], [fused_A_X_s], G_star_graph)
    partition_learning_metrics["normalized_global_SHD"] = np.mean(shd / len(nodes))
    if verbose:
        print(f"Partition fused graph: SHD={shd:.2f}, SHD/n={shd/len(nodes):.2f} ")

    shd, _, auc, tpr, fpr = get_scores(["CD serial "], [A_X_v], G_star_graph)
    serial_learning_metrics["normalized_global_SHD"] = np.mean(shd / len(nodes))
    if verbose:
        print(f"Serial global graph: SHD={shd:.2f}, SHD/n={shd/len(nodes):.2f} ")

    G_star_subproblems = partition_problem(causal_partition, G_star_adj, df)
    serial_learning_subproblems = partition_problem(causal_partition, A_X_v, df)
    for idx, subsets in enumerate(G_star_subproblems):
        subset_size = subsets[0].shape[0]

        # extract G_star on our subset
        G_star_substructure = subsets[0]
        G_star_subgraph = nx.from_numpy_array(
            G_star_substructure, create_using=nx.DiGraph
        )

        # get result of partition learning
        local_learned_structure = results[idx]
        local_learned_graph = nx.from_numpy_array(
            local_learned_structure, create_using=nx.DiGraph
        )

        # extract serial result on our subset
        serial_subproblem = serial_learning_subproblems[idx]
        serial_substructure = serial_subproblem[0]
        serial_subgraph = nx.from_numpy_array(
            serial_substructure, create_using=nx.DiGraph
        )
        if verbose:
            _, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
            # get a set of positions to use for both graphs
            pos = nx.spring_layout(G_star_subgraph)

            # plot G_star
            nx.draw(G_star_subgraph, pos=pos, with_labels=True, ax=axes[0])
            axes[0].set_title("G_star local")

            # plot partition learning
            # find and draw misclassified edges
            find_misclassified_edges(
                local_learned_graph, G_star_subgraph, plot=True, ax=axes[1], pos=pos
            )
            # nx.draw(local_learned_graph, pos=pos, with_labels=True, ax=axes[1])

        # get_scores returns: shd, sid, auc, tpr, fpr
        shd, _, auc, tpr, fpr = get_scores(
            ["CD partition"], [local_learned_graph], G_star_subgraph
        )
        partition_learning_normalized_shd.append(shd / subset_size)

        if verbose:
            axes[1].set_title(
                f"Partition learned output \n SHD={shd:.2f}, SHD/subset size={shd/subset_size:.2f} \n AUC={auc:.2f}, TPR={tpr:.2f}, FPR = {fpr:.2f}"
            )

            # plot serial learning output on local subset
            # find and draw misclassified edges
            find_misclassified_edges(
                serial_subgraph, G_star_subgraph, plot=True, ax=axes[2], pos=pos
            )
            # nx.draw(serial_subgraph, pos=pos, with_labels=True, ax=axes[2])

        # get_scores returns: shd, sid, auc, tpr, fpr
        shd, _, auc, tpr, fpr = get_scores(
            ["Serial local subgraph"], [serial_subgraph], G_star_subgraph
        )
        serial_learning_normalized_shd.append(shd / subset_size)

        if verbose:
            axes[2].set_title(
                f"Serial learned output on local subset \n SHD={shd:.2f}, SHD/subset size={shd/subset_size:.2f} \n  AUC={auc:.2f}, TPR={tpr:.2f}, FPR = {fpr:.2f}"
            )
            legend_elements = [
                Line2D([0], [0], color="r", label="false positive"),
                Line2D([0], [0], color="orange", ls="--", label="false negative"),
            ]
            plt.legend(handles=legend_elements)
            plt.show()

    partition_learning_metrics["normalized_local_SHD"] = np.mean(
        partition_learning_normalized_shd
    )
    serial_learning_metrics["normalized_local_SHD"] = np.mean(
        serial_learning_normalized_shd
    )

    return partition_learning_metrics, serial_learning_metrics


def multiple_heirarchical_trials(num_trials=10):
    partition_local_shd = []
    partition_global_shd = []
    serial_local_shd = []
    serial_global_shd = []
    for _ in range(num_trials):
        partition_trial_results, serial_trial_results = heirarchical_vs_serial(
            verbose=False
        )
        partition_local_shd.append(partition_trial_results["normalized_local_SHD"])
        partition_global_shd.append(partition_trial_results["normalized_global_SHD"])
        serial_local_shd.append(serial_trial_results["normalized_local_SHD"])
        serial_global_shd.append(serial_trial_results["normalized_global_SHD"])

    # plotting
    _, ax = plt.subplots()
    ax.violinplot(
        [partition_local_shd, partition_global_shd, serial_local_shd, serial_global_shd]
    )
    ax.set_xticks(
        [y + 1 for y in range(4)],
        labels=[
            "Learning on subset \n SHD/Subset size",
            "Fused output \n SHD/number of nodes",
            "Serial learning, result on subset \n SHD/Subset size",
            "Serial learning, result on whole graph \n SHD/n",
        ],
    )
    plt.show()
    return


def learn_on_partition(partition, superstructure, df):
    subproblems = partition_problem(partition, superstructure, df)
    results = []
    for subproblem in tqdm(subproblems):
        results.append(_local_structure_learn(subproblem))

    df_obs = df.drop(columns=["target"])
    data_obs = df_obs.to_numpy()

    fused_graph = fusion(partition, results, data_obs)
    return subproblems, results, fused_graph


def single_trial_partition_comparison(verbose=True):
    outdir = "./"

    ## Generate a problem instance
    # G_star, data, and superstructure
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

    if verbose:
        plt.figure()
        nx.draw(G_star_graph)
        plt.show()
        plt.figure()
        nx.draw(nx.from_numpy_array(superstructure))
        plt.show()
        assess_superstructure(G_star_adj, superstructure)

    if verbose:
        # visualize the partitions
        visualize_partitions(
            disjoint_partition, causal_partition, G_star_graph, superstructure, nodes
        )

    ## Learning Globally
    # call the causal learner on the full data A(X_v) and superstructure
    if verbose:
        print("Beginning global learning.")
    A_X_v = sp_gies(df, skel=superstructure, outdir=None)

    ## Learning Locally
    # non-parallelized version
    if verbose:
        print("Beginning local learning.")
    causal_subproblems, causal_results, causal_fused_A_X_s = learn_on_partition(
        causal_partition, superstructure, df
    )
    disjoint_subproblems, disjoint_results, disjoint_fused_A_X_s = learn_on_partition(
        disjoint_partition, superstructure, df
    )
    (
        edge_covering_subproblems,
        edge_covering_results,
        edge_covering_fused_A_X_s,
    ) = learn_on_partition(edge_cover_partition, superstructure, df)
    if verbose:
        print("Successfully fused partition output.")

    # ## Assess output
    # Compare the results of the fused results from the overlapping vs disjoint partition
    serial_shd, _, auc, tpr, fpr = get_scores(["CD serial "], [A_X_v], G_star_graph)
    causal_shd, _, auc, tpr, fpr = get_scores(
        ["CD causal partition"], [causal_fused_A_X_s], G_star_graph
    )
    disjoint_shd, _, auc, tpr, fpr = get_scores(
        ["CD disjoint partition"], [disjoint_fused_A_X_s], G_star_graph
    )
    edge_covering_shd, _, auc, tpr, fpr = get_scores(
        ["CD edge-covering partition"], [edge_covering_fused_A_X_s], G_star_graph
    )

    return serial_shd, causal_shd, disjoint_shd, edge_covering_shd


def multiple_trial_partition_comparisons(num_trials=10):
    serial_shd_list = []
    causal_shd_list = []
    disjoint_shd_list = []
    edge_covering_shd_list = []
    for _ in range(num_trials):
        (
            serial_shd,
            causal_shd,
            disjoint_shd,
            edge_covering_shd,
        ) = single_trial_partition_comparison(verbose=False)
        serial_shd_list.append(serial_shd)
        causal_shd_list.append(causal_shd)
        disjoint_shd_list.append(disjoint_shd)
        edge_covering_shd_list.append(edge_covering_shd)

    # plotting
    _, ax = plt.subplots()
    ax.violinplot(
        [serial_shd_list, causal_shd_list, disjoint_shd_list, edge_covering_shd_list]
    )
    ax.set_xticks(
        [y + 1 for y in range(4)],
        labels=[
            "Serial results",
            "Fused causal partition",
            "Fused disjoint partition",
            "Fused edge-covering partition",
        ],
    )
    ax.set_ylabel("SHD")
    ax.set_title(f"Results over {num_trials} independent trials")
    plt.show()
    return


def multiple_heirarchical_trials(num_trials=10):
    partition_local_shd = []
    partition_global_shd = []
    serial_local_shd = []
    serial_global_shd = []
    for _ in range(num_trials):
        partition_trial_results, serial_trial_results = heirarchical_vs_serial(
            verbose=False
        )
        partition_local_shd.append(partition_trial_results["normalized_local_SHD"])
        partition_global_shd.append(partition_trial_results["normalized_global_SHD"])
        serial_local_shd.append(serial_trial_results["normalized_local_SHD"])
        serial_global_shd.append(serial_trial_results["normalized_global_SHD"])

    # plotting
    _, ax = plt.subplots()
    ax.violinplot(
        [partition_local_shd, partition_global_shd, serial_local_shd, serial_global_shd]
    )
    ax.set_xticks(
        [y + 1 for y in range(4)],
        labels=[
            "Learning on subset \n SHD/Subset size",
            "Fused output \n SHD/number of nodes",
            "Serial learning, result on subset \n SHD/Subset size",
            "Serial learning, result on whole graph \n SHD/n",
        ],
    )
    plt.show()
    return


multiple_trial_partition_comparisons(num_trials=20)
