# Imports
import numpy as np
import networkx as nx

from tqdm import tqdm

# Imports for code development
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pdb
from diagnostics import assess_superstructure, find_misclassified_edges, localize_errors
from helpers import artificial_superstructure
from build_heirarchical_random_graphs import (
    directed_heirarchical_graph,
)

from cd_v_partition.utils import (
    get_data_from_graph,
    delta_causality,
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
    PEF_partition,
    modularity_partition,
)
from cd_v_partition.vis_partition import create_partition_plot
from cd_v_partition.fusion import fusion


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


def _local_structure_learn(subproblem):
    skel, data = subproblem
    adj_mat = sp_gies(data, skel=skel, outdir=None)
    return adj_mat


def run_causal_discovery_nonparallelized(
    partition, superstructure, df, full_cand_set=False
):
    subproblems = partition_problem(partition, superstructure, df)
    results = []
    for subproblem in tqdm(subproblems):
        results.append(_local_structure_learn(subproblem))

    df_obs = df.drop(columns=["target"])
    data_obs = df_obs.to_numpy()

    fused_graph = fusion(partition, results, data_obs, full_cand_set=full_cand_set)
    return subproblems, results, fused_graph


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


def single_trial_local_performance(verbose=True):
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


def single_trial_partition_comparison(
    base_partition=heirarchical_partition, verbose=True
):
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
        superstructure, disjoint_method=base_partition
    )
    PEF_style_partition = PEF_partition(df, min_size_frac=0.05)

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
    (
        causal_subproblems,
        causal_results,
        causal_fused_A_X_s,
    ) = run_causal_discovery_nonparallelized(causal_partition, superstructure, df)
    (
        disjoint_subproblems,
        disjoint_results,
        disjoint_fused_A_X_s,
    ) = run_causal_discovery_nonparallelized(disjoint_partition, superstructure, df)
    (
        edge_covering_subproblems,
        edge_covering_results,
        edge_covering_fused_A_X_s,
    ) = run_causal_discovery_nonparallelized(edge_cover_partition, superstructure, df)
    (
        PEF_subproblems,
        PEF_results,
        PEF_fused_A_X_s,
    ) = run_causal_discovery_nonparallelized(
        PEF_style_partition, superstructure, df, full_cand_set=True
    )
    if verbose:
        print("Successfully fused partition output.")

    # ## Assess output
    # Compare the results of the fused results from the overlapping vs disjoint partition
    serial_metrics = get_scores(["CD serial "], [A_X_v], G_star_graph)
    causal_metrics = get_scores(
        ["CD causal partition"], [causal_fused_A_X_s], G_star_graph
    )
    disjoint_metrics = get_scores(
        ["CD disjoint partition"], [disjoint_fused_A_X_s], G_star_graph
    )
    edge_covering_metrics = get_scores(
        ["CD edge-covering partition"], [edge_covering_fused_A_X_s], G_star_graph
    )
    PEF_metrics = get_scores(["CD PEF partition"], [PEF_fused_A_X_s], G_star_graph)

    return (
        serial_metrics,
        causal_metrics,
        disjoint_metrics,
        edge_covering_metrics,
        PEF_metrics,
    )


def multiple_trial_local_performance(num_trials=10):
    partition_local_shd = []
    partition_global_shd = []
    serial_local_shd = []
    serial_global_shd = []
    for _ in range(num_trials):
        partition_trial_results, serial_trial_results = single_trial_local_performance(
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


def multiple_trial_partition_comparisons(
    base_partition=heirarchical_partition, metric="shd", num_trials=10
):
    serial_metric_list = []
    causal_metric_list = []
    disjoint_metric_list = []
    edge_covering_metric_list = []
    PEF_metric_list = []
    for _ in range(num_trials):
        (
            serial_metrics,
            causal_metrics,
            disjoint_metrics,
            edge_covering_metrics,
            PEF_metrics,
        ) = single_trial_partition_comparison(
            base_partition=base_partition, verbose=False
        )
        serial_metric_list.append(serial_metrics)
        causal_metric_list.append(causal_metrics)
        disjoint_metric_list.append(disjoint_metrics)
        edge_covering_metric_list.append(edge_covering_metrics)
        PEF_metric_list.append(PEF_metrics)

    # extract target metric
    score_outputs = ["shd", "sid", "auc", "tpr", "fpr"]

    def _extract_target_metric(
        metric_list, target_metric=metric, score_outputs=score_outputs
    ):
        assert len(score_outputs) == len(metric_list[0])
        target_metric_idx = score_outputs.index(target_metric)
        return [metrics[target_metric_idx] for metrics in metric_list]

    # plotting
    _, ax = plt.subplots()
    ax.violinplot(
        [
            _extract_target_metric(serial_metric_list),
            _extract_target_metric(causal_metric_list),
            _extract_target_metric(disjoint_metric_list),
            _extract_target_metric(edge_covering_metric_list),
            _extract_target_metric(PEF_metric_list),
        ]
    )
    ax.set_xticks(
        [y + 1 for y in range(5)],
        labels=[
            "Serial results",
            "Fused causal partition",
            "Fused disjoint partition",
            "Fused edge-covering partition",
            "Fused PEF partition",
        ],
    )
    ax.set_ylabel(metric)
    ax.set_title(
        f"Results over {num_trials} independent trials \n base partition is {str(base_partition)}"
    )
    plt.show()
    return


def visualize_modularity_vs_heirarchical_partitions():
    outdir = "./"

    ## Generate a problem instance
    # G_star, data, and superstructure
    G_star_graph, _, superstructure, df = generate_heirarchical_instance(
        num_nodes=50, superstructure_mode="artificial"
    )
    # post-processing
    assert set(G_star_graph.nodes()) == set(range(G_star_graph.number_of_nodes()))
    nodes = list(range(G_star_graph.number_of_nodes()))
    superstructure_graph = adj_to_dag(superstructure)

    ## Partition
    # create a disjoint and expansive causal
    heirarchical_disjoint, heirarchical_causal, _ = generate_partitions(
        superstructure, disjoint_method=heirarchical_partition
    )
    mod_disjoint, mod_causal, _ = generate_partitions(
        superstructure,
        disjoint_method=(lambda adj: modularity_partition(adj, resolution=0.8)),
    )
    PEF_style_partition = PEF_partition(df, min_size_frac=0.05)
    # pdb.set_trace()

    print("DISJOINT HEIRARCHICAL IN SUPERSTRUCTURE")
    create_partition_plot(
        superstructure_graph,
        nodes,
        heirarchical_disjoint,
        "{}/heirarchical_disjoint.png".format(outdir),
    )
    print("DISJOINT MODULARITY IN SUPERSTRUCTURE")
    create_partition_plot(
        superstructure_graph,
        nodes,
        mod_disjoint,
        "{}/modularity_disjoint.png".format(outdir),
    )
    return


def single_trial_localize_errors(num_nodes=50, verbose=False):
    outdir = "./"

    ## Generate a problem instance
    # G_star, data, and superstructure
    G_star_graph, _, superstructure, df = generate_heirarchical_instance(
        num_nodes=num_nodes, superstructure_mode="artificial"
    )
    # post-processing
    assert set(G_star_graph.nodes()) == set(range(G_star_graph.number_of_nodes()))
    nodes = list(range(G_star_graph.number_of_nodes()))
    superstructure_graph = adj_to_dag(superstructure)

    ## Partition
    # create a disjoint and expansive causal
    heirarchical_disjoint, heirarchical_causal, _ = generate_partitions(
        superstructure, disjoint_method=heirarchical_partition
    )
    ## Learning
    # learn globally
    A_X_v = sp_gies(df, skel=superstructure, outdir=None)

    # learn locally over causal partition and fuse result
    (
        _,
        _,
        causal_fused_A_X_s,
    ) = run_causal_discovery_nonparallelized(heirarchical_causal, superstructure, df)

    # Localize errors on partition result
    _ = get_scores(["CD causal partition"], [causal_fused_A_X_s], G_star_graph)
    # returned values are total_dist, fpos_dist, fneg_dist
    causal_distance_to_boundary = localize_errors(
        causal_fused_A_X_s,
        superstructure_graph,
        G_star_graph,
        heirarchical_causal,
        normalized=False,
        verbose=verbose,
        title="Causal partition",
    )
    # if verbose:
    print(
        f"Causal partition has ave. dist = {causal_distance_to_boundary[0]}, fpos dist = {causal_distance_to_boundary[1]}, fneg dist = {causal_distance_to_boundary[2]}"
    )

    # Localize error on serial result
    _ = get_scores(["CD serial "], [adj_to_dag(A_X_v)], G_star_graph)
    serial_distance_to_boundary = localize_errors(
        adj_to_dag(A_X_v),
        superstructure_graph,
        G_star_graph,
        heirarchical_causal,
        normalized=False,
        verbose=verbose,
        title="Serial result",
    )
    # if verbose:
    print(
        f"Serial result has ave. dist = {serial_distance_to_boundary[0]}, fpos dist = {serial_distance_to_boundary[1]}, fneg dist = {serial_distance_to_boundary[2]}"
    )

    return causal_distance_to_boundary, serial_distance_to_boundary


def multiple_trial_localize_errors(num_trials=10, num_nodes=50, metric="overall_dist"):
    causal_error_distances = []
    serial_error_distances = []
    for _ in range(num_trials):
        (
            causal_distance_to_boundary,
            serial_distance_to_boundary,
        ) = single_trial_localize_errors(num_nodes=num_nodes, verbose=False)
        causal_error_distances.append(causal_distance_to_boundary)
        serial_error_distances.append(serial_distance_to_boundary)

    # extract target metric
    score_outputs = ["overall_dist", "fpos_dist", "fneg_dist"]

    def _extract_target_metric(
        metric_list, target_metric=metric, score_outputs=score_outputs, ignore_nan=True
    ):
        assert len(score_outputs) == len(metric_list[0])
        target_metric_idx = score_outputs.index(target_metric)
        if ignore_nan:
            return [
                metrics[target_metric_idx]
                for metrics in metric_list
                if not np.isnan(metrics[target_metric_idx])
            ]
        return [metrics[target_metric_idx] for metrics in metric_list]

    # plotting
    _, ax = plt.subplots()
    ax.violinplot(
        [
            _extract_target_metric(causal_error_distances),
            _extract_target_metric(serial_error_distances),
        ]
    )
    ax.set_xticks(
        [y + 1 for y in range(2)],
        labels=[
            "Causal partition result",
            "Serial result",
        ],
    )
    ax.set_ylabel(metric)
    ax.set_title(f"Results over {num_trials} independent trials")
    plt.show()

    return


multiple_trial_localize_errors(num_trials=100, num_nodes=100, metric="overall_dist")
# single_trial_localize_errors(num_nodes=100, verbose=True)
