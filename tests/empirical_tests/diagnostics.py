import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import pdb

from cd_v_partition.utils import get_scores


def assess_superstructure(G_star_adj_mat, super_adj_mat):
    """Creates disjoint partition by greedily maximizing modularity. Using networkx built-in implementaiton.

    Args:
        G_star_adj_mat (np.ndarray): the adjacency matrix for the target graph
        super_adj_mat (np.ndarray): the adjacency matrix for the superstructure we're evaluating

    Returns:
        bool: whether or not G_star is a sub-graph of G_super
    """
    G_star = nx.from_numpy_array(G_star_adj_mat, create_using=nx.DiGraph())
    true_edge_set = set(G_star.edges())
    G_super = nx.from_numpy_array(super_adj_mat, create_using=nx.DiGraph())
    super_edge_set = set(G_super.edges())

    graph_confusion_matrix(G_star_adj_mat, super_adj_mat, print_name="SUPERSTRUCTURE")

    # try discarding direction to see if that improves estimate
    undirected_super = G_super.to_undirected()
    undirected_star = G_star.to_undirected()
    graph_confusion_matrix(
        nx.adjacency_matrix(undirected_star),
        nx.adjacency_matrix(undirected_super),
        print_name="UNDIRECTED SUPERSTRUCTURE",
    )

    # test whether every edge in G_star is in G_super
    return true_edge_set.issubset(super_edge_set)


def graph_confusion_matrix(target_adj_mat, estimate_adj_mat, print_name=None):
    """Displays true-positive, false-positive, and false-negative rates for an estimate graph wrt a target graph.

    Args:
        target_adj_mat (np.ndarray): the adjacency matrix for the target graph
        estimate_adj_mat (np.ndarray): the adjacency matrix for the estimate we're evaluating

    Returns:
    """
    G_target = nx.from_numpy_array(target_adj_mat, create_using=nx.DiGraph())
    target_edge_set = set(G_target.edges())
    G_estimate = nx.from_numpy_array(estimate_adj_mat, create_using=nx.DiGraph())
    estimate_edge_set = set(G_estimate.edges())

    # TP = edges in target which are in estimate
    TP_count = len(target_edge_set.intersection(estimate_edge_set))
    # FP = edges in estimate which are NOT in target
    FP_count = len(estimate_edge_set.difference(target_edge_set))
    # FN = edges not in estimate which ARE in target
    FN_count = len(target_edge_set.difference(estimate_edge_set))

    if print_name is None:
        print_name = "GRAPH"

    # display
    print(
        f"""ASSESSING {print_name} ESTIMATE \n
            total edges in target = {len(target_edge_set)}, in estimate = {len(estimate_edge_set)} \n
            edges in target which appear in estimate={TP_count} \n
            edges in estimate which are NOT in target = {FP_count} \n
            edges NOT in estimate which ARE in target = {FN_count}"""
    )
    return


def find_misclassified_edges(
    estimate_graph,
    target_graph,
    plot=True,
    ax=None,
    pos=None,
    partition=None,
    title=None,
):
    """Identifies false positive and false negative edges in estimate_graph. Optionally displays them.

    Args:
        estimate_graph (nx.DiGraph): the estimated directed graph
        target_graph (nx.DiGraph: the target directed graph
        plot (bool): Optional. Whether to display plots
        ax (matplotlib.axes): Optional. Axes on which to display plots.
        pos (dict): Optional. Positions for nodes in plots.
        partition (dict): Optional. Used to color nodes in plots.

    Returns:
    """
    target_edge_set = set(target_graph.edges())
    estimate_edge_set = set(estimate_graph.edges())

    fpos_edges = list(estimate_edge_set.difference(target_edge_set))
    fneg_edges = list(target_edge_set.difference(estimate_edge_set))

    if plot:
        plot_here = False
        if ax is None:
            plot_here = True
            _, ax = plt.subplots()
        if pos is None:
            pos = nx.spring_layout(target_graph)
        if title is not None:
            ax.set_title(title)

        # draw and label nodes
        labels = dict(
            [(idx, str(idx)) for idx in range(target_graph.number_of_nodes())]
        )
        # partition is provided, color nodes to indicate partition membership
        if partition is not None:
            overlap = find_overlap_nodes(partition)
            nx.draw_networkx_nodes(
                estimate_graph, node_color=list(overlap.values()), pos=pos, ax=ax
            )

        # otherwise, use no node colors
        else:
            nx.draw_networkx_nodes(estimate_graph, pos=pos, ax=ax)
        nx.draw_networkx_labels(estimate_graph, labels=labels, pos=pos, ax=ax)
        # draw true positive edges in solid black lines
        tpos_edges = list(estimate_edge_set.intersection(target_edge_set))
        nx.draw_networkx_edges(estimate_graph, pos=pos, edgelist=tpos_edges, ax=ax)

        # draw false positive edges in solid red lines
        nx.draw_networkx_edges(
            estimate_graph,
            pos=pos,
            edgelist=fpos_edges,
            edge_color="red",
            ax=ax,
            label="false positive",
        )
        # draw false negative edges in dashed red lines
        nx.draw_networkx_edges(
            estimate_graph,
            pos=pos,
            edgelist=fneg_edges,
            edge_color="orange",
            style="dashed",
            ax=ax,
            label="false negative",
        )
        if plot_here:
            legend_elements = [
                Line2D([0], [0], color="r", label="false positive"),
                Line2D([0], [0], color="orange", ls="--", label="false negative"),
                Patch(facecolor="yellow", edgecolor="black", label="Nodes in overlap"),
                Patch(facecolor="purple", edgecolor="black", label="Non-overlap"),
            ]
            ax.legend(handles=legend_elements)
            plt.show()

    return fpos_edges, fneg_edges


def localize_errors(
    learned_graph: nx.DiGraph,
    superstructure_graph: nx.DiGraph,
    G_star_graph: nx.DiGraph,
    partition: dict,
    normalized=True,
    verbose=False,
    title=None,
):
    """Identifies errors in learned_adj_mat and computes their distance to the boundary
    of the provided partition. Distance is computed in the superstructure.

    Args:
        learned_graph (nx.DiGraph): the estimated directed graph
        superstructure_graph (nx.DiGraph): the superstructure
        G_star_graph (nx.DiGraph): G_star
        partition (dict): partition as a dictionary {comm_id : [nodes]}
        normalized (bool): default = True. Whether to normalize distance by diameter
        verbose (bool): optional print/plots

    Returns:
        total_dist (float): average distance to boundary of errors. If normalized=True,
                                        value is normalized by diameter of superstructure.
                                        Value will be nan if no false edges occur.
        fpos_dist (float): avergage distance-to-boundary of false pos edges. If normalized=True,
                                        value is normalized by diameter of superstructure.
                                        Value will be nan if no false positive edges occur.
        fneg_dist (float): avergage distance-to-boundary of false neg edges. If normalized=True,
                                        value is normalized by diameter of superstructure
                                        Value will be nan if no false negative edges occur.
    """
    if verbose:
        pos = nx.spring_layout(G_star_graph)
        for graph, name in [
            (G_star_graph, "G star"),
            (superstructure_graph, "superstructure"),
            (learned_graph, "learned"),
        ]:
            plt.figure()
            plt.title(name)
            nx.draw(graph, pos=pos)
            plt.show()

    fpos_edges, fneg_edges = find_misclassified_edges(
        learned_graph, G_star_graph, plot=verbose, partition=partition, title=title
    )

    # pre-compute pairwise shortest paths in (undirected) superstructure
    undirected_super = superstructure_graph.to_undirected()
    all_pairs_shortest_path_dict = dict(
        nx.all_pairs_shortest_path_length(undirected_super)
    )
    fpos_dists = []
    for edge in fpos_edges:
        fpos_dists.append(
            find_dist_to_partition(edge, partition, all_pairs_shortest_path_dict)
        )
    fneg_dists = []
    for edge in fneg_edges:
        fneg_dists.append(
            find_dist_to_partition(edge, partition, all_pairs_shortest_path_dict)
        )

    ## Aggregate and return
    total_dist = np.mean(fpos_dists + fneg_dists)
    fpos_dist = np.mean(fpos_dists)
    fneg_dist = np.mean(fneg_dists)

    if normalized:
        # Take advantage of precomputed shortest paths to compute diameter
        eccentricity = nx.eccentricity(
            undirected_super, sp=all_pairs_shortest_path_dict
        )
        diam = nx.diameter(undirected_super, e=eccentricity)
        return total_dist / diam, fpos_dist / diam, fneg_dist / diam
    return total_dist, fpos_dist, fneg_dist


def find_dist_to_partition(edge: tuple, partition: dict, pairwise_dist: dict):
    """Computes minimum distance of either end point of edge to the boundary
    of the provided partition. Distance is taken from pairwise_dist.

    Args:
        edge (tuple): edge whose distance to boundary is computed
        graph (nx.DiGraph): the graph in which distance is evaluated
        partition (dict): partition as a dictionary {comm_id : [nodes]}
        pairwise_dist (dict): querying pairwise_dict[u][v] returns shortest path length from u to v

    Returns:
        float: distance of edge to boundary
    """
    min_dist = np.inf
    # find distance for each endpoint. Return minimum of the two
    for v in edge:
        clusters_containing_v = get_cluster_membership(v, partition)
        # if either endpoint is in multiple clusters, then distance to boundary of the edge is zero
        if len(clusters_containing_v) > 1:
            return 0
        # otherwise, this endpoint is contained in exactly one cluster.
        else:
            v_cluster_idx = clusters_containing_v[0]
            nodes_in_other_clusters = _get_nodes_in_other_clusters(
                partition, v_cluster_idx
            )
            # iterate through all nodes in different clusters than v
            for node in nodes_in_other_clusters:
                # if we find a node in a different cluster than v's that's closer than the min_dist
                # observed so far, update min_dist.
                if pairwise_dist[v][node] < min_dist:
                    min_dist = pairwise_dist[v][node]
    if min_dist == np.inf:
        pdb.set_trace()
    return min_dist


def _get_nodes_in_other_clusters(partition: dict, target_cluster_idx: int):
    """Finds indices of nodes that belong to clusters other than target_cluster_idx.
    Note that for overlapping partitions, a node can BOTH be in the target_cluster_idx
    cluster AND be in a different cluster.

    Args:
        partition (dict): partition as a dictionary {comm_id : [nodes]}
        target_cluster_idx (int): index of cluster in partition to exclude

    Returns:
        set: indices of nodes that belong to different clusters than target_cluster_idx
    """
    nodes_in_other_clusters = set()
    for idx in range(len(partition)):
        if idx != target_cluster_idx:
            # add all nodes in clusters other than target_cluster_idx to set
            nodes_in_other_clusters.update(set(partition[idx]))
    return nodes_in_other_clusters


def get_cluster_membership(v, partition):
    """Finds indices of clusters in partition that v belongs to

    Args:
        v (int): index of node
        partition (dict): partition as a dictionary {comm_id : [nodes]}

    Returns:
        list: indices of clusters in partition that v belongs to
    """
    membership_idx_list = []
    for idx, cluster in enumerate(list(partition.values())):
        if v in cluster:
            membership_idx_list.append(idx)

    return membership_idx_list


def find_overlap_nodes(partition):
    """Returns a dictionary with entries {node: 0} if node is not in any
    overlap, or {node: 1} if node is in some overlap

    Args:
        partition (dict): partition as a dictionary {comm_id : [nodes]}

    Returns:
        dict: dictionary indicating whether a node is or is not in any overlap
    """
    # find set of all nodes
    node_set = set()
    for cluster in list(partition.values()):
        node_set.update(set(cluster))

    is_in_overlap = dict()
    for v in node_set:
        # v is in the overlap if multiple clusters contain v
        clusters_containing_v = get_cluster_membership(v, partition)
        if len(clusters_containing_v) > 1:
            is_in_overlap[v] = 1
        else:
            is_in_overlap[v] = 0
    return is_in_overlap
