import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import pdb


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
    estimate_graph, target_graph, plot=True, ax=None, pos=None
):
    """Identifies false positive and false negative edges in estimate_graph. Optionally displays them.

    Args:
        estimate_graph (nx.DiGraph): the estimated directed graph
        target_graph (nx.DiGraph: the target directed graph

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

        # draw and label nodes
        labels = dict(
            [(idx, str(idx)) for idx in range(target_graph.number_of_nodes())]
        )
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
            ax.legend()
            plt.show()

    # pdb.set_trace()
    return fpos_edges, fneg_edges
