import numpy as np
import networkx as nx
import random

import pdb


def artificial_superstructure(
    G_star_adj_mat, frac_retain_direction=0.1, frac_extraneous=0.5
):
    """Creates a superstructure by discarding some of the directions in edges of G_star and adding
    extraneous edges.

    Args:
        G_star_adj_mat (np.ndarray): the adjacency matrix for the target graph
        frac_retain_direction (float): what percentage of edges will retain their direction information
        frac_extraneous (float): adds frac_extraneous*m many additional edges, for m the number of edges in G_star

    Returns:
        super_adj_mat (np.ndarray): an adjacency matrix for the superstructure we've created
    """
    G_star = nx.from_numpy_array(G_star_adj_mat, create_using=nx.DiGraph())
    true_edge_set = set(G_star.edges())

    # returns a deepcopy
    G_super = G_star.to_undirected()
    # add extraneous edges
    m = G_star.number_of_edges()
    nodes = list(G_star.nodes())
    G_super.add_edges_from(pick_k_random_edges(k=int(frac_extraneous * m), nodes=nodes))

    return nx.adjacency_matrix(G_super).toarray()


def pick_k_random_edges(k, nodes):
    return list(zip(random.choices(nodes, k=k), random.choices(nodes, k=k)))


def latent_projection(node_subset, G_star_adj_mat):
    """Generates the latent projection of G_star onto node_subset

    Args:
        node_subset (list): indices of nodes in G_star on which the latent projection is created
        G_star_adj_mat (np.ndarray): the adjacency matrix for the target graph

    Returns:
        latent_proj_mat (np.ndarray): an adjacency matrix for the latent projection
    """
    G_star = nx.from_numpy_array(G_star_adj_mat, create_using=nx.DiGraph())
    L = nx.DiGraph()
    L.add_nodes_from(node_subset)

    # collect all edges in G_star on node_subset
    for edge in G_star.edges():
        pdb.set_trace()

    return
