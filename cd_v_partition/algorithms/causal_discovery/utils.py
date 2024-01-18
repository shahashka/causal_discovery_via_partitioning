import itertools

import numpy as np


def weight_colliders(adj_mat: np.ndarray, weight: int = 1):
    """
    Find and add weights to collider sets in a given adjacency matrix. Collider sets are x->y<-z
    when there is no edge between $(x,z)$.

    Args:
        adj_mat (np.ndarray): $p \\times p$ adjacency matrix.
        weight (int): Edges that are part of a collider set are weighted with this weight.

    Returns:
        An array representing the weighted adjacency matrix.
    """
    weighted_adj_mat = adj_mat
    for col in np.arange(adj_mat.shape[1]):
        incident_nodes = np.argwhere(adj_mat[:, col] == 1).flatten()

        # For all edges incident on the node corresponding to this column
        for i, j in itertools.combinations(incident_nodes, 2):
            # Filter for directed edges
            if adj_mat[col, i] == 0 and adj_mat[col, j] == 0:
                # Check if a pair of source nodes is connected
                if adj_mat[i, j] == 0 and adj_mat[j, i] == 0:
                    # If not then this is a collider
                    weighted_adj_mat[i, col] = weight
                    weighted_adj_mat[j, col] = weight

    return weighted_adj_mat
