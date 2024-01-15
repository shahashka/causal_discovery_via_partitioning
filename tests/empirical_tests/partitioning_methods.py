import numpy as np
import networkx as nx

from cd_v_partition.vis_partition import create_partition_plot

import pdb


def expansive_causal_partition(adj_mat, partition):
    """Creates a causal partition by adding the outer-boundary of each cluster to that cluster.

    Args:
        adj_mat (np.ndarray): the adjacency matrix for the superstructure
        partition (dict): the estimated partition as a dictionary {comm_id : [nodes]}

    Returns:
        dict: the causal partition as a dictionary {comm_id : [nodes]}
    """
    G = nx.from_numpy_array(adj_mat)

    causal_partition = dict()
    for idx, c in enumerate(list(partition.values())):
        outer_node_boundary = nx.node_boundary(G, c)
        expanded_cluster = set(c).union(outer_node_boundary)
        causal_partition[idx] = list(expanded_cluster)
    return causal_partition


def modularity_partition(adj_mat, resolution=1, cutoff=2, best_n=2):
    """Creates disjoint partition by greedily maximizing modularity. Using networkx built-in implementaiton.

    Args:
        adj_mat (np.ndarray): the adjacency matrix for the superstructure
        resolution (float): resolution parameter, trading off intra- versus inter-group edges.
        cutoff (int): lower limit on number of communities before termination
        best_n (int): upper limit on number of communities before termination
        See networkx documentation for more.

    Returns:
        dict: the estimated partition as a dictionary {comm_id : [nodes]}
    """
    G = nx.from_numpy_array(adj_mat)
    community_lists = nx.community.greedy_modularity_communities(
        G, cutoff=cutoff, best_n=best_n
    )

    partition = dict()
    for idx, c in enumerate(community_lists):
        partition[idx] = list(c)
    return partition


def heirarchical_partition(adj_mat, max_community_size=0.5):
    """Creates disjoint partition via heirarchical community detection

    Args:
        adj_mat (np.ndarray): the adjacency matrix for the superstructure
        max_community_szie (float): controls the size of the largest community in returned partition
        See networkx documentation for more.

    Returns:
        dict: the estimated partition as a dictionary {comm_id : [nodes]}
    """
    G = nx.from_numpy_array(adj_mat)
    n = G.number_of_nodes()
    nodes = list(range(n))
    community_iterator = nx.community.girvan_newman(G)
    for communities in community_iterator:
        if max([len(com) for com in communities]) < (max_community_size * n):
            ith_partition = dict()
            for idx, c in enumerate(communities):
                ith_partition[idx] = list(c)
            return ith_partition
        # if an error gets thrown here bc community_iterator ends, it means the algorithm
        # didn't produce any partitions with sufficiently small clusters
    return
