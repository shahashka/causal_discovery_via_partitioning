from __future__ import annotations

import subprocess
import typing
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage


def modularity_partition(
    adj_mat: np.ndarray,
    data: pd.DataFrame,
    resolution: int = 1,
    cutoff: int = 1,
    best_n: int = None,
):
    """
    Creates disjoint partition by greedily maximizing modularity. Using
    networkx built-in implementaiton.

    Args:
        adj_mat (np.ndarray): the adjacency matrix for the superstructure
        data (pd.DataFrame): unused parameter
        resolution (float): resolution parameter, trading off intra- versus
            inter-group edges.
        cutoff (int): lower limit on number of communities before termination
        best_n (int): upper limit on number of communities before termination
        See networkx documentation for more.

    Returns:
        dict: the estimated partition as a dictionary {comm_id : [nodes]}
    """
    G = nx.from_numpy_array(adj_mat)
    community_lists = nx.community.greedy_modularity_communities(
        G, resolution=resolution, cutoff=cutoff, best_n=best_n
    )

    partition = dict()
    for idx, c in enumerate(community_lists):
        partition[idx] = list(c)
    return partition


def expansive_causal_partition(
    adj_mat: np.ndarray,
    data: pd.DataFrame,
    resolution: int = 1,
    cutoff: int = 1,
    best_n: int = None,
):
    """
    Creates a causal partition by adding the outer-boundary of each cluster
    to that cluster.

    First uses greedy modularity to create a disjoint partition, then adds
    the outer-boundary of each cluster to create a causal partition

    Args:
        adj_mat (np.ndarray): the adjacency matrix for the superstructure
        data (Any): unused parameter
        resolution (float): resolution parameter, trading off intra- versus
            inter-group edges.
        cutoff (int): lower limit on number of communities before termination
        best_n (int): upper limit on number of communities before termination

    Returns:
        dict: the causal partition as a dictionary {comm_id : [nodes]}
    """
    partition = modularity_partition(
        adj_mat, data=data, resolution=resolution, cutoff=cutoff, best_n=best_n
    )
    G = nx.from_numpy_array(adj_mat)

    causal_partition = dict()
    for idx, c in enumerate(list(partition.values())):
        outer_node_boundary = nx.node_boundary(G, c)
        expanded_cluster = set(c).union(outer_node_boundary)
        causal_partition[idx] = list(expanded_cluster)
    return causal_partition


def rand_edge_cover_partition(
    adj_mat: np.ndarray,
    data: pd.DataFrame,
    resolution: int = 1,
    cutoff: int = 1,
    best_n: int = None,
):
    """Creates a random edge covering partition.

    Uses greedy modularity to create a disjoint partition. Then, randomly
    chooses cut edges and randomly assigns endpoints to communities.
    Recursively adds any shared endpoints to the same community

    Args:
        adj_mat (np.ndarray): Adjacency matrix for the graph
        data (pd.DataFrame): unused parameter
        resolution (float): resolution parameter, trading off intra-
            versus inter-group edges.
        cutoff (int): lower limit on number of communities before termination
        best_n (int): upper limit on number of communities before termination

    Returns:
        dict: the overlapping partition as a dictionary {comm_id : [nodes]}
    """
    partition = modularity_partition(
        adj_mat, data=data, resolution=resolution, cutoff=cutoff, best_n=best_n
    )
    graph = nx.from_numpy_array(adj_mat)

    def edge_coverage_helper(i, j, comm, cut_edges, node_to_comm):
        if comm not in node_to_comm[i]:
            node_to_comm[i] += [comm]
        if comm not in node_to_comm[j]:
            node_to_comm[j] += [comm]
        cut_edges.remove((i, j))
        return node_to_comm, cut_edges

    node_to_comm = dict()
    for comm_id, comm in partition.items():
        for node in comm:
            node_to_comm[node] = [comm_id]
    cut_edges = []
    for edge in graph.edges():
        if node_to_comm[edge[0]] != node_to_comm[edge[1]]:
            cut_edges.append(edge)

    # Randomly choose a cut edge until all edges are covered
    while len(cut_edges) > 0:
        edge_ind = np.random.choice(np.arange(len(cut_edges)))
        i = cut_edges[edge_ind][0]
        j = cut_edges[edge_ind][1]

        # Randomly choose an endpoint and associated community
        possible_comms = list(set(node_to_comm[i] + node_to_comm[j]))
        comm = np.random.choice(possible_comms)
        node_to_comm, cut_edges = edge_coverage_helper(
            i, j, comm, cut_edges, node_to_comm
        )

    edge_cover_partition = dict()
    # Update the hard partition
    for n, comms in node_to_comm.items():
        for c in comms:
            if c in edge_cover_partition.keys():
                edge_cover_partition[c] += [n]
            else:
                edge_cover_partition[c] = [n]
    return edge_cover_partition


""" BEGIN UNUSED PARTITION ALGS"""


def hierarchical_partition(
    adj_mat: np.ndarray, max_community_size: float = 0.5
):
    """Creates disjoint partition via heirarchical community detection

    Args:
        adj_mat (np.ndarray): the adjacency matrix for the superstructure
        max_community_szie (float): controls the size of the largest community
            in returned partition See networkx documentation for more
            information.

    Returns:
        dict: the estimated partition as a dictionary {comm_id : [nodes]}
    """
    G = nx.from_numpy_array(adj_mat)
    n = G.number_of_nodes()
    community_iterator = nx.community.girvan_newman(G)
    for communities in community_iterator:
        if max([len(com) for com in communities]) < (max_community_size * n):
            ith_partition = dict()
            for idx, c in enumerate(communities):
                ith_partition[idx] = list(c)
            return ith_partition
        # if an error gets thrown here bc community_iterator ends, it means
        # the algorithm didn't produce any partitions with sufficiently small
        # clusters
    return


def oslom_algorithm(
    nodes: typing.Iterable,
    dat_file: Path | str,
    oslom_dir: Path | str,
    structure_type: str | None = "dag",
) -> dict:
    """
    Overlapping partitioning methods which take an input graph
    (superstructure) and partition nodes according to an objective overlapping
    nodes ideally render the partitions conditionally independent.

    Args:
        nodes (): ...
        data_dir (Path | str): The directory containing the *.dat file which
            holds the edges of the structure to partition.
        oslom_dir (Path | str): The directory containing the OSLOM binary
        structure_type (str | None): Specify the structure type as either
            the 'dag',  'superstructure', or 'superstructure_weighted'.
            If weighted then weights in the *.dat are used by OSLOM.
            Defaults to 'dag'.

    Returns:
        The estimated partition as a dictionary {comm_id : [nodes]}
    """
    # Run the OSLOM code externally
    structure_type = structure_type if structure_type is not None else "dag"
    weight_flag = "-w" if "weight" in structure_type else "-uw"
    subprocess.run(
        [
            f"{oslom_dir}/oslom_undir",
            "-f",
            dat_file,
            f"{weight_flag}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    # Read the output partition file and return the partition as a dictionary
    partition_file = f"{dat_file}_oslo_files/tp"
    with open(partition_file, "rb") as f:
        lines = f.readlines()
    lines = lines[1::2]
    lines = [[int(node) for node in line.split()] for line in lines]
    partition = dict(zip(np.arange(len(lines)), lines))
    homeless_nodes = list(nodes)
    for part in lines:
        for n in part:
            if n in homeless_nodes:
                homeless_nodes.remove(n)

    if len(homeless_nodes) > 0:
        partition[len(lines)] = homeless_nodes
    return partition


""" END UNUSED PARTITION ALGS"""


def partition_problem(
    partition: dict, structure: np.ndarray, data: pd.DataFrame
):
    """
    Split the graph structure and dataset according to the given graph
    partition.

    Args:
        partition (dict): the partition as a dictionary {comm_id : [nodes]}
        structure (np.ndarray): the adjacency matrix for the initial structure
        data (pd.DataFrame): the dataset, columns correspond to nodes in the
            graph.

    Returns:
        List of tuples holding the sub structure and data subsets for
        each partition.
    """
    sub_problems = []
    k = list(partition.keys())
    k.sort()
    for i in k:
        sub_nodes = partition[i]
        sub_structure = structure[sub_nodes][:, sub_nodes]
        sub_nodes = list(sub_nodes)
        sub_nodes.append(-1)  # add 'target' vector at the end of dataframe
        sub_data = data.iloc[:, sub_nodes]
        sub_problems.append((sub_structure, sub_data))
    return sub_problems


def PEF_partition(
    adj_mat: np.ndarray,
    data: pd.DataFrame,
    resolution: int = 1,
    cutoff: int = 1,
    best_n: int = None,
    min_size_frac: float = 0.05,
) -> dict:
    """
    Perform the modified hierarchical clustering on the data, as described in
    `Learning Big Gaussian Bayesian Networks: Partition, Estimation and Fusion'

    Args:
        adj_mat (np.ndarray): Adjacency matrix for the graph
        data (pd.DataFrame): the dataset, columns correspond to nodes in the
            graph.
        resolution (float): unused parameter
        cutoff (int): unused parameter
        best_n (int): unuserd parameter
        min_size_frac (float): determines the minimimum returned cluster size.

    Returns:
        The estimated partition as a dictionary {comm_id : [nodes]}
    """
    # Compute agglomerative clustering matrix
    # data_mat has shape num_nodes x num_samples
    data_mat = data.drop(["target"], axis=1).to_numpy().T
    # Using average linkage
    # Datapoint distance is determined by correlation of df columns
    linkage_matrix = linkage(data_mat, method="average", metric="correlation")

    # Extract partitions
    # for each level, we'll want to compute how many big clusters are
    # in that partition
    min_size = int(min_size_frac * data_mat.shape[0])

    # a cluster is big if it has size min_size
    def num_big_clusters(partition, min_size=min_size):
        """
        Counts how many clusters in an input partition are of size at least
        `min_size`.

        Args:
            partition (list): the partition, as a list of sets or list of lists
            min_size (int): the size threshold over which clusters are
                considered "big"

        Returns:
            int: the number of clusters of size at least min_size
        """
        return np.sum([len(cluster) >= min_size for cluster in partition])

    # k_list stores the number of big clusters per partition
    k_list = []

    # For each level, find the clusters for that level
    partitions_list = []
    dct = {i: {i} for i in range(data_mat.shape[0])}
    k_list.append(num_big_clusters(list(dct.values())))
    # get the partition where every node is in its own cluster
    partitions_list.append(list(dct.values()))
    for i, row in enumerate(linkage_matrix, data_mat.shape[0]):
        dct[i] = dct[row[0]].union(dct[row[1]])
        del dct[row[0]]
        del dct[row[1]]
        current_partition = list(dct.values())
        # save partition with clusters ordered from largest to smallest
        current_partition.sort(key=len, reverse=True)
        partitions_list.append(current_partition)
        k_list.append(num_big_clusters(current_partition))

    # select the first partition to hit the largest count of "big clusters"
    max_k = max(k_list)
    max_k_idx = k_list.index(max_k)
    max_k_partition = partitions_list[max_k_idx]

    # reduce the size of max_k_partition until it only contains max_k clusters.
    # do this by merging smaller clusters into larger clusters
    # merging procedure is based on correlation distance
    corr_mat = data.drop(["target"], axis=1).corr().to_numpy()
    dist_mat = np.subtract(1, np.abs(corr_mat))

    # partition should be sorted so that largest clusters appear first
    def cluster_distance(C1, C2, dist_mat=dist_mat) -> float:
        """
        Computes the distance between two clusters, defined as the MINIMUM
        pairwise distance between their elements.

        Args:
            C1 (set): the indices of nodes in the first cluster
            C2 (set): the indices of nodes in the second cluster
            dist_mat (np.ndarray): matrix of pairwise distances between all
                nodes

        Returns:
            The minimum pairwise distance.
        """
        # subselect distance matrix to get all entries corresponding to
        # pairs with one elt in C1 and one elt in C2
        eltwise_dists = dist_mat[np.ix_(list(C1), list(C2))]
        return np.min(eltwise_dists)

    def cluster_distance_matrix(partition: list) -> np.ndarray:
        """
        Given a partition, computes the pairwise distance between pairs of
        clusters according to cluster_distance. Only computes distance between
        "big" clusters and "small" clusters.

        Args:
            partition (list): the partition, as a list of sets or list of lists

        Returns:
            Matrix of pairwise distances between clusters. np.nan entries
            signfiy a pair of clusters was not compared.
        """
        partition.sort(key=len, reverse=True)
        cluster_dist_mat = np.full(
            shape=(len(partition), len(partition)), fill_value=np.nan
        )
        # the small clusters have index max_k to len(partition)
        for jj in range(max_k, len(partition)):
            for ii in range(0, jj):
                # find minimum pairwise distance between any pair of elements
                # in the cluster
                cluster_dist_mat[ii, jj] = cluster_distance(
                    partition[ii], partition[jj]
                )
        return cluster_dist_mat

    while len(max_k_partition) > max_k:
        cluster_dist_mat = cluster_distance_matrix(max_k_partition)
        # find indices of clusters that have minimal pairwise distance
        ii_star, jj_star = np.unravel_index(
            np.nanargmin(cluster_dist_mat), cluster_dist_mat.shape
        )
        # merge the smaller cluster into the larger cluster
        C_ii_star = max_k_partition[ii_star].union(max_k_partition[jj_star])
        # need to delete jj_star first in order to avoid changing the index of
        # ii_star, because ii_star <jj_star
        assert ii_star < jj_star
        del max_k_partition[jj_star]
        del max_k_partition[ii_star]

        max_k_partition.append(C_ii_star)
        max_k_partition.sort(key=len, reverse=True)

    if min([len(cluster) for cluster in max_k_partition]) == 1:
        warnings.warn(
            "Warning: PEF partition produced at least one cluster with size 1."
        )

    # convert partition to dict form
    partition_dict = dict()
    for idx, cluster in enumerate(max_k_partition):
        partition_dict[idx] = list(cluster)

    return partition_dict
