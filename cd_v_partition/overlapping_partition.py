from __future__ import annotations

import subprocess
import networkx as nx
from pathlib import Path

import numpy as np
import pandas as pd

import pdb


def expansive_causal_partition(adj_mat: np.ndarray, partition: dict):
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


def modularity_partition(
    adj_mat: np.ndarray, resolution: int = 1, cutoff: int = 1, best_n=None
):
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
    print("ENTERED FUNCTION")
    pdb.set_trace()
    G = nx.from_numpy_array(adj_mat)
    community_lists = nx.community.greedy_modularity_communities(
        G, cutoff=cutoff, best_n=best_n
    )

    partition = dict()
    for idx, c in enumerate(community_lists):
        partition[idx] = list(c)
    return partition


def heirarchical_partition(adj_mat: np.ndarray, max_community_size: float = 0.5):
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


def rand_edge_cover_partition(adj_mat: np.ndarray, partition: dict):
    """Creates a random edge covering partition from an initial hard partition.

    Randomly chooses cut edges and randomly assigns endpoints to communities. Recursively
    adds any shared endpoints to the same community
    Args:
        adj_mat (np.ndarray): Adjacency matrix for the graph
        partition (dict): the estimated partition as a dictionary {comm_id : [nodes]}

    Returns:
        dict: the overlapping partition as a dictionary {comm_id : [nodes]}
    """
    graph = nx.from_numpy_array(adj_mat)

    def edge_coverage_helper(i, j, comm, cut_edges, node_to_comm):
        node_to_comm[i] = comm
        node_to_comm[j] = comm
        cut_edges.remove((i, j))

        # Any other edges that share the same endpoint must be in the same community
        # E.g. if edges (1,2) and (2,3) are cut then nodes 1,2,3 must all be in the
        # same community to ensure edge coverage
        for edge in cut_edges:
            if i in edge or j in edge:
                edge_coverage_helper(edge[0], edge[1], comm, cut_edges, node_to_comm)
        return node_to_comm, cut_edges

    node_to_comm = dict()
    for comm_id, comm in partition.items():
        for node in comm:
            node_to_comm[node] = comm_id
    cut_edges = []
    for edge in graph.edges():
        if node_to_comm[edge[0]] != node_to_comm[edge[1]]:
            cut_edges.append(edge)

    # Randomly choose a cut edge until all edges are covered
    while len(cut_edges) > 0:
        edge_ind = np.random.choice(np.arange(len(cut_edges)))
        i = cut_edges[edge_ind][0]
        j = cut_edges[edge_ind][1]

        # Randomly choose an endpoint and associated community to start
        # putting all endpoints into.
        comm = np.random.choice([node_to_comm[i], node_to_comm[j]])
        node_to_comm, cut_edges = edge_coverage_helper(
            i, j, comm, cut_edges, node_to_comm
        )

    # Update the hard partition
    for n, c in node_to_comm.items():
        if n not in partition[c]:
            partition[c] += [n]
    return partition


def oslom_algorithm(
    nodes,
    dat_file: Path | str,
    oslom_dir: Path | str,
    structure_type: str | None = "dag",
) -> dict:
    """
    Overlapping partitioning methods which take an input graph (superstructure) and partition
    nodes according to an objective overlapping nodes ideally render the partitions conditionally
    independent.

    Args:
        nodes (): ...
        data_dir (Path | str): The directory containing the *.dat file which holds the edges of the
            structure to partition.
        oslom_dir (Path | str): The directory containing the OSLOM binary
        structure_type (str | None): Specify the structure type as either the 'dag',  'superstructure',
            or 'superstructure_weighted'. If weighted then weights in the *.dat are used by OSLOM.
            Defaults to 'dag'.

    Returns:
        dict: The estimated partition as a dictionary {comm_id : [nodes]}
    """
    # Run the OSLOM code externally
    structure_type = structure_type if structure_type is not None else "dag"
    weight_flag = "-w" if "weight" in structure_type else "-uw"
    subprocess.run(
        [
            "{}/oslom_undir".format(oslom_dir),
            "-f",
            dat_file,
            "{}".format(weight_flag),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    # Read the output partition file and return the partition as a dictionary
    partition_file = "{}_oslo_files/tp".format(dat_file, structure_type)
    with open(partition_file, "rb") as f:
        lines = f.readlines()
    lines = lines[1::2]
    lines = [[int(node) for node in l.split()] for l in lines]
    partition = dict(zip(np.arange(len(lines)), lines))
    homeless_nodes = list(nodes)
    for part in lines:
        for n in part:
            if n in homeless_nodes:
                homeless_nodes.remove(n)

    if len(homeless_nodes) > 0:
        partition[len(lines)] = homeless_nodes
    return partition


def partition_problem(partition: dict, structure: np.ndarray, data: pd.DataFrame):
    """Split the graph structure and dataset according to the given graph partition

    Args:
        partition (dict): the partition as a dictionary {comm_id : [nodes]}
        structure (np.ndarray): the adjacency matrix for the initial structure
        data (pd.DataFrame): the dataset, columns correspond to nodes in the graph

    Returns:
        list: a list of tuples holding the sub structure and data subsets for each partition
    """
    sub_problems = []
    for _, sub_nodes in partition.items():
        sub_structure = structure[sub_nodes][:, sub_nodes]
        sub_nodes = list(sub_nodes)
        sub_nodes.append(-1)  # add 'target' vector at the end of dataframe
        sub_data = data.iloc[:, sub_nodes]
        sub_problems.append((sub_structure, sub_data))
    return sub_problems
