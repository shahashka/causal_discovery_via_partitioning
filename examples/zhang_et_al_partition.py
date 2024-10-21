import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import pdb


def zhang_et_al_partition(
    adj_mat: np.ndarray,
    data: pd.DataFrame = None,
    resolution: int = None,
    cutoff: int = None,
    best_n: int = None,
    plotting: bool = False,
):
    """Creates a partition according to the procedure outlined in section C of Zhang et al.

    First uses greedy modularity to create a disjoint partition, then adds the outer-boundary of each
    cluster to create a causal partition
    Args:
        adj_mat (np.ndarray): the adjacency matrix for the superstructure
        data (Any): unused parameter
        resolution (float): unused parameter
        cutoff (int):  unused parameter
        best_n (int):  unused parameter
        plotting (bool): whether to display plots to illustrate code
    Returns:
        dict: the causal partition as a dictionary {comm_id : [nodes]}
    """
    original_G = nx.from_numpy_array(adj_mat)
    # Define adjoint graph (or line graph)
    line_graph = nx.line_graph(original_G)

    def get_edge_mapping(G, L):
        """Returns a dictionary mapping edges in the line graph to the corresponding
        nodes and edges in the original graph.
        """

        line_graph_to_og_graph = {}
        for edge in L.edges():
            node1, node2 = edge
            original_edge1 = tuple(node1)
            original_edge2 = tuple(node2)
            original_node = set(original_edge1).intersection(set(original_edge2)).pop()
            line_graph_to_og_graph[edge] = (
                original_node,
                original_edge1,
                original_edge2,
            )

        return line_graph_to_og_graph

    # Make dictionaries for mapping node identities
    line_graph_idx_to_node_name = dict(
        zip(range(len(line_graph.nodes())), line_graph.nodes())
    )
    line_graph_node_name_to_idx = dict(
        zip(line_graph.nodes(), range(len(line_graph.nodes())))
    )

    # Get Laplacian matrix of adjoint
    # L is symmetric so left and right eigenvectors are identical and real-valued
    # With numpy
    laplacian_matrix = nx.laplacian_matrix(line_graph).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
    try:
        # For large matrices, running into numerical issues, so check imag. this way.
        assert np.all(np.isclose(np.imag(eigenvalues), 0))
        assert np.all(np.isclose(np.imag(eigenvectors), 0))
    except:
        raise Exception(
            "zhang_et_al_partition: Laplacian matrix should be symmetric, but complex eigenvectors/eigenvalues produced."
        )

    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    # Sort eigenvalues and eigenvectors by ascending eigenvalue
    ascending_idxs = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[ascending_idxs]
    eigenvectors = eigenvectors[:, ascending_idxs]

    # Select Fiedler vector (eigenvector of second smallest eigenvalue)
    v2 = eigenvectors[:, 1]

    # Optional: plot
    if plotting:
        nx.draw_networkx(line_graph, node_color=v2 > 0, with_labels=False)
        plt.title("Line graph \n nodes colored by sign of second eigenvector")
        plt.show()

    # Define communities
    A_idxs = np.where(v2 >= 0)[0].tolist()
    B_idxs = np.where(v2 < 0)[0].tolist()
    # Convert indices to sets of nodes
    line_graph_A = set([line_graph_idx_to_node_name[idx] for idx in A_idxs])
    line_graph_B = set([line_graph_idx_to_node_name[idx] for idx in B_idxs])
    # Find edge cut separating A and B
    C = [
        e
        for e in line_graph.edges()
        if (e[0] in line_graph_A and e[1] in line_graph_B)
        or (e[1] in line_graph_A and e[0] in line_graph_B)
    ]

    # Optional: plot
    if plotting:
        node_color_map = []
        edge_color_map = []
        for node in line_graph.nodes():
            if node in line_graph_A:
                node_color_map.append("blue")
            else:
                node_color_map.append("red")
        for edge in line_graph.edges():
            if edge in C:
                edge_color_map.append("green")
            else:
                edge_color_map.append("red")
        nx.draw_networkx(
            line_graph,
            node_color=node_color_map,
            edge_color=edge_color_map,
            with_labels=True,
        )
        plt.title("Line graph with edge cut shown in green")
        plt.show()

    # Build vertex sets in original graph, according to Lemma 2 in Zhang et al.
    # Each edge in the line graph corresponds to a node and two edges in the original graph
    line_graph_to_og_graph = get_edge_mapping(original_G, line_graph)

    # For each edge in line graph, return list of nodes in original graph corresponding to edge data.
    def unpack_line_graph_edge_data(line_graph_edge_data):
        return [
            line_graph_edge_data[0],
            *line_graph_edge_data[1],
            *line_graph_edge_data[2],
        ]

    def flatten(xss):
        return [x for xs in xss for x in xs]

    M = set(
        flatten(
            [unpack_line_graph_edge_data(line_graph_to_og_graph[edge]) for edge in C]
        )
    )
    original_A = set(flatten(line_graph_A)).difference(M)
    original_B = set(flatten(line_graph_B)).difference(M)

    # Produce final overlapping vertex partition
    V1 = original_A.union(M)
    V2 = original_B.union(M)

    # Optional: plot
    if plotting:
        node_color_map = []
        for node in original_G.nodes():
            if node in original_A:
                node_color_map.append("blue")
            elif node in original_B:
                node_color_map.append("red")
            else:
                node_color_map.append("purple")
        nx.draw_networkx(
            original_G,
            node_color=node_color_map,
            with_labels=True,
        )
        # Custom legend
        red_patch = mpatches.Patch(color="red", label="V1 \ V2")
        blue_patch = mpatches.Patch(color="blue", label="V2 \ V1")
        purple_patch = mpatches.Patch(color="purple", label="Overlap")
        plt.legend(handles=[red_patch, blue_patch, purple_patch])
        plt.title("Final overlapping partition")
        plt.show()
    return {0: V1, 1: V2}


# Generate 2-community symmetric SBM
p = 0.5
q = 0.05
community_size = 10
G = nx.stochastic_block_model(
    sizes=[community_size, community_size], p=[[p, q], [q, p]]
)
while not nx.is_connected(G):
    G = nx.stochastic_block_model(
        sizes=[community_size, community_size], p=[[p, q], [q, p]]
    )
# remove self-loops: zhang method does not allow for self-loops.
G.remove_edges_from(nx.selfloop_edges(G))
nx.draw(G)
plt.show()
adj_mat = nx.to_numpy_array(G)

partition_dict = zhang_et_al_partition(adj_mat=adj_mat, plotting=True)
