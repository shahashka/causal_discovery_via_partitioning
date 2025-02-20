from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pylab
from matplotlib.patches import PathPatch
from netgraph import Graph, get_curved_edge_paths
from scipy.ndimage import gaussian_filter
from scipy.spatial.ckdtree import cKDTree
from scipy.spatial.distance import cdist


def create_partition_plot(
    G: nx.Graph | nx.DiGraph,
    nodes: list[str],
    partition: dict[int, list[int]],
    save_name: Path | str,
    ax=None,
    node_size=1,
    edge_width=2,
):
    """
    Create plot of overlapping partitions with patches.

    Args:
        G (nx.Graph | nx.DiGraph): Graph to plot.
        nodes (list[str]): List of node names.
        partition (dict[int, list[int]]): Partition.
        save_name (Path | str):

    Returns:
        ...
    """
    node_to_partition = dict(zip(nodes, [[] for _ in np.arange(len(nodes))]))
    for key, value in partition.items():
        for node in value:
            node_to_partition[node] += [key]
    pos, overlaps = _partition_layout(G, node_to_partition)

    if ax is None:
        _, ax = plt.subplots()
    cm = pylab.get_cmap("plasma")
    colors = []
    num_colors = len(partition)
    for i in range(num_colors):
        colors.append(
            cm(1.0 * i / num_colors)
        )  # color will now be an RGBA tuple

    color_map = dict(zip(np.arange(num_colors + 1), colors + ["gray"]))
    colors = dict(
        zip(
            nodes,
            [
                (
                    color_map[comm[0]]
                    if node not in overlaps
                    else color_map[num_colors]
                )
                for node, comm in node_to_partition.items()
            ],
        )
    )

    Graph(
        G,
        edge_width=edge_width,
        node_size=node_size,
        edge_color="black",
        node_layout=pos,
        node_color=colors,
        arrows=True,
        ax=ax,
    )

    for comm, nodes in partition.items():
        _create_patches(pos, ax, nodes, color_map[comm])
    plt.savefig(save_name)


# https://stackoverflow.com/questions/73265089/networkx-how-to-draw-bounding-area-containing-a-set-of-nodes
def _create_patches(node_positions, ax, subset, color):
    if len(subset) == 1:
        return
    # Using the nodes in the subset, construct the minimum spanning tree using
    # distance as the weight parameter.
    xy = np.array([node_positions[node] for node in subset])
    distances = cdist(xy, xy)
    h = nx.Graph()
    h.add_weighted_edges_from(
        [
            (subset[ii], subset[jj], distances[ii, jj])
            for ii, jj in itertools.combinations(range(len(subset)), 2)
        ]
    )
    h = nx.minimum_spanning_tree(h)

    # -------------------------------------------------------------------------
    # Compute an edge routing that avoids other nodes. Here I use
    # a modified version of the Fruchterman-Reingold algorithm to
    # place edge control points while avoiding the nodes.
    # Change the default origin and scale to make the canvas a bit
    # larger such that the curved edges can curve outside the bbox
    # covering the nodes.
    edge_paths = get_curved_edge_paths(
        list(h.edges),
        node_positions,
        k=0.25,
        origin=(-0.5, -0.5),
        scale=(2, 2),
    )

    # ------------------------------------------------------------------------
    # Use nearest neighbour interpolation to partition the canvas into 2
    # regions.

    xy1 = np.concatenate(list(edge_paths.values()), axis=0)
    z1 = np.ones(len(xy1))

    xy2 = np.array(
        [node_positions[node] for node in node_positions if node not in subset]
    )
    z2 = np.zeros(len(xy2))

    # Add a frame around the axes.
    # This reduces the desired mask in regions where there are no nearby point
    # from the exclusion list.
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xx = np.linspace(xmin, xmax, 100)
    yy = np.linspace(ymin, ymax, 100)

    left = np.c_[np.full_like(xx, xmin), yy]
    top = np.c_[xx, np.full_like(yy, ymax)]
    right = np.c_[np.full_like(xx, xmax), yy]
    bottom = np.c_[xx, np.full_like(yy, ymin)]

    xy3 = np.concatenate([left, top, right, bottom], axis=0)
    z3 = np.zeros(len(xy3))

    xy = np.concatenate([xy1, xy2, xy3], axis=0)
    z = np.concatenate([z1, z2, z3])
    tree = cKDTree(xy)
    xy_grid = np.meshgrid(xx, yy)
    _, indices = tree.query(np.reshape(xy_grid, (2, -1)).T, k=1)
    z_grid = z[indices].reshape(xy_grid[0].shape)

    # smooth output
    z_smooth = gaussian_filter(z_grid, 1.5)

    contour = ax.contour(
        xy_grid[0], xy_grid[1], z_smooth, np.array([0.9]), alpha=0
    )
    patch = PathPatch(
        contour.collections[0].get_paths()[0],
        facecolor=color,
        alpha=0.5,
        zorder=-1,
    )
    ax.add_patch(patch)


def _partition_layout(
    g: nx.Graph | nx.DiGraph, partition: dict[int, list[int]]
) -> tuple[dict[int, tuple[float, float]], list]:
    """
    Compute the layout for a modular graph.

    Args:
        g (nx.Graph | nx.DiGraph): Graph to plot.
        partition (dict[int, list[int]]): Graph partitions.

    Returns:
        Node positions.
    """

    pos_partitions = _position_partitions(g, partition, scale=3.0)
    pos_nodes = _position_nodes(g, partition, scale=1.0)
    pos = {node: pos_partitions[node] + pos_nodes[node] for node in g.nodes()}
    overlaps = _find_overlaps(partition)
    return pos, overlaps


def _position_partitions(g, partition, **kwargs):
    # create a weighted graph, in which each node corresponds to a partition,
    # and each edge weight to the number of edges between partitions
    between_partition_edges = _find_between_partition_edges(g, partition)

    partitions = set()
    for comm in partition.values():
        for c in comm:
            partitions.add(c)

    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(partitions)
    for (ci, cj), edges in between_partition_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for partitions
    pos_partitions = nx.spring_layout(
        hypergraph, k=10 * len(g.nodes()) / np.sqrt(len(partitions)), **kwargs
    )

    # set node positions to position of partition
    pos = dict()

    # (Nathaniel) NOTE: This looks like an error, since `partition` is
    # being over-written in each iteration. Is this intended?
    # Since this is just for plotting a sample visual, it's not a
    # drastic issue.
    for node, partition in partition.items():  # noqa: B020
        pos_c = np.mean([pos_partitions[c] for c in partition], axis=0)
        pos[node] = pos_c

    return pos


def _find_overlaps(partition):
    overlaps = []
    for node, comm in partition.items():
        if len(comm) > 1:
            overlaps.append(node)
    return overlaps


def _find_between_partition_edges(g, partition):
    edges = {}
    for ni, nj in g.edges():
        ci = partition[ni]
        cj = partition[nj]
        if len(set(ci).intersection(set(cj))) == 0:
            for j in cj:
                for i in ci:
                    try:
                        edges[(i, j)] += [(ni, nj)]
                    except KeyError:
                        edges[(i, j)] = [(ni, nj)]

    return edges


def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within partitions.
    """

    partitions = dict()
    for node, partition in partition.items():  # noqa: B020
        for c in partition:
            try:
                partitions[c] += [node]
            except KeyError:
                partitions[c] = [node]
    pos = dict()
    for _ci, nodes in partitions.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(
            subgraph, k=5 / np.sqrt(len(nodes)), **kwargs
        )
        pos.update(pos_subgraph)

    return pos
