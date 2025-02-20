# Run superstructure creation, partition, local discovery and screening
# for a base case network with assumed community structure
from cd_v_partition.utils import (
    get_random_graph_data,
    get_data_from_graph,
    evaluate_partition,
    delta_causality,
)
from cd_v_partition.causal_discovery import pc, weight_colliders, sp_gies
from cd_v_partition.overlapping_partition import oslom_algorithm, partition_problem
from cd_v_partition.vis_partition import create_partition_plot
from cd_v_partition.fusion import fusion

import networkx as nx
import numpy as np
import pandas as pd
import argparse
import itertools
import functools
import time

from concurrent.futures import ProcessPoolExecutor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--create",
        action="store_true",
        help="Flag to toggle base case dataset creation",
    )
    return parser.parse_args()


def run_base_case(algorithm, structure_type, nthreads, data_dir):
    """Run a partitioning algorithm on a base case example. Specify whether to partition the DAG, superstructure or
    weighted superstructure

    Args:
        algorithm (str): currently only supports 'oslom'
        structure_type (str): 'dag', 'superstructure', or 'superstructre_weighted'
        data_dir (str): folder where the dataset, adjacency matrix and edge list files are stored (see create_base_case for details )
    """
    if algorithm == "oslom":
        # Load information from data directory
        df = pd.read_csv("{}/data.csv".format(data_dir), header=0)
        adj = pd.read_csv(
            "{}/{}.csv".format(data_dir, structure_type.split("_")[0]), header=0
        )
        nodes = adj.columns.to_numpy(dtype=int)
        G = nx.DiGraph(adj.to_numpy())

        # Run OSLOM using the correct edge.dat file corersponding to the specified structure
        start_part = time.time()
        oslom_partition = oslom_algorithm(
            nodes, "{}/edges_{}.dat".format(data_dir, structure_type), "./OSLOM2/"
        )
        part_time = time.time() - start_part
        num_partitions = len(oslom_partition)

        # Evalute the partition
        evaluate_partition(oslom_partition, G, nodes)
        create_partition_plot(
            G,
            nodes,
            oslom_partition,
            "{}/oslom_{}.png".format(data_dir, structure_type),
        )

        # Partition problem
        print("Running local causal discovery...")
        start_part_problem = time.time()
        subproblems = partition_problem(oslom_partition, adj.to_numpy(), df)
        part_problem_time = time.time() - start_part_problem

        # Launch processes and run locally
        func_partial = functools.partial(_local_structure_learn)
        results = []
        chunksize = max(1, num_partitions // nthreads)
        print("Launching processes")

        start_local_learn = time.time()
        with ProcessPoolExecutor(max_workers=nthreads) as executor:
            for result in executor.map(func_partial, subproblems, chunksize=chunksize):
                results.append(result)
        local_learn_time = time.time() - start_local_learn

        # Merge globally
        start_fusion = time.time()
        data = df.to_numpy()
        est_graph_partition = fusion(oslom_partition, results, data)
        fusion_time = time.time() - start_fusion
        est_graph_partition = nx.adjacency_matrix(
            est_graph_partition, nodelist=np.arange(len(nodes))
        ).todense()

        # Call serial method
        start_serial = time.time()
        est_graph_serial = _local_structure_learn((adj.to_numpy(), df))
        serial_time = time.time() - start_serial

        # Compare causal metrics
        d_scores = delta_causality(
            est_graph_serial, est_graph_partition, adj.to_numpy()
        )
        print(
            "Delta causality: SHD {}, SID {}, AUC {}, TPR {} , FPR {}".format(
                d_scores[0], d_scores[1], d_scores[2], d_scores[3], d_scores[4]
            )
        )

        print(
            "Time to solution (s): CD_serial {} , CD_partition {}".format(
                serial_time,
                part_time + part_problem_time + local_learn_time + fusion_time,
            )
        )

    else:
        NotImplementedError()


def create_base_case_net(
    graph_type, n, p, k, ncommunities, alpha, collider_weight, nsamples, outdir
):
    """Create a base case network to use for evaluation. This network is comprised of a base random graph that is
       tiled to construct a network with community structure.  Also generates a superstructure
       of this network by generating data from the network and running the PC algorithm with the given alpha value.

       Saves files to the output directory corresponding to the tiled adjacency matrix (*_tiled.csv), and edge lists
       for the generated DAG, superstructure and weighted superstructure.

    Args:
        graph_type (str): 'erdos_renyi', 'scale_free', or 'small_world', specify the type of random graph for each community
        n (int): number of nodes per community
        p (float): probability of connection (erdos_renyi) or rewiring (small_free)
        k (int): number of edges to attach from a new node to existing nodes (scale_free) or number of nearest neighbors connected in ring (small_world)
        ntiles (int): number of communities
        alpha (float): siginficance threshold for the PC algorithm
        collider_weight (int): weight edges in a collider set by this value
        nsamples (int): number of observational samples to generate for the entire graph
        outdir (str): save the output files here

    Returns:
        (nx.DiGraph, pandas DataFrame): the final directed network and the dataset of sampled observational values
    """

    # Create a random 'base' network
    (arcs, nodes, _, _), _ = get_random_graph_data(
        graph_type, num_nodes=n, nsamples=0, iv_samples=0, p=p, m=k, save=False
    )
    net = nx.DiGraph()
    net.add_edges_from(arcs)
    net.add_nodes_from(nodes)

    # Create a tiled network with community structure, save to dataset directory
    print("Creating tiled net")
    nodes = np.arange(n * ncommunities)
    tiled_net = _construct_tiling(net, num_tiles=ncommunities)
    df_edges = pd.DataFrame(list(tiled_net.edges))
    df_edges.to_csv(
        "{}/edges_dag.dat".format(outdir), sep="\t", header=None, index=None
    )
    adj_mat = nx.adjacency_matrix(tiled_net, nodelist=nodes).toarray()
    df = pd.DataFrame(adj_mat)
    df.to_csv("{}/dag.csv".format(outdir), index=False)

    print("Generate data")
    # Generate data from the tiled network
    (arcs, nodes, _, _), df = get_data_from_graph(
        nodes,
        list(tiled_net.edges()),
        nsamples=nsamples,
        iv_samples=0,
        save=True,
        outdir=outdir,
    )

    # Use the data to generate a superstructure using the pc algorithm
    superstructure, p_values = pc(df, alpha=alpha, outdir=outdir)
    superstructure_w = weight_colliders(superstructure, weight=collider_weight)
    weights = superstructure_w  # np.multiply(superstructure_w, np.abs(p_values)) # p_values are negative? pc algorithm pMax matrix is confusing, for now take absolute value
    superstructure_net = nx.from_numpy_array(weights, create_using=nx.Graph)

    # Save the super structure adjacency matrix, edges and weighted superstructure edges
    df_adj = pd.DataFrame(superstructure)
    df_adj.to_csv("{}/superstructure.csv".format(outdir), index=None)

    df_edges = pd.DataFrame(list(superstructure_net.edges))
    df_edges.to_csv(
        "{}/edges_superstructure.dat".format(outdir), sep="\t", header=None, index=None
    )

    df_edges["weight"] = [
        superstructure_net.get_edge_data(u, v)["weight"]
        for (u, v) in superstructure_net.edges
    ]
    df_edges.to_csv(
        "{}/edges_superstructure_weighted.dat".format(outdir),
        sep="\t",
        header=None,
        index=None,
    )

    # Checks for sanity
    print("Number of colliders: {}".format(_count_colliders(tiled_net)))
    _check_superstructure(
        superstructure, nx.adjacency_matrix(tiled_net, nodelist=np.arange(len(nodes)))
    )

    return superstructure_net, df


def _local_structure_learn(subproblem):
    """Call causal discovery algorithm on subproblem. Right now uses SP-GIES

    Args:
        subproblem (tuple np.ndarray, pandas DataFrame): the substructure adjacency matrix and corresponding data

    Returns:
        np.ndarray: Estimated DAG adjacency matrix for the subproblem
    """
    skel, data = subproblem
    adj_mat = sp_gies(data, outdir=None, skel=skel, use_pc=True)
    return adj_mat


def _construct_tiling(net, num_tiles):
    """Helper function to construct the tiled/community network from a base net.
    The tiling is done so that nodes in one community are preferentially attached
    (proportional to degree) to nodes in other communities.

    Args:
        net (nx.DiGraph): the directed graph for one community
        num_tiles (int): the number of tiles or communities to create

    Returns:
        nx.DiGraph: the final directed graph with community structure
    """
    if num_tiles == 1:
        return net
    num_nodes = len(list(net.nodes()))
    degree_sequence = sorted((d for _, d in net.in_degree()), reverse=True)
    dmax = max(degree_sequence)
    tiles = [net for _ in range(num_tiles)]

    # First add all communities as disjoint graphs
    tiled_graph = nx.disjoint_union_all(tiles)

    # Each node is preferentially attached to other nodes
    # The number of attached nodes is given by a probability distribution over
    # A = 1, 2 ... min(dmax,4) where the probability is equal to the in_degree=A/number of nodes
    # in the community
    A = np.min([dmax, 4])
    in_degree_a = [sum(np.array(degree_sequence) == a) for a in range(A)]
    leftover = num_nodes - sum(in_degree_a)
    in_degree_a[-1] += leftover
    probs = np.array(in_degree_a) / (num_nodes)

    # Add connections based on random choice over probability distribution
    for t in range(1, num_tiles):
        for i in range(num_nodes):
            node_label = t * num_nodes + i
            if len(list(tiled_graph.predecessors(node_label))) == 0:
                num_connected = np.random.choice(np.arange(A), size=1, p=probs)
                dest = np.random.choice(np.arange(t * num_nodes), size=num_connected)
                connections = [(node_label, d) for d in dest]
                tiled_graph.add_edges_from(connections)
    return tiled_graph


def _count_colliders(G):
    """Helper function to count the number of colliders in the graph G. For every
    triple x-y-z determine if the edges are in a collider orientation. This counts
    as one collider set.

    Args:
        G (nx.DiGraph): Directed graph

    Returns:
        int: number of collider sets in the graph
    """
    num_colliders = 0
    non_colliders = 0

    # Find all triples x-y-z
    for x, y, z in itertools.permutations(G.nodes, 3):
        if G.has_edge(x, y) and G.has_edge(z, y):
            num_colliders += 1
        elif G.has_edge(x, y) and G.has_edge(y, z):
            non_colliders += 1
        elif G.has_edge(y, x) and G.has_edge(z, y):
            non_colliders += 1
        elif G.has_edge(y, x) and G.has_edge(y, z):
            non_colliders += 1
    return num_colliders, num_colliders / (num_colliders + non_colliders)


# Check that this is a superstructure
def _check_superstructure(S, G):
    """Make sure that S is a superstructure of G. This means all edges in G are constrained
       by S.


    Args:
        S (np.ndarray): adjacency matrix for the superstructure
        G (np.ndattay): adjacency matrix for the DAG
    """
    num_wrong = 0
    for row in np.arange(S.shape[0]):
        for col in np.arange(S.shape[1]):
            if G[row, col] == 1:
                if S[row, col] == 0:
                    num_wrong += 1
    print(
        "Number of missed edges in superstructure (which has {} edges) {} out of {} edges".format(
            np.sum(S > 0), num_wrong, np.sum(G > 0)
        )
    )


if __name__ == "__main__":
    args = get_args()
    if args.create:
        create_base_case_net(
            "scale_free",
            n=10,
            p=0.3,
            k=4,
            ncommunities=5,
            alpha=5e-1,
            collider_weight=10,
            nsamples=int(1e6),
            outdir="./datasets/base_case/",
        )
    st_types = ["superstructure", "superstructure_weighted", "dag"]
    for t in st_types:
        print(t)
        run_base_case("oslom", t, 2, "./datasets/base_case/")
