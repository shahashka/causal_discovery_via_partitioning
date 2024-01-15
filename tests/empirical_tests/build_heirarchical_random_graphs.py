# Imports
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

import causaldag as cd
import pandas as pd
from graphical_models import rand, GaussDAG

from cd_v_partition.utils import edge_to_adj
from duplicate_functions import apply_causal_order

import pdb


def directed_heirarchical_graph(num_nodes):
    G = nx.DiGraph(
        nx.scale_free_graph(
            num_nodes,
            alpha=0.2,
            gamma=0.5,
            beta=0.3,
            delta_in=0.0,
            delta_out=0.0,
        )
    )
    # find and remove cycles
    G.remove_edges_from(nx.selfloop_edges(G))
    cycle_list = nx.find_cycle(G, orientation="original")
    while len(cycle_list) > 0:
        edge_data = cycle_list[-1]
        G.remove_edge(edge_data[0], edge_data[1])
        # nx.find_cycle will throw an error nx.exception.NetworkXNoCycle when all cycles have been removed
        try:
            cycle_list = nx.find_cycle(G, orientation="original")
        except:
            break
    return G


# adds support for random heirarchical networks generated via specific params in a scale_free graph
def duplicate_get_random_graph_data(
    graph_type, num_nodes, nsamples, iv_samples, p, k, seed=42, save=False, outdir=None
):
    """
    Create a random Gaussian DAG and corresponding observational and interventional dataset.
    Note that the generated topology with Networkx undirected, using graphical_models a causal ordering
    is imposed on this graph which makes it a DAG. Each node has a randomly sampled
    bias and variance from which Gaussian data is generated (children are sums of their parents)
    Save a data.csv file containing the observational and interventional data samples and target column
    Save a ground.txt file containing the edge list of the generated graph.
            Parameters:
                    graph_type (str): erdos_renyi, scale_free (Barabasi-Albert) or small_world (Watts-Strogatz)
                    num_nodes (int): number of nodes in the generated graph
                    nsamples (int): number of observational samples to generate
                    iv_samples (int) : number of interventional samples to generate
                    p (float): probability of edge creation (erdos_renyi) or rewiring (small_world)
                    k (int): number of edges to attach from a new node to existing nodes (scale_free) or number of nearest neighbors connected in ring (small_world)
                    seed (int): random seed
                    save (bool): flag to save the dataset (data.csv) and graph (ground.txt)
                    outdir (str): directory to save the data to if save is True
            Returns:
                    arcs (list of edges), nodes (list of node indices), bias (bias terms for Gaussian generative model),
                    var (variance terms for Gaussian generative model) , df (pandas DataFrame containing sampled observational, interventional data and target indices)
    """
    if graph_type == "erdos_renyi":
        random_graph_model = lambda nnodes: nx.erdos_renyi_graph(nnodes, p=p, seed=seed)
    elif graph_type == "scale_free":
        random_graph_model = lambda nnodes: nx.barabasi_albert_graph(
            nnodes, m=k, seed=seed
        )
    elif graph_type == "small_world":
        random_graph_model = lambda nnodes: nx.watts_strogatz_graph(
            nnodes, k=k, p=p, seed=seed
        )
    elif graph_type == "heirarchical":
        random_graph_model = lambda nnodes: nx.Graph(
            nx.scale_free_graph(
                nnodes,
                alpha=0.2,
                gamma=0.5,
                beta=0.3,
                delta_in=0.0,
                delta_out=0.0,
                seed=seed,
            ).to_undirected()
        )
    else:
        print("Unsupported random graph")
        return

    dag = rand.directed_random_graph(num_nodes, random_graph_model)
    nodes_inds = list(dag.nodes)
    bias = np.random.normal(0, 1, size=len(nodes_inds))
    var = np.abs(np.random.normal(0, 1, size=len(nodes_inds)))

    # G_star_adj = edge_to_adj(list(dag.arcs), nodes=nodes_inds)
    # G_star_graph = nx.from_numpy_array(G_star_adj, create_using=nx.DiGraph)
    # plt.figure()
    # nx.draw(G_star_graph)
    # plt.show()
    # pdb.set_trace()

    bn = GaussDAG(nodes=nodes_inds, arcs=dag.arcs, biases=bias, variances=var)
    data = bn.sample(nsamples)

    df = pd.DataFrame(data=data, columns=nodes_inds)
    df["target"] = np.zeros(data.shape[0])

    if iv_samples > 0:
        i_data = []
        for ind, i in enumerate(nodes_inds):
            samples = bn.sample_interventional(
                cd.Intervention({i: cd.ConstantIntervention(val=0)}), iv_samples
            )
            samples = pd.DataFrame(samples, columns=nodes_inds)
            samples["target"] = ind + 1
            i_data.append(samples)
        df_int = pd.concat(i_data)
        df = pd.concat([df, df_int])
    if save:
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        df.to_csv("{}/data.csv".format(outdir), index=False)
        with open("{}/ground.txt".format(outdir), "w") as f:
            for edge in dag.arcs:
                f.write(str(edge) + "\n")
    return (dag.arcs, nodes_inds, bias, var), df


# for m in [1, 2, 5, 10, 18]:
#     G = nx.barabasi_albert_graph(n, m)
#     plt.figure()
#     nx.draw(G)
#     plt.show()
#     pdb.set_trace()

# n = 50
# alpha = 0.2
# gamma = 0.5
# beta = 1 - alpha - gamma
# delta_in = 0.0
# delta_out = 0.0

# G = nx.scale_free_graph(
#     n, alpha=alpha, beta=beta, gamma=gamma, delta_in=delta_in, delta_out=delta_out
# )
# plt.figure()
# nx.draw(G)
# plt.show()
# pdb.set_trace()
