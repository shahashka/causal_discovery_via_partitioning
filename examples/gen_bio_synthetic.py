import tuner
import networkx as nx
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from cd_v_partition.overlapping_partition import oslom_algorithm
from cd_v_partition.vis_partition import create_partition_plot
from cd_v_partition.utils import evaluate_partition, get_data_from_graph
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
import itertools
from tqdm import tqdm
# Impose a causal ordering according to degree distribution, return a directed graph
def apply_causal_order(undirected_graph):
    deg_dist = np.array(list(undirected_graph.degree()), dtype=int)[:, 1]
    num_nodes = len(deg_dist)
    normalize = np.sum(np.array(list(undirected_graph.degree()), dtype=int)[:, 1])
    prob = [deg_dist[i] / normalize for i in np.arange(num_nodes)]
    causal_order = list(
        np.random.choice(np.arange(num_nodes), size=num_nodes, p=prob, replace=False)
    )

    undirected_edges = undirected_graph.edges()
    directed_edges = []
    for e in undirected_edges:
        self_loop = e[0]==e[1]
        if not self_loop:
            if causal_order.index(e[0]) > causal_order.index(e[1]):
                directed_edges.append(e[::-1])
            else:
                directed_edges.append(e)
    directed_graph = nx.DiGraph()
    directed_graph.add_nodes_from(undirected_graph.nodes())
    directed_graph.add_edges_from(directed_edges)
    return directed_graph


def generate_networks():
    data = pd.read_csv("./datasets/bionetworks/ecoli/ecoli_hiTRN.csv", header=0)
    edges = data[["TF", "gene"]].to_numpy()
    G = nx.Graph()
    G.add_edges_from(edges)
    G = nx.convert_node_labels_to_integers(G)
    print(tuner.metric_signature(G))

    # G = model.gen_topology(nodes=100) # for debugging

    start = time.time()
    population = tuner.replicate(G)
    tuning = time.time() - start
    print("Time for tuning {}".format(tuning))  # About 30 min for 1000 node network

    start = time.time()
    for i in range(30):
        copy = tuner.duplicate(population[0])
        copy = nx.convert_node_labels_to_integers(copy)
        copy_dir = apply_causal_order(copy)
        mat = nx.adjacency_matrix(
            copy_dir, nodelist=np.arange(len(copy_dir.nodes()))
        ).todense()
        nx.draw_spring(copy_dir, with_labels=True)
        plt.savefig(
            "./datasets/bionetworks/ecoli/synthetic_copies/net_{}.png".format(i)
        )
        plt.clf()
        np.savetxt(
            "./datasets/bionetworks/ecoli/synthetic_copies/net_{}.txt".format(i), mat
        )
    gen_time = time.time() - start

    print("Generation time {}".format(gen_time))

def hierarchical_cluster():
    G_adj = np.loadtxt("./datasets/bionetworks/ecoli/synthetic_copies/net_0.txt")
    G_dir = nx.from_numpy_array(G_adj, create_using=nx.DiGraph)
    G_dir= nx.convert_node_labels_to_integers(G_dir)
    df = get_data_from_graph(np.arange(len(G_dir.nodes)), list(G_dir.edges), nsamples=int(1e3), iv_samples=0, outdir=None)[-1]
    df = df.drop(columns=["target"]).transpose()
    print(df.shape)
    print("Start scipy clustering")
    Z = linkage(df.to_numpy())
    print(Z)
    print("Start sklearn clustering")
    labels = AgglomerativeClustering().fit_predict(df)
    keys = list(set(labels))
    values = [list(np.argwhere(labels==k).flatten()) for k in keys]
    partition = dict(zip(keys, values))
    print(partition)
    
    create_partition_plot(
        G_dir,
        np.arange(len(G_dir.nodes())),
        partition,
        "./Proximity-OSN-Model/hierachy_net.png",
    )
def analyze_net():
    data = pd.read_csv("./datasets/bionetworks/ecoli/ecoli_hiTRN.csv", header=0)
    edges = data[["TF", "gene"]].values.tolist()
        
    G_dir = nx.DiGraph()
    G_dir.add_edges_from(edges)
    G_dir = nx.convert_node_labels_to_integers(G_dir)
    
    G_undir =nx.Graph()
    G_undir.add_edges_from(edges)
    G_undir = nx.convert_node_labels_to_integers(G_undir)

    graph_for_vis = apply_causal_order(G_undir)
    # metrics = tuner.metric_signature(G)
    # print(metrics)
    # plt.plot(metrics["deg_dist"][::-1], marker='o')

    # G_adj = np.loadtxt("./datasets/bionetworks/ecoli/synthetic_copies/net_0.txt")
    # G = nx.Graph(G_adj)
    # metrics = tuner.metric_signature(G)
    # print(metrics)
    # plt.plot(metrics["deg_dist"][::-1], marker='o')
    # plt.savefig("./Proximity-OSN-Model/analyze.png")
    # plt.clf()

    #G_dir = nx.DiGraph(G_adj)
    _, ax = plt.subplots(figsize=(10, 10))
    node_size = 300

    def topo_pos(G):
        """Display in topological order, with simple offsetting for legibility"""
        pos_dict = {}
        for i, node_list in enumerate(nx.topological_generations(G)):
            x_offset = node_size*len(node_list) / 2
            y_offset = node_size
            for j, name in enumerate(node_list):
                pos_dict[name] = np.array([ node_size*j - x_offset,  -i * y_offset])

        return pos_dict

    # Same example data as top answer, but directed
    pos = topo_pos(graph_for_vis)

    nx.draw(G_dir, pos)

    # nx.draw_spring(G_dir)
    plt.savefig("./Proximity-OSN-Model/net_real.png")

    # df_edges = pd.DataFrame(list(G_dir.edges))
    # df_edges.to_csv(
    #     "./Proximity-OSN-Model/edges_dag.dat", sep="\t", header=None, index=None
    # )

    nodes = list(G_dir.nodes())
    # oslom_partition = oslom_algorithm(
    #     nodes, "./Proximity-OSN-Model/edges_dag.dat", "./OSLOM2/"
    # )
    G_adj = np.loadtxt("./datasets/bionetworks/ecoli/synthetic_copies/net_0.txt")
    G_adj = G_adj[0:10][:,0:10]
    G_dir = nx.from_numpy_array(G_adj, create_using=nx.DiGraph)
    gv_partition = nx.community.girvan_newman(G_dir)
    print("finished partitioning")
    i =0
    for communities in gv_partition:
        print(i)
        if max([len(com) for com in communities]) < (0.5 * G_adj.shape[0]):
            ith_partition = dict()
            for idx, c in enumerate(communities):
                ith_partition[idx] = list(c)
            break
        i += 1
        break
            
    print(ith_partition)
    # Evalute the partition
    create_partition_plot(
        G_dir,
        nodes,
        ith_partition,
        "./Proximity-OSN-Model/GN_net.png"
    )

    # evaluate_partition(gv_partition_dict, G_dir, nodes,)


analyze_net()
