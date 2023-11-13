# Relevant graph operations, metrics, and data generation
import networkx as nx
import numpy as np
import itertools
import cdt
from graphical_models import rand, GaussDAG
import causaldag as cd
import pandas as pd
import os
from sklearn.metrics import roc_curve

# Helper function to convert an adjacency matrix into an edge list
def adj_to_edge(adj, nodes,ignore_weights=False):
    edges = []
    for (row,col) in itertools.product(np.arange(adj.shape[0]), np.arange(adj.shape[1])):
        if adj[row,col] != 0:
            if ignore_weights:
                edges.append((nodes[row], nodes[col]))
            else:
                edges.append((nodes[row], nodes[col], {'weight' :adj[row,col]}))
    return edges

# Helper function to convert an adjacency matrix into a Networkx Digraph
def adj_to_dag(adj, all_nodes,fixed_edges=None):
    dag = nx.DiGraph()
    dag.add_nodes_from(all_nodes)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if np.abs(adj[i,j]) > 0:
                dag.add_edge(all_nodes[i], all_nodes[j], weight=np.abs(adj[i,j]))
    if fixed_edges:
        dag.add_edges_from(fixed_edges)
    return dag

# Helper function to convert a list of edges into an adjacency matrix
def edge_to_adj(edges, all_nodes):
    adj_mat = np.zeros((len(all_nodes), len(all_nodes)))
    for e in edges:
        start = all_nodes.index(e[0])
        end = all_nodes.index(e[1])
        adj_mat[start,end] = 1
    return adj_mat

# Helper function to convert a list of edges into a Networkx Digraph
def edge_to_dag(edges):
    dag = nx.DiGraph()
    dag.add_edges_from(edges)
    return dag

# Calculate the true positive rate and false positive rate between a true graph and predicted graph 
def tpr_fpr_score(y_true, y_pred):
    if type(y_pred) == nx.DiGraph:
        y_pred = nx.adjacency_matrix(y_pred)
        y_pred = y_pred.todense()
    if type(y_true) == nx.DiGraph:
        y_true = nx.adjacency_matrix(y_true)
        y_true=y_true.todense()
    y_pred = np.array(y_pred != 0, dtype=int).flatten()
    y_true = np.array(y_true != 0, dtype=int).flatten()
    fpr, tpr, _= roc_curve(y_true.flatten(), y_pred.flatten())
    return tpr[1], fpr[1]

# Helper function to print the SHD, SID, AUC for a set of algorithms and networks
# Also handles averaging over several sets of networks (e.g the random comparison averages over 30 different generated graphs)
# Default turn off sid, since it is computationally expensive
def get_scores(alg_names, networks, ground_truth, get_sid=False):
    for name, net in zip(alg_names, networks):
        if type(net) == list and type(ground_truth) == list:
            shd = 0
            sid = 0
            auc = 0
            tpr_fpr = [0,0]
            for n,g in zip(net, ground_truth):
                shd += cdt.metrics.SHD(g, n, False)
                sid += cdt.metrics.SID(g, n) if get_sid else 0
                if name!='NULL':
                    tpr_fpr += tpr_fpr_score(g,n)
                auc +=  cdt.metrics.precision_recall(g, n)[0]
            print("{} SHD: {} SID: {} AUC: {} TPR: {}".format(name, shd/len(net), sid/len(net), auc/len(net), tpr_fpr[0]/len(net)))
        elif type(net) != list and type(ground_truth) == list:
            shd = 0
            sid = 0
            auc = 0
            for g in ground_truth:
                shd += cdt.metrics.SHD(g, net, False)
                sid +=cdt.metrics.SID(g, net) if get_sid else 0
                auc +=  cdt.metrics.precision_recall(g, net)[0]
                if name!='NULL':
                    tpr_fpr += tpr_fpr_score(g,n)
            print("{} SHD: {} SID: {} AUC: {} TPR: {}".format(name, shd/len(ground_truth), sid/len(ground_truth), auc/len(ground_truth), tpr_fpr[0]/len(ground_truth)))
        else:
            shd = cdt.metrics.SHD(ground_truth, net, False)
            sid = cdt.metrics.SID(ground_truth, net) if get_sid else 0
            auc, pr = cdt.metrics.precision_recall(ground_truth, net)
            if name!='NULL':
                tpr_fpr = tpr_fpr_score(ground_truth, net)
            else :
                tpr_fpr= [0,0]
            print("{} SHD: {} SID: {} AUC: {}, TPR,FPR: {}".format(name, shd, sid, auc, tpr_fpr))
        return shd, sid, auc, tpr_fpr 

# Create a random gaussian DAG and correposning observational and interventional dataset.
# Save a data.csv file containing the observational and interventional data samples and target vector
# Save a ground.txt file containing the edge list of the generated graph 
def get_random_graph_data(graph_type, n, nsamples, iv_samples, p, k, seed=42, save=False, outdir=None):
    if graph_type == 'erdos_renyi':
        random_graph_model = lambda nnodes: nx.erdos_renyi_graph(nnodes, p=p, seed=seed)
    elif graph_type == 'scale_free':
        random_graph_model = lambda nnodes: nx.barabasi_albert_graph(nnodes, m=k, seed=seed)
    elif graph_type == 'small_world':
        random_graph_model = lambda nnodes: nx.watts_strogatz_graph(nnodes, k=k, p=p, seed=seed)
    else:
        print("Unsupported random graph")
        return
    dag = rand.directed_random_graph(n, random_graph_model)
    nodes_inds = list(dag.nodes)
    bias = np.random.normal(0,1,size=len(nodes_inds))
    var = np.abs(np.random.normal(0,1,size=len(nodes_inds)))

    bn = GaussDAG(nodes= nodes_inds, arcs=dag.arcs, biases=bias,variances=var)
    data = bn.sample(nsamples)

    df = pd.DataFrame(data=data, columns=nodes_inds)
    df['target'] = np.zeros(data.shape[0])

    if iv_samples > 0:
        i_data = []
        for ind, i in enumerate(nodes_inds):
            samples = bn.sample_interventional(cd.Intervention({i: cd.ConstantIntervention(val=0)}), iv_samples)
            samples = pd.DataFrame(samples, columns=nodes_inds)
            samples['target'] = ind + 1
            i_data.append(samples)
        df_int = pd.concat(i_data)
        df = pd.concat([df, df_int])
    if save:
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        df.to_csv("{}/data.csv".format(outdir), index=False)
        with open("{}/ground.txt".format(outdir), "w") as f:
            for edge in dag.arcs:
                f.write(str(edge) +"\n")
    return (dag.arcs, nodes_inds, bias, var), df

# Get data set from a predefined graph
# Save a data.csv file containing the observational and interventional data samples and target vector
# Save a ground.txt file containing the edge list of the generated graph 
def get_data_from_graph(nodes, edges, nsamples, iv_samples, save=False, outdir=None):
    bias = np.random.normal(0,1,size=len(nodes))
    var = np.abs(np.random.normal(0,1,size=len(nodes)))
    bn = GaussDAG(nodes= nodes, arcs=edges, biases=bias,variances=var)
    data = bn.sample(nsamples)

    df = pd.DataFrame(data=data, columns=nodes)
    df['target'] = np.zeros(data.shape[0])

    if iv_samples > 0:
        i_data = []
        for ind, i in enumerate(nodes):
            samples = bn.sample_interventional(cd.Intervention({i: cd.ConstantIntervention(val=0)}), iv_samples)
            samples = pd.DataFrame(samples, columns=nodes)
            samples['target'] = ind + 1
            i_data.append(samples)
        df_int = pd.concat(i_data)
        df = pd.concat([df, df_int])
    if save:
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        df.to_csv("{}/data.csv".format(outdir), index=False)
        with open("{}/ground.txt".format(outdir), "w") as f:
            for edge in edges:
                f.write(str(edge) +"\n")
    return (edges, nodes, bias, var), df