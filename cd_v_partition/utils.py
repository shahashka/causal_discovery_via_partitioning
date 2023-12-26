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
import math

def adj_to_edge(adj, nodes, ignore_weights=False):
    '''
    Helper function to convert an adjacency matrix into an edge list. Optionally include weights so that
    the edge tuple is (i,j, weight)
            Parameters:
                    adj (np.ndarray): p x p adjacency matrix
                    nodes (list): a list of node names in order corresponding to rows/cols of adj
                    ignore_weights: flag to include weights, where the weight is the value in the adj
            Returns:
                    list of edges corresponding to nonzero values in adj
    '''
    edges = []
    for (row,col) in itertools.product(np.arange(adj.shape[0]), np.arange(adj.shape[1])):
        if adj[row,col] != 0:
            if ignore_weights:
                edges.append((nodes[row], nodes[col]))
            else:
                edges.append((nodes[row], nodes[col], {'weight' :adj[row,col]}))
    return edges


def adj_to_dag(adj, nodes):
    '''
    Helper function to convert an adjacency matrix into a Networkx DiGraph
            Parameters:
                    adj (np.ndarray): p x p adjacency matrix
                    nodes (list): a list of node names in order corresponding to rows/cols of adj
            Returns:
                    nx.DiGraph 
    '''
    dag = nx.DiGraph()
    dag.add_nodes_from(nodes)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if np.abs(adj[i,j]) > 0:
                dag.add_edge(nodes[i], nodes[j], weight=np.abs(adj[i,j]))
    return dag

def edge_to_adj(edges, nodes):
    '''
    Helper function to convert a list of edges into an adjacency matrix
            Parameters:
                    edges (list): list of (i,j) tuples corresponding to directed edges
                    nodes (list): a list of node names in order corresponding to rows/cols of adj
            Returns:
                    np.ndarray representing the adjacency matrix
    '''
    
    adj_mat = np.zeros((len(nodes), len(nodes)))
    for e in edges:
        start = nodes.index(e[0])
        end = nodes.index(e[1])
        adj_mat[start,end] = 1
    return adj_mat

def edge_to_dag(edges):
    '''
    Helper function to convert a list of edges into a Networkx DiGraph
            Parameters:
                    edges (list): list of (i,j) tuples corresponding to directed edges
            Returns:
                    nx.DiGraph 
    '''
    dag = nx.DiGraph()
    dag.add_edges_from(edges)
    return dag


def tpr_fpr_score(y_true, y_pred):
    '''
    Calculate the true positive rate and false positive scores between a true graph and predicted graph using
    sklearn.roc_curve. We choose the point correponding to the maximum threshold i.e if the adjacency matrix 
    only has 1s or 0s, the max threshold is 1 and the tpr corresponds to the number of correct 1s. 
    
            Parameters:
                    y_true (np.ndarray or nx.DiGraph): ground truth adjacency matrix or directed graph
                    y_pred (np.ndarray or nx.DiGraph): estimated adjancecy matrix or directed graph
            Returns:
                    (float, float) corresponding to true positive rate and false positive rate in ROC curve
                    at the maximum threshold value
    '''
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

def get_scores(alg_names, networks, ground_truth, get_sid=False):
    '''
    Calculate metrics Structural Hamming Distance (SHD), Structural Interventional Distance
    (SID), AUC, TPR,FPR for a set of algorithms and networks
    Also handles averaging over several sets of networks (e.g the random comparison averages over several different generated graphs)
    Default turn off sid, since it is computationally expensive
            Parameters:
                    alg_names (list of str): list of algorithm names
                    networks (list of np.ndarray or list of nx.DiGraph): list of estimated graphs corresponding to algorithm names
                    ground_truth (np.ndarray or nx.DiGraph): the true graph to compare to
                    get_sid (bool) : flag to calculate SID which is computationally expensive, returned SID is 0 if this is False
            Returns:
                    floats corresponding to SHD, SID, AUC, (TPR, FPR)
    '''
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
                # Precision/recall requires 0,1 array 
                if type(n) == np.ndarray:
                    g = g!=0
                    n = n!=0
                auc +=  cdt.metrics.precision_recall(g, n)[0]
            print("{} SHD: {} SID: {} AUC: {} TPR: {}".format(name, shd/len(net), sid/len(net), auc/len(net), tpr_fpr[0]/len(net)))
        elif type(net) != list and type(ground_truth) == list:
            shd = 0
            sid = 0
            auc = 0
            for g in ground_truth:
                shd += cdt.metrics.SHD(g, net, False)
                sid +=cdt.metrics.SID(g, net) if get_sid else 0
                if name!='NULL':
                    tpr_fpr += tpr_fpr_score(g,n)
                # Precision/recall requires 0,1 array 
                if type(net) == np.ndarray:
                    g = g!=0
                    net = net!=0
                auc +=  cdt.metrics.precision_recall(g, net)[0]
            print("{} SHD: {} SID: {} AUC: {} TPR: {}".format(name, shd/len(ground_truth), sid/len(ground_truth), auc/len(ground_truth), tpr_fpr[0]/len(ground_truth)))
        else:
            shd = cdt.metrics.SHD(ground_truth, net, False)
            sid = cdt.metrics.SID(ground_truth, net) if get_sid else 0
            auc=0
            if name!='NULL':
                tpr_fpr = tpr_fpr_score(ground_truth, net)
            else :
                tpr_fpr= [0,0]
            # Precision/recall requires 0,1 array 
            if type(net) == np.ndarray:
                    ground_truth = ground_truth!=0
                    net = net!=0
            auc, pr = cdt.metrics.precision_recall(ground_truth, net)

            print("{} SHD: {} SID: {} AUC: {}, TPR,FPR: {}".format(name, shd, sid, auc, tpr_fpr))
        return shd, sid, auc, tpr_fpr[0], tpr_fpr[1]


def get_random_graph_data(graph_type, num_nodes, nsamples, iv_samples, p, k, seed=42, save=False, outdir=None):
    '''
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
    '''
    if graph_type == 'erdos_renyi':
        random_graph_model = lambda nnodes: nx.erdos_renyi_graph(nnodes, p=p, seed=seed)
    elif graph_type == 'scale_free':
        random_graph_model = lambda nnodes: nx.barabasi_albert_graph(nnodes, m=k, seed=seed)
    elif graph_type == 'small_world':
        random_graph_model = lambda nnodes: nx.watts_strogatz_graph(nnodes, k=k, p=p, seed=seed)
    else:
        print("Unsupported random graph")
        return
    dag = rand.directed_random_graph(num_nodes, random_graph_model)
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

 
def get_data_from_graph(nodes, edges, nsamples, iv_samples, save=False, outdir=None):
    '''
    Get data set from a predefined graph using the Gaussian DAG generative model (same as get_random_graph_data)
    Save a data.csv file containing the observational and interventional data samples and target vector
    Save a ground.txt file containing the edge list of the generated graph
    Save a data.csv file containing the observational and interventional data samples and target column
    Save a ground.txt file containing the edge list of the generated graph.
            Parameters:
                    nodes (list): list of node names
                    edges (list of tuples): list of directed edge tuples (i,j) where i and j are in nodes
                    nsamples (int): number of observational samples to generate
                    iv_samples (int) : number of interventional samples to generate
                    save (bool): flag to save the dataset (data.csv) and graph (ground.txt)
                    outdir (str): directory to save the data to if save is True
            Returns:
                    edges (list of edges), nodes (list of node indices), bias (bias terms for Gaussian generative model),
                    var (variance terms for Gaussian generative model) , df (pandas DataFrame containing sampled observational, interventional data and target indices)
    '''
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


# Partitions is a dictionary with key,values <partition id>: <list of nodes in partition>
# From this paper https://arxiv.org/abs/0910.5072
# Ranges from 1 to -1. Positive values are better, 1 indicates fully connected graph 
def _modularity_overlapping(partitions, nodes, A):
    A = A.todense() # Sparse matrix indexing support is poor 
    def mod_cluster(part, nodes, A, S, D, n_edges, n_nodes):
        part_nodes = part[1]
        node_mod = np.zeros(len(part_nodes))
        for ind, i in enumerate(part_nodes):
            within_cluster = np.sum([A[i,j] for j in part_nodes if j!=i])
            intra_cluster = np.sum([A[i,j] for j in nodes if j not in part_nodes])
            d_i = D[i]
            s_i = S[i]
            node_mod[ind] += (within_cluster - intra_cluster)/(d_i*s_i) if d_i !=0 else 0
        return n_edges/(math.comb(n_nodes,2)*n_nodes) * sum(node_mod)
    
    K = len(partitions)
    # S is the number of clusters each node i belongs to
    S = np.zeros(len(nodes))
    
    # D is the degree of each node in the graph
    D = np.zeros(len(nodes))
    for i in nodes:
        S[i] = sum([1 for p in partitions.values() if i in p])
        D[i] = np.sum(A[i]) + np.sum(A[:,[i]]) - 2
    n_nodes = [len(p) for p in partitions.values()]
    p_inds = [np.array(list(p), dtype=int) for p in partitions.values()]
    n_edges = [np.sum(A[p][:,p]) for p in p_inds]
    
    mod_by_cluster = np.zeros(K)
    for i, part in enumerate(partitions.items()):
        mod_by_cluster[i] = mod_cluster(part, nodes, A, S, D, n_edges[i], n_nodes[i])
        
    return 1/K* sum(mod_by_cluster)

def evaluate_partition(partition, G, nodes, df):
    """Evaluate the partition over a graph with the edge coverage and overlapping
    modularity scores

    Args:
        partition (dict): keys are community ids, values are lists of nodes 
        G (nx.DiGraph): the original structure that is being partitioned
        nodes (list): list of nodes in order of adjacency matrix
        df (pandas DataFrame): dataframe of sampled values
    """
    # Edge coverage
    covered = 0
    for e in list(G.edges):
        is_covered = False
        for _,p in partition.items():
            if e[0] in p and e[1] in p:
                is_covered = True
        covered += 1 if is_covered else 0
    print("Percent of edges covered by partition: {}".format(covered/len(G.edges)))
    
    # Modularity of partitions
    mod_overlap = _modularity_overlapping(partition, nodes, nx.adjacency_matrix(G, nodelist=nodes))
    print("Modularity for Overlapping partitions: {}".format(mod_overlap))
    
    
def delta_causality(est_graph_serial, est_graph_partition, true_graph):
    """Calculate the difference in scores (SHD, AUC, SID, TPR_FPR) between
    the serial estimated grpah and the partitioned estimated graph. The difference
    is calculated as serial_score - partition_score.

    Args:
        est_graph_serial (np.ndarray or nx.DiGraph): the estimated graph from running the causal discovery algorithm on the entire data and node set
        est_graph_partition (np.ndarray or nx.DiGraph): the estimated graph from running the causal discovery algorithm on the partitioned data and node sets
        true_graph (np.ndarray or nx.DiGraph): the ground truth graph to compare to 
    
    Returns:
        list (float, float, float, float, float): Delta SHD, AUC, SID, TPR, FPR. Note that the sign here is relative to the serial implmentation (we do not take the aboslute value)
    """
    scores_s = get_scores(["CD serial"], [est_graph_serial], true_graph)
    scores_p = get_scores(["CD partition"], [est_graph_partition], true_graph)
    delta = [s-p for (s,p) in zip(scores_s, scores_p)]
    return delta
    