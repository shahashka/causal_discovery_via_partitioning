import networkx as nx
import numpy as np
from cd_v_partition.vis_partition import create_partition_plot
from cd_v_partition.overlapping_partition import partition_problem
from cd_v_partition.utils import get_random_graph_data, get_data_from_graph, delta_causality
from cd_v_partition.causal_discovery import sp_gies
from cd_v_partition.fusion import screen_projections
import functools 
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import itertools
import matplotlib.patches as mpatches

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
        if causal_order.index(e[0]) > causal_order.index(e[1]):
            directed_edges.append(e[::-1])
        else:
            directed_edges.append(e)
    directed_graph = nx.DiGraph()
    directed_graph.add_edges_from(directed_edges)
    return directed_graph

def create_two_comms(graph_type, n, m1, m2,p1,p2, nsamples):
    # generate the edges set
    comm_1 = get_random_graph_data(graph_type=graph_type, num_nodes=n, nsamples=0, iv_samples=0, p=p1, k=m1)[0][0]
    comm_2 = get_random_graph_data(graph_type=graph_type, num_nodes=n, nsamples=0, iv_samples=0, p=p2, k=m2)[0][0]
    
    comm_1 = nx.DiGraph(comm_1)
    comm_2 = nx.DiGraph(comm_2)
    
    # connect the two communities using preferential attachment
    num_tiles=2
    degree_sequence = sorted((d for _, d in comm_1.in_degree()), reverse=True)
    dmax = max(degree_sequence)
    tiles = [comm_1, comm_2]

    # First add all communities as disjoint graphs
    tiled_graph = nx.disjoint_union_all(tiles)

    # Each node is preferentially attached to other nodes
    # The number of attached nodes is given by a probability distribution over
    # A = 1, 2 ... min(dmax,4) where the probability is equal to the in_degree=A/number of nodes
    # in the community
    A = np.min([dmax, 4])
    in_degree_a = [sum(np.array(degree_sequence) == a) for a in range(A)]
    leftover = n - sum(in_degree_a)
    in_degree_a[-1] += leftover
    probs = np.array(in_degree_a) / (n)

    # Add connections based on random choice over probability distribution
    for t in range(1, num_tiles):
        for i in range(n):
            node_label = t * n + i
            if len(list(tiled_graph.predecessors(node_label))) == 0:
                num_connected = np.random.choice(np.arange(A), size=1, p=probs)
                dest = np.random.choice(np.arange(t * n), size=num_connected)
                connections = [(node_label, d) for d in dest]
                tiled_graph.add_edges_from(connections)
    causal_tiled_graph = apply_causal_order(tiled_graph)
    init_partition= {0:list(np.arange(n)), 1:list(np.arange(n, 2*n))}
    create_partition_plot(
        causal_tiled_graph,
        list(causal_tiled_graph.nodes()),
        init_partition,
        "{}/two_comm.png".format("./tests/empirical_tests")
    )
    return init_partition, causal_tiled_graph
    
    
def run_causal_discovery(partition, nsamples, graph):
    # Generate data
    df = get_data_from_graph(list(graph.nodes()), list(graph.edges()), nsamples=nsamples, iv_samples=0)[1]
    nodes = list(df.columns)
    nodes.remove('target')
    
    # Break up problem according to provided partition
    adj = nx.adjacency_matrix(graph, nodelist=nodes).todense()
    subproblems = partition_problem(partition, adj, df)
    
    # Local learn
    func_partial = functools.partial(_local_structure_learn)
    results = []
    num_partitions = len(partition)
    nthreads = 4
    chunksize = max(1, num_partitions // nthreads)
    print("Launching processes")

    with ProcessPoolExecutor(max_workers=nthreads) as executor:
        for result in executor.map(func_partial, subproblems, chunksize=chunksize):
            results.append(result)

    # Merge globally
    # data = df.to_numpy()
    # cor = np.corrcoef(data)
    est_graph_partition = screen_projections(partition, results)
    est_graph_partition = nx.adjacency_matrix(
        est_graph_partition, nodelist=np.arange(len(graph.nodes))
    ).todense()

    # Call serial method
    est_graph_serial = _local_structure_learn((adj, df))

    # Compare causal metrics
    d_scores = delta_causality(
        est_graph_serial, est_graph_partition, adj
    )
    return d_scores[-2] # this is the true positive rate  


def _local_structure_learn(subproblem):
    """Call causal discovery algorithm on subproblem. Right now uses SP-GIES

    Args:
        subproblem (tuple np.ndarray, pandas DataFrame): the substructure adjacency matrix and corresponding data

    Returns:
        np.ndarray: Estimated DAG adjacency matrix for the subproblem
    """
    skel, data = subproblem
    adj_mat = sp_gies(data, outdir=None, skel=None, use_pc=True, alpha=0.5)
    return adj_mat

def define_causal_partition(partition, graph):
    unmarked_nodes = list(np.arange(len(graph.nodes)))
    cut_nodes = set()
    for n in graph.nodes():
        comm_n = int(n >= 50)
        for m in nx.neighbors(graph, n):
            comm_m = int(m >=50)
            if comm_n != comm_m:
                cut_nodes.add(m)
                cut_nodes.add(n)
    for a,b in itertools.combinations(cut_nodes, 2):
        if nx.has_path(graph, a,b) or nx.has_path(graph, b,a):
            #print("Path found between {} and {}".format(a,b))
            comm_a = int(n >= 50)
            partition[comm_a] += [b]
    partition[0] = list(set(partition[0]))
    partition[1] = list(set(partition[1]))
    create_partition_plot(
        graph,
        list(graph.nodes()),
        partition,
        "{}/causal_cover.png".format("./tests/empirical_tests")
    )
    return partition

    
def define_rand_edge_coverage(partition, graph):
    unmarked_nodes = list(np.arange(len(graph.nodes)))
    for n in graph.nodes():
        comm_n = int(n >= 50)
        for m in nx.neighbors(graph, n):
            n_unmarked = (n in unmarked_nodes)
            m_unmarked = (m in unmarked_nodes)
            if  n_unmarked or m_unmarked:
                comm_m = int(m >=50)
                if comm_n != comm_m:
                    if m%2: # Randomly assign the cut nodes to one or the other partition to ensure edge coverage
                        partition[comm_n] += [m]
                    else:
                        partition[comm_m] += [n]
                    if m_unmarked:
                        unmarked_nodes.remove(m)
                    if n_unmarked:
                        unmarked_nodes.remove(n)
    partition[0] = list(set(partition[0]))
    partition[1] = list(set(partition[1]))
    create_partition_plot(
        graph,
        list(graph.nodes()),
        partition,
        "{}/edge_cover.png".format("./tests/empirical_tests")
    )
    return partition
        

num_repeats = 30
scores_edge_cover = []
scores_hard_partition = []
scores_causal_partition = []
sample_range = [1e2, 1e3, 1e4, 1e5]
for ns in sample_range:
    score_ec = []
    score_hp = []
    score_cp = []
    for i in range(num_repeats):
        print(i)
        init_partition, graph = create_two_comms("scale_free", n=50, m1=2, m2=1,p1=0.5,p2=0.5, nsamples=0)
        d_shd = run_causal_discovery(init_partition, nsamples=int(ns), graph=graph)
        score_hp.append(d_shd)
        
        partition = define_rand_edge_coverage(init_partition, graph)
        d_shd = run_causal_discovery(partition, nsamples=int(ns), graph=graph)
        score_ec.append(d_shd)
        
        partition = define_causal_partition(init_partition, graph)
        d_shd = run_causal_discovery(partition, nsamples=int(ns), graph=graph)
        score_cp.append(d_shd)
        
        
    scores_edge_cover.append(score_ec)
    scores_hard_partition.append(score_hp)
    scores_causal_partition.append(score_cp)
    
labels = []
def add_label(violin, label):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), label))
plt.clf()
_, ax = plt.subplots()
add_label(ax.violinplot(scores_edge_cover, showmeans=True, showmedians=False), label='edge_cover')
add_label(ax.violinplot(scores_hard_partition, showmeans=True, showmedians=False), label='hard_partition')
add_label(ax.violinplot(scores_causal_partition, showmeans=True, showmedians=False), label='causal_partition')

ax.set_xticks(
    np.arange(1, len(sample_range) + 1), labels=['1e2', '1e3', '1e4', '1e5'], rotation=45
)
plt.legend(*zip(*labels), loc=2)
plt.savefig("./tests/empirical_tests/causal_part_test_sparse.png")
    
# init_partition, graph = create_two_comms("scale_free", n=50, m1=5, m2=2,p1=0.5,p2=0.5, nsamples=0)
# partition = define_causal_partition(init_partition, graph)
# d_shd = run_causal_discovery(partition, nsamples=int(1e3), graph=graph)
# print(d_shd)