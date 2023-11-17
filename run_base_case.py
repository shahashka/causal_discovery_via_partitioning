# Run superstructure creation, partition, local discovery and screening 
# for a base case network with assumed community structure
from utils import get_random_graph_data, get_data_from_graph
from causal_discovery import pc, weight_colliders
import networkx as nx
import numpy as np
import pandas as pd

def run_base_case(algorithm, structure_type, data_dir):
    print("TODO")               
def create_base_case_net(graph_type, n, p, k, ncommunities, alpha, collider_weight, nsamples, outdir):
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
    (arcs, nodes, _, _), _ = get_random_graph_data(graph_type, n=n, nsamples=0, 
                                                   iv_samples=0, p=p, k=k, save=False)
    net = nx.DiGraph()
    net.add_edges_from(arcs)
    net.add_nodes_from(nodes)
    
    # Create a tiled network with community structure, save to dataset directory
    nodes = np.arange(n*ncommunities)
    tiled_net = _construct_tiling('base_case', net, num_tiles=ncommunities)
    df_edges = pd.DataFrame(list(tiled_net.edges))
    df_edges.to_csv("{}/edges_dag.dat".format(outdir),sep='\t', header=None, index=None)
    adj_mat = nx.adjacency_matrix(tiled_net, nodelist=nodes).toarray()
    df = pd.DataFrame(adj_mat)
    df.to_csv("{}/tiled.csv".format(outdir), index=False)
    
    # Generate data from the tiled network
    (arcs, nodes, _, _), df = get_data_from_graph( nodes, list(tiled_net.edges()),
                                                  nsamples=nsamples, iv_samples=0, save=True, 
                                                  outdir=outdir)
    
    # Use the data to generate a superstructure using the pc algorithm
    data = df.drop(columns=['target'])
    data = data.to_numpy()
    superstructure, p_values = pc(data, alpha=alpha, outdir=outdir) 
    superstructure = weight_colliders(superstructure, weight=collider_weight)
    weights = np.multiply(superstructure, p_values)
    superstructure_net = nx.from_numpy_matrix(weights, create_using=nx.Graph)
    
    # Save the super structure edges and weighted superstructure edges               
    df_edges = pd.DataFrame(list(superstructure_net.edges))
    df_edges.to_csv("{}/edges_superstructure.dat".format(outdir),sep='\t', header=None, index=None)

    df_edges['weight'] = [superstructure_net.get_edge_data(u,v)["weight"] for (u,v) in superstructure_net.edges]
    df_edges.to_csv("{}/edges_superstructure_weighted.dat".format(outdir),sep='\t', header=None, index=None)
    
    # Checks for sanity 
    print("Number of colliders: {}".format(_count_colliders(tiled_net)))
    #check_superstructure(superstructure, nx.adjacency_matrix(tiled_net, nodelist=np.arange(len(nodes))))

    return superstructure_net, df


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
    in_degree_a = [sum(np.array(degree_sequence)==a) for a in range(A)]
    leftover = num_nodes-sum(in_degree_a)
    in_degree_a[-1] += leftover
    probs = np.array(in_degree_a)/(num_nodes)
    
    # Add connections based on random choice over probability distribution
    for t in range(1,num_tiles):
        for i in range(num_nodes):
            node_label = t*num_nodes + i
            if len(list(tiled_graph.predecessors(node_label))) == 0:
                num_connected = np.random.choice(np.arange(A),size=1, p=probs)
                dest = np.random.choice(np.arange(t*num_nodes),size=num_connected)
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
    collider_nodes = []
    # Find all triples x-y-z
    for (x,y,z) in nx.all_triplets(G):
        if G.has_edge(x,y) and G.has_edge(z,y):
            num_colliders += 1
            collider_nodes.append(y)
    print( len(set(collider_nodes)))
    return num_colliders

# Check that this is a superstructure 
def _check_superstructure(S, G):
    """Make sure that S is a superstructure of G. This means all edges in G are contrained 
       by S. 
    

    Args:
        S (np.ndarray): adjacency matrix for the superstructure
        G (np.ndattay): adjacency matrix for the DAG 
    """
    for row in np.arange(S.shape[0]):
        for col in np.arange(S.shape[1]):
            if G[row,col] == 1:
                print(row,col)
                assert(S[row,col]>0)
                
if __name__ == '__main__':
    create_base_case_net('scale_free', n=10, p=0.9, k=5, ncommunities=5, 
                         alpha=0.5, collider_weight=10, nsamples=int(1e5),
                         outdir="./datasets/base_case/")