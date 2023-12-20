import networkx as nx
import numpy as np
import itertools
# Final fusion step to merge subgraphs
# In infinite data limit this is done by screening for conflicting edges during union over subgraphs 

def screen_projections(partition, local_cd_adj_mats):
    """Fuse subgraphs by taking the union and resolving conflicts by favoring no edge over
    directed edge. Leave bidirected edges as is. This is the method used for 'infinite' data limit problems

    Args:
        partition (dict): the partition as a dictionary {comm_id : [nodes]}
        local_cd_adj_mat (list[np.ndarray]): list of adjacency matrices for each local subgraph 

    Returns:
        nx.DiGraph: the final global directed graph with all nodes and edges
    """
    # Take the union over graphs
    local_cd_graphs = _convert_local_adj_mat_to_graph(partition, local_cd_adj_mats)
    global_graph = _union_with_overlaps(local_cd_graphs)
    
    # global_graph = no edge if (no edge in comm1) or (no edge in comm2)
    for comm, adj_comm in zip(partition.values(), local_cd_adj_mats):
        for row, col in itertools.product(np.arange(adj_comm.shape[0]),np.arange(adj_comm.shape[0])):
            i = comm[row]
            j = comm[col]
            if (not adj_comm[row,col] and not adj_comm[col,row]) and global_graph.has_edge(i,j):
                global_graph.remove_edge(i,j)
    return global_graph

def fusion(partition, local_cd_adj_mats):
    """Fuse subgraphs by taking the union and resolving conflicts by taking the higher
    weighted edge (for now). Eventually we want to the proof to inform how the merge happens here
    and we also want to consider finite data affects.

    Args:
        partition (dict): the partition as a dictionary {comm_id : [nodes]}
        local_cd_adj_mat (list[np.ndarray]): list of adjacency matrices for each local subgraph 

    Returns:
        nx.DiGraph: the final global directed graph with all nodes and edges
    """
    # Convert adjacency matrices to nx.DiGraphs, make sure to label nodes using the partition
    local_cd_graphs = _convert_local_adj_mat_to_graph(partition, local_cd_adj_mats)
    
    # Take the union over graphs
    global_graph = _union_with_overlaps(local_cd_graphs)
    
    #  Resolve conflicts by favoring higher weights
    global_graph_resolved = global_graph.copy() # TODO this is an expensive copy 
    for (i,j) in global_graph.edges():
        if global_graph.has_edge(j,i):
            weight_ij = global_graph.get_edge_data(i,j)['weight']
            weight_ji = global_graph.get_edge_data(j,i)['weight']
            print("Conflict found, weights: {} {}".format(weight_ij, weight_ji))
            if weight_ij > weight_ji:
                global_graph_resolved.remove_edge(j,i)
            else:
                global_graph_resolved.remove_edge(i,j)

    return global_graph_resolved

def _convert_local_adj_mat_to_graph(partition, local_cd_adj_mats):
    """
    Helper function to convert the local adjacency matrices (resultant of the causal discovery
    method) into networkx DiGraphs using the correct global node index for each partition.
    """
    local_cd_graphs = []
    for part,adj in zip(partition.items(), local_cd_adj_mats):
        _, node_ids = part
        subgraph = nx.from_numpy_array(adj, create_using=nx.DiGraph)
        nx.relabel_nodes(subgraph, mapping=dict(zip(np.arange(len(node_ids)), node_ids)),copy=False)
        local_cd_graphs.append(subgraph)
    return local_cd_graphs

def _union_with_overlaps(graphs):
    """
    Helper function that reimplements networkx.union_all, except remove the
    requirement that the node sets be disjoint ie we allow for overlapping nodes/edges
    between graphs 
    """
    R = None
    seen_nodes = set()
    for i, G in enumerate(graphs):
        G_nodes_set = set(G.nodes)
        if i == 0:
            # Union is the same type as first graph
            R = G.__class__()
        seen_nodes |= G_nodes_set
        R.graph.update(G.graph)
        R.add_nodes_from(G.nodes(data=True))
        R.add_edges_from(
            G.edges(keys=True, data=True) if G.is_multigraph() else G.edges(data=True)
        )

    if R is None:
        raise ValueError("cannot apply union_all to an empty list")

    return R


def _cycle_resolution(G):
    print("TODO")
    
    
# TODO implement conflict resolution from this paper: https://arxiv.org/abs/1403.2150



    