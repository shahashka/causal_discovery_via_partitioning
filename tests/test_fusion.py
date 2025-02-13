import networkx as nx
import numpy as np

from cd_v_partition.fusion import fusion, screen_projections
from cd_v_partition.utils import edge_to_adj, get_data_from_graph

# Test case for a simple chain graph
G_star_edges = [(0, 1), (1, 2), (2, 3)]
chain = nx.DiGraph(G_star_edges)
data = get_data_from_graph(np.arange(4), G_star_edges, nsamples=int(1e3), iv_samples=0)
samples = data[-1].drop(columns=["target"]).to_numpy()

partition = {0: [0, 1, 2], 1: [1, 2, 3]}
comm1 = nx.DiGraph()
comm1.add_nodes_from(partition[0])
comm1.add_edges_from([(0, 1), (1, 2)])
comm2 = nx.DiGraph()
comm2.add_nodes_from(partition[1])
comm2.add_edges_from([(1, 2), (2, 3)])

# Directed edges agree
# Comm 1: 0->1->2
# Comm 2: 1->2->3
local_adj_mats = [
    nx.adjacency_matrix(comm1, nodelist=[0, 1, 2]),
    nx.adjacency_matrix(comm2, nodelist=[1, 2, 3]),
]
ss_edges = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
ss = edge_to_adj(ss_edges, list(np.arange(4)))
test1, _ = screen_projections(partition, local_adj_mats)
assert test1.edges() == chain.edges()  # 0->1->2->3
test1, _ = fusion(ss, partition, local_adj_mats, samples, full_cand_set=True)
print(test1.edges())
assert list(test1.edges()) == G_star_edges  # 0->1->2->3

# Directed edges conflict over direction
# Comm 1: 0->1->2
# Comm 2: 1<-2->3
comm2.remove_edge(1, 2)
comm2.add_edge(2, 1)
local_adj_mats = [
    nx.adjacency_matrix(comm1, nodelist=[0, 1, 2]),
    nx.adjacency_matrix(comm2, nodelist=[1, 2, 3]),
]
test2, _ = screen_projections(partition, local_adj_mats)
chain.add_edge(2, 1)
assert test2.edges() == chain.edges()  # 0->1-2->3
test2, _ = fusion(ss, partition, local_adj_mats, samples, full_cand_set=True)
print(list(test2.edges()), G_star_edges)
assert list(test2.edges()) == G_star_edges  # 0->1->2->3

# Comm1 has directed edge, Comm2 has no edge
# Comm 1: 0->1->2
# Comm 2: 1, 2->3
comm2.remove_edge(2, 1)
local_adj_mats = [
    nx.adjacency_matrix(comm1, nodelist=[0, 1, 2]),
    nx.adjacency_matrix(comm2, nodelist=[1, 2, 3]),
]
test3, _ = screen_projections(partition, local_adj_mats)
chain.remove_edge(2, 1)
chain.remove_edge(1, 2)
assert test3.edges() == chain.edges()  # 0->1,2->3
test3, _ = fusion(ss, partition, local_adj_mats, samples, full_cand_set=True)
assert list(test3.edges()) == G_star_edges  # 0->1->2->3

# Comm1 has no edge, Comm2 has directed edge
# Comm 1: 0->1,2
# Comm 2: 1<-2->3
comm1.remove_edge(1, 2)
comm2.add_edge(2, 1)
local_adj_mats = [
    nx.adjacency_matrix(comm1, nodelist=[0, 1, 2]),
    nx.adjacency_matrix(comm2, nodelist=[1, 2, 3]),
]
test4, _ = screen_projections(partition, local_adj_mats)
assert test4.edges() == chain.edges()  # 0->1,2->3
test4, _ = fusion(ss, partition, local_adj_mats, samples, full_cand_set=True)
assert list(test4.edges()) == G_star_edges  # 0->1->2->3
print("All tests passed!")
