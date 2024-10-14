from cd_v_partition.fusion import screen_projections, fusion, screen_projections_pag2cpdag
import networkx as nx
from cd_v_partition.utils import get_data_from_graph, edge_to_adj
import numpy as np

# Test case for a simple chain graph
G_star_edges = [(0, 1), (1, 2), (2, 3)]
chain = nx.DiGraph(G_star_edges)
data = get_data_from_graph(np.arange(4), G_star_edges, nsamples=int(1e3), iv_samples=0)
samples = data[-1].drop(columns=['target']).to_numpy()

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
ss_edges = [(0,1), (1,0), (1,2), (2,1), (2,3), (3,2)]
ss = edge_to_adj(ss_edges, list(np.arange(4)))
test1 = screen_projections(ss, partition, local_adj_mats, finite_lim=False)
assert(test1.edges() == chain.edges())  # 0->1->2->3
test1 = fusion(ss, partition, local_adj_mats, samples, full_cand_set=False) # since partitions overlap, only add overlapping cand set
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
test2 = screen_projections(ss, partition, local_adj_mats, finite_lim=False)
chain.add_edge(2, 1)
assert test2.edges() == chain.edges()  # 0->1-2->3
test2 = fusion(ss, partition, local_adj_mats, samples, full_cand_set=False)
print(list(test2.edges()), G_star_edges)
assert (
    list(test2.edges()) == G_star_edges
)  # 0->1->2->3 

# Comm1 has directed edge, Comm2 has no edge
# Comm 1: 0->1->2
# Comm 2: 1, 2->3
comm2.remove_edge(2, 1)
local_adj_mats = [
    nx.adjacency_matrix(comm1, nodelist=[0, 1, 2]),
    nx.adjacency_matrix(comm2, nodelist=[1, 2, 3]),
]
test3 = screen_projections(ss, partition, local_adj_mats, finite_lim=False)
chain.remove_edge(2, 1)
chain.remove_edge(1, 2)
assert test3.edges() == chain.edges()  # 0->1,2->3
test3 = fusion(ss, partition, local_adj_mats, samples, full_cand_set=False)
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
test4 = screen_projections(ss, partition, local_adj_mats, finite_lim=False)
assert test4.edges() == chain.edges()  # 0->1,2->3
test4 = fusion(ss, partition, local_adj_mats, samples, full_cand_set=False)
assert list(test4.edges()) == G_star_edges  # 0->1->2->3



## PAG fusion method tests

# Arrowheads agree
# Comm 1: 0 -> 1 <- 2
# Comm 2: 1 <- 2 -> 3
# Expected Output: 0 -> 1 <- 2 -- 3
expected_edges = [(0,1), (2,1), (2,3), (3,2)]
adj_1 = np.array([[0, 2, 0], [3, 0, 3], [0, 2, 0]])
adj_2 = np.array([[ 0, 3, 0 ], [ 2, 0, 2] , [ 0, 3, 0]])
test5 = screen_projections_pag2cpdag(ss, partition, [adj_1, adj_2], finite_lim=False)
print(test5.edges())
assert set(test5.edges()) == set(expected_edges)
# Comm 1: 0 -> 1 <- 2
# Comm 2: 1 <-o 2 -> 3
# Expected Output: 0 -> 1 <- 2 -- 3
adj_1 = np.array([[0, 2, 0], [3, 0, 3], [0, 2, 0]])
adj_2 = np.array([[ 0, 1, 0 ], [ 2, 0, 2] , [ 0, 3, 0]])
test6 = screen_projections_pag2cpdag(ss, partition, [adj_1, adj_2], finite_lim=False)
assert set(test6.edges()) == set(expected_edges)

# Arrowheads disagree
# Comm 1: 0 -> 1 <- 2
# Comm 2: 1 o-o 2 -> 3
# Expected Output: 0 -> 1 <- 2 -- 3
expected_edges = [(0,1), (2,1), (2,3), (3,2)]
adj_1 = np.array([[0, 2, 0], [3, 0, 3], [0, 2, 0]])
adj_2 = np.array([[ 0, 1, 0 ], [ 1, 0, 2] , [ 0, 3, 0]])
test7 = screen_projections_pag2cpdag(ss, partition, [adj_1, adj_2], finite_lim=False)
print(test7.edges(), expected_edges)
assert set(test7.edges()) == set(expected_edges)

# Comm 1: 0 -> 1 <- 2
# Comm 2: 1 o-> 2 -> 3
# Expected Output: 0 -> 1 <- 2 -- 3
adj_1 = np.array([[0, 2, 0], [3, 0, 3], [0, 2, 0]])
adj_2 = np.array([[ 0, 2, 0 ], [ 1, 0, 2] , [ 0, 3, 0]])
test8 = screen_projections_pag2cpdag(ss, partition, [adj_1, adj_2], finite_lim=False)
assert set(test8.edges()) == set(expected_edges)

# Edges disagree
# Comm 1: 0 -> 1 -> 2
# Comm 2: 1, 2 -> 3
# Expected Output: 0 -- 1, 2 -- 3
expected_edges = [(0,1), (1,0), (2,3), (3,2)]
adj_1 = np.array([[0, 2, 0], [3, 0, 2], [0, 3, 0]])
adj_2 = np.array([[ 0, 0, 0 ], [ 0, 0, 2] , [ 0, 3, 0]])
test9 = screen_projections_pag2cpdag(ss, partition, [adj_1, adj_2], finite_lim=False)
print(test9.edges())
assert set(test9.edges()) == set(expected_edges)
print("All tests passed!")
