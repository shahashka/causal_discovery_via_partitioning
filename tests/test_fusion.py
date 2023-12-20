from cd_v_partition.fusion import screen_projections
import networkx as nx

# Test case for a simple chain graph
chain = nx.DiGraph([(0,1), (1,2), (2,3)])
partition = {0:[0,1,2], 1:[1,2,3]}
comm1 = nx.DiGraph()
comm1.add_nodes_from(partition[0])
comm1.add_edges_from([(0,1), (1,2)])
comm2 = nx.DiGraph()
comm2.add_nodes_from(partition[1])
comm2.add_edges_from([(1,2), (2,3)])

# Directed edges agree
local_adj_mats = [nx.adjacency_matrix(comm1, nodelist=[0,1,2]), nx.adjacency_matrix(comm2, nodelist=[1,2,3])]
test1 = screen_projections(partition, local_adj_mats)
assert(test1.edges() == chain.edges())

# Directed edges conflict over direction
comm2.remove_edge(1,2)
comm2.add_edge(2,1)
local_adj_mats = [nx.adjacency_matrix(comm1, nodelist=[0,1,2]), nx.adjacency_matrix(comm2, nodelist=[1,2,3])]
test2 = screen_projections(partition, local_adj_mats)
chain.add_edge(2,1)
assert(test2.edges() == chain.edges())

# Comm1 has directed edge, Comm2 has no edge
comm2.remove_edge(2,1)
local_adj_mats = [nx.adjacency_matrix(comm1, nodelist=[0,1,2]), nx.adjacency_matrix(comm2, nodelist=[1,2,3])]
test3 = screen_projections(partition, local_adj_mats)
chain.remove_edge(2,1)
chain.remove_edge(1,2)
assert(test3.edges() == chain.edges())

# Comm1 has no edge, Comm2 has directed edge
comm1.remove_edge(1,2)
comm2.add_edge(2,1)
local_adj_mats = [nx.adjacency_matrix(comm1, nodelist=[0,1,2]), nx.adjacency_matrix(comm2, nodelist=[1,2,3])]
test4 = screen_projections(partition, local_adj_mats)
assert(test4.edges() == chain.edges())




