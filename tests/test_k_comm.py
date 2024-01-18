from cd_v_partition.utils import create_k_comms
from cd_v_partition.vis_partition import create_partition_plot
import numpy as np
import networkx as nx

# Generate 5 comm graphs with varying modularity 
num_edges = np.arange(0,0.1,0.01)
for n in num_edges:
    hard_partition, comm_graph = create_k_comms(graph_type="scale_free", n=10, 
                                                           m_list=5*[1], 
                                                           p_list=5*[0.1], 
                                                           k=5,rho=n)
    create_partition_plot(comm_graph, list(comm_graph.nodes()), hard_partition, save_name="./tests/comm_{}.png".format(n))
    print(nx.community.modularity(comm_graph, hard_partition.values()))