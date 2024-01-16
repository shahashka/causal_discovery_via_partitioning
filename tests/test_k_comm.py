from cd_v_partition.utils import create_k_comms
from cd_v_partition.vis_partition import create_partition_plot
import numpy as np
import networkx as nx

# Generate 5 comm graphs with varying modularity 
num_edges = np.arange(1,200,50)
for n in num_edges:
    hard_partition, comm_graph = create_k_comms(graph_type="scale_free", n=25, 
                                                           m_list=5*[2], 
                                                           p_list=5*[0.2], 
                                                           k=5,tune_mod=n)
    create_partition_plot(comm_graph, list(comm_graph.nodes()), hard_partition, save_name="./tests/comm_{}.png".format(n))
    print(nx.community.modularity(comm_graph, hard_partition.values()))