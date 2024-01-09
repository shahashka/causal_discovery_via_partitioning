import networkx as nx
import numpy as np

from cd_v_partition.utils import get_random_graph_data, delta_causality, get_scores
from cd_v_partition.overlapping_partition import partition_problem
from cd_v_partition.causal_discovery import sp_gies, pc
import matplotlib.pyplot as plt
import pylab
from netgraph import Graph
from cd_v_partition.vis_partition import _create_patches

# Create G_star
(sf_arcs, _,_,_), df = get_random_graph_data(graph_type='scale_free', num_nodes=50, nsamples=int(1e4), iv_samples=0, p=0.2, k=1)
print(len(sf_arcs))
sf_graph = nx.DiGraph(sf_arcs)
sf_adj = nx.adjacency_matrix(sf_graph, nodelist=np.arange(50)).todense()

# Set boundary arbitrarily so that 'inner' subset is 0-39 and 'boundary' is '40-49'
sub_nodes = np.arange(40)
partition = {0:sub_nodes, 1:np.arange(40,50)}

# Partition the data and run on 'inner' subset and 'full' subset
subproblems = partition_problem(partition, sf_adj, df)
skel, data = subproblems[0]
alpha = 5e-1 # large alpha -> more denser superstructure
inner_adj = sp_gies(data, outdir=None, skel=None, use_pc=True, alpha=alpha)
full_adj = sp_gies(df, outdir=None, skel=None, use_pc=True, alpha=alpha)

# ignore learned parameters
inner_adj[inner_adj>0] = 1
full_adj[full_adj>0] = 1


# causal learner on 'inner' subset is A_40
# causal learner on 'inner' + 'boundary' is A_50
# Vis A_50
A_50 = nx.from_numpy_array(full_adj, create_using=nx.DiGraph)
pos = nx.spring_layout(A_50, k=5/np.sqrt(50))

# Annoying vis stuff
cm = pylab.get_cmap("plasma")
colors = []
num_colors = len(partition)
for i in range(num_colors):
    colors.append(cm(1.0 * i / num_colors))  # color will now be an RGBA tuple
color_map = dict(zip(np.arange(num_colors), colors))
overlaps = []
colors = dict(
    zip(
        np.arange(50),
        [
            color_map[0] if node<40  else color_map[1]
            for node in np.arange(50)
        ]
    )
)
# Wherever A_40 disagrees with A(X_50), color the edge red 
edges = list(A_50.edges())
edge_colors = ['red' if (edge[0] <40 and edge[1] < 40) and (full_adj[edge[0], edge[1]] != inner_adj[edge[0], edge[1]]) else 'black' for edge in edges]
edge_colors = dict(zip(edges, edge_colors))
_, ax = plt.subplots()

# Plot the graph
Graph(
    A_50,
    edge_width=1,
    node_size=5,
    edge_color=edge_colors,
    node_layout=pos,
    node_color=colors,
    arrows=True,
    ax=ax,
)

# Create the patches to indicate inner and boundary 
for comm, nodes in partition.items():
    _create_patches(pos, ax, nodes, color_map[comm])

plt.savefig("./tests/empirical_tests/boundary_alpha_{}.png".format(alpha))

get_scores(['full'], [full_adj], sf_adj)
delta_causality(full_adj[sub_nodes][:,sub_nodes], inner_adj, sf_adj[sub_nodes][:,sub_nodes])
