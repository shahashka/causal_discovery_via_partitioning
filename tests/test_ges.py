from cd_v_partition.causal_discovery import sp_gies
from cd_v_partition.utils import get_data_from_graph, get_scores, edge_to_adj
import numpy as np
import itertools

"""
    Create a simple 10 node graph with 2 colliders 
    Make sure the GES algorithm finds the correct graph
    Test if SP-GES can find the correct graph with a superstructure
    Test the effect of the number of samples
"""

nodes = np.arange(10)
edges = [(0, 3), (0, 4), (0, 5), (1, 5), (1, 6), (1, 7), (1, 8), (2, 8), (2, 9)]
(arcs, _, _, _), data = get_data_from_graph(
    nodes, edges, nsamples=int(1e5), iv_samples=0, save=False, outdir=None
)
obs_data = data.drop(columns=["target"]).to_numpy()
adj_mat = sp_gies(data, skel=None, use_pc=False, outdir=None)

#Check that the graph is correct
for row, col in itertools.product(np.arange(adj_mat.shape[0]), np.arange(adj_mat.shape[1])):
    if (row, col) in edges:
        assert adj_mat[row, col] != 0
    else:
        assert adj_mat[row, col] == 0

# Check metrics
shd, sid, auc, tpr, fpr = get_scores(
    ["GES"], [adj_mat], edge_to_adj(arcs, list(np.arange(10)))
)
assert shd == 0 and tpr == 1
print("All tests passed!")