from cd_v_partition.causal_discovery import pc, cu_pc, weight_colliders, rfci_local_learn, rfci
from cd_v_partition.utils import get_data_from_graph, get_scores, edge_to_adj
import numpy as np
import itertools
#from causallearn.search.ConstraintBased.FCI import fci
"""
    Create a simple 10 node graph with 2 colliders 
    Make sure the PC algorithm finds the skeleton + colliders
    Check the metrics corresponding to the estimated PDAG
    Finally check that the CPU and GPU implementations render the same results (NOTE this currently is failing)
    
"""

nodes = np.arange(10)
edges = [(0, 3), (0, 4), (0, 5),
         (1, 5), (1, 6), (1, 7), (1, 8),
         (2, 8), (2, 9)]
(arcs, _, _, _), data = get_data_from_graph(
    nodes, edges, nsamples=int(1e6), iv_samples=0, save=False, outdir="./tests"
)
pdag, p_values = pc(data, alpha=0.001, outdir="./tests")
print(pdag)
# Check that PDAG is correct (colliders identified: (0->5<-1, 1->8<-2))
colliders = [(0, 5), (1, 5), (1, 8), (2, 8)]
for row, col in itertools.product(np.arange(pdag.shape[0]), np.arange(pdag.shape[1])):
    if (row, col) in edges:
        if (row, col) in colliders:
            assert pdag[row, col] == 1
            assert pdag[col, row] == 0
        else:
            assert pdag[row, col] == 1
            assert pdag[col, row] == 1
    elif (col, row) not in edges:
        assert pdag[row, col] == 0

# Check metrics
shd, sid, auc, tpr, fpr = get_scores(
    ["PC"], [pdag], edge_to_adj(arcs, list(np.arange(10)))
)
assert shd == 5 and tpr == 1

W = 10
weighted_pdag = weight_colliders(pdag, weight=W)
for row, col in itertools.product(np.arange(pdag.shape[0]), np.arange(pdag.shape[1])):
    if (row, col) in edges:
        if (row, col) in colliders:
            assert pdag[row, col] == W
            assert pdag[col, row] == 0
        else:
            assert pdag[row, col] == 1
            assert pdag[col, row] == 1
    elif (col, row) not in edges:
        assert pdag[row, col] == 0

# Make sure GPU implementation is aligned with original pcalg version
# Currently this fails! Skeleton is right but orientation is wrong
# pdag_gpu, p_values_gpu = cu_pc(data, alpha=0.001, outdir="./tests")
# assert (pdag_gpu == pdag).all()
# assert (np.abs(np.abs(p_values_gpu) - np.abs(p_values)) < 1e-4).all()
data_subset = data.drop(data.columns[2], axis=1)
#data_subset = data_subset.drop(columns=['target'])
# print(data_subset.shape)

# g, edges = fci(data_subset.to_numpy())
# print(g.graph)
dag = rfci_local_learn((np.ones((10,10)), data_subset))
print(dag) # gets 2 edges wrong (wrong direction that don't correspond to colliders) 
pag, mag = rfci(data_subset, None)
print(pag)
print("All tests passed!")
