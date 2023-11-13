from causal_discovery import pc, cu_pc
from utils import get_data_from_graph
import numpy as np
import itertools
nodes = np.arange(10)
edges = [(0, 3), (0, 4), (0, 5), (1, 5), 
            (1, 6),  (1, 7), (1, 8),  (2,8), (2, 9)]
(arcs, _, _, _), data = get_data_from_graph(nodes, edges, nsamples=int(1e5), iv_samples=0,
                                            save=False, outdir="./tests")
obs_data = data.drop(columns=['target']).to_numpy()
pdag, p_values = pc(obs_data, alpha=0.001, outdir="./tests")
print(p_values)
# Check that PDAG is correct (colliders identified: (0->5<-1, 1->8<-2))
colliders = [(0,5), (1,5), (1,8), (2,8)]
for (row,col) in itertools.product(np.arange(pdag.shape[0]), np.arange(pdag.shape[1])):
    if (row, col) in edges:
        assert(pdag[row,col] == 1)
        if (row,col) not in colliders:
            assert(pdag[col, row] == 1)
        else:
            assert(pdag[col, row] == 0)

# Make sure GPU implementation is aligned with original pcalg version 
# Currently this fails! Skeleton is right but orientation is wrong 

pdag_gpu, p_values_gpu = cu_pc(obs_data, alpha=0.001, outdir="./tests")
assert((pdag_gpu == pdag).all())
assert((np.abs(np.abs(p_values_gpu) - np.abs(p_values)) < 1e-4).all())
