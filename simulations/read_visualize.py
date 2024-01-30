import numpy as np
import pandas as pd
from cd_v_partition.utils import edge_to_adj, get_scores
from common_funcs import save
def read_experiment(path, parameter, sweep, algs, num_repeats, fusion_algs,x_label, nnodes):
    for f in fusion_algs:
        path_data = "{}/{}/{}_".format(path, f, parameter)
        scores = []
        for alg in algs:
            scores_by_alg = np.zeros((num_repeats, len(sweep), 5))
            for i, s in enumerate(sweep):
                for r in np.arange(num_repeats):
                    if parameter == "nnodes":
                        nnodes = s
                    edges_true = pd.read_csv("{}{}/{}/edges_true.csv".format(path_data, s, r), header=0).to_numpy()
                    G_star = edge_to_adj(edges_true, list(np.arange(nnodes)))
                    edges = pd.read_csv("{}{}/{}/edges_{}.csv".format(path_data, s, r, alg), header=0).to_numpy()
                    scores_by_alg[r][i] = get_scores([alg], [edge_to_adj(edges,list(np.arange(nnodes)))],G_star)
            scores.append(scores_by_alg)
        save(path, scores, algs, num_repeats, sweep, x_label, screen=f=='screen_projections', time=False)
                    
                    
#read_experiment("./simulations/experiment_1/", "samples", [10**i for i in range(1,7)], ['serial', 'pef', 'edge_cover', 'causal', 'mod'],5,['screen_projections'], "Number of samples", 50 )
#read_experiment("./simulations/experiment_2/", "rho", np.arange(0,0.5,0.1), ['serial', 'pef', 'edge_cover', 'causal', 'mod'],5,['screen_projections'], "Rho", 50 )
read_experiment("./simulations/experiment_5_no_pef/", "nnodes", [10**i for i in range(1,4)], ['serial', 'edge_cover', 'causal', 'mod'],1,['fusion'], "Number of nodes", 50 )
read_experiment("./simulations/experiment_5_pef_only/", "nnodes", [10**i for i in range(1,5)], ['pef'],1,['fusion'], "Number of nodes", 50 )
