from cd_v_partition.vis_experiment import vis_experiment
import numpy as np

cd_algs = ["GES", "PC", "NOTEARS", "RFCI-PAG"]

# dir = "simulations/experiment_1_refactor_multi_algs_new_exps_corr_ss"
# for cd_alg in cd_algs:
#     vis_experiment(1, dir, ['no_partition', 'modularity', 'edge_cover', 'expansive_causal', 'PEF'],
#                     cd_alg, 10, "num_samples", [10**i for i in np.arange(1, 6)])

dir = "simulations/experiment_1_refactor_multi_algs_new_exps_rfci_tmlr"
for cd_alg in cd_algs:
    vis_experiment(
        1,
        dir,
        ["no_partition", "modularity", "edge_cover", "expansive_causal"],
        cd_alg,
        10,
        "num_samples",
        [10**i for i in np.arange(1, 7)],
    )

# dir = "simulations/experiment_2_refactor_multi_algs"
# for cd_alg in cd_algs:
#     vis_experiment(2, dir, ['no_partition', 'modularity', 'edge_cover', 'expansive_causal', 'PEF'],
#                     cd_alg, 10, "inter_edge_prob", [0, 0.01, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5])

# dir = "simulations/experiment_3_refactor_multi_algs"
# for cd_alg in cd_algs:
#     vis_experiment(3, dir, ['no_partition', 'modularity', 'edge_cover', 'expansive_causal', 'PEF'],
#                     cd_alg, 10, "frac_extraneous_edges", np.arange(0, 4, 0.5))

# dir = "simulations/experiment_4_refactor_multi_algs"
# for cd_alg in cd_algs:
#     vis_experiment(4, dir, ['no_partition', 'modularity'],
#                     cd_alg, 10, "alpha", np.arange(0.1, 1, 0.1))

# dir = "simulations/experiment_5_refactor_multi_algs"
# for cd_alg in cd_algs:
#     vis_experiment(5, dir, ['no_partition', 'modularity', 'edge_cover', 'expansive_causal'],
#                     cd_alg, 1, "num_nodes", [10**i for i in np.arange(1, 5)])

# dir = "simulations/experiment_5_refactor_multi_algs_small_comms"
# vis_experiment(5, dir, ['modularity', 'edge_cover', 'expansive_causal'],
#                      "GES", 1, "num_nodes", [10**i for i in np.arange(4, 5)])
