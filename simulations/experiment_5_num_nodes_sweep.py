# Experiment 5: hierarchical networks, num samples 1e5,
# artificial superstructure with 10% extraneous edges,
# num_trials=30, default modularity (rho=0.01), fusion + screen projections
# Sweep the number of nodes 10 1e4

from cd_v_partition.config import SimulationConfig
from cd_v_partition.experiment import Experiment
from cd_v_partition.vis_experiment import vis_experiment
import numpy as np
import copy

if __name__ == "__main__":
    exp_5 = Experiment(3)
    large_num_nodes = int(1e4)
    dir = "simulations/experiment_5_refactor_multi_algs_sizes"
    sim_cfg = SimulationConfig(
        graph_per_spec=1,
        experiment_id=dir,
        partition_fn=[
            "modularity",
            "edge_cover",
            "expansive_causal",
        ],  #'modularity', 'edge_cover', 'expansive_causal',
        num_samples=[int(1e4)],
        graph_kind="hierarchical",
        num_nodes=[large_num_nodes],
        causal_learn_fn=["GES"],  # ["GES", "PC", "RFCI", "NOTEARS"],
        merge_fn=["screen"],
    )

    sim_cfg_pef = copy.copy(sim_cfg)

    sim_cfg_pef.partition_fn = ["PEF"]
    sim_cfg_pef.merge_fn = ["fusion"]
    sim_cfg_pef.merge_full_cand_set = [True]

    exp_5.run(sim_cfg, random_state=1)
    # exp_5.run(sim_cfg_pef, random_state=1)
    for cd_alg in sim_cfg.causal_learn_fn:
        vis_experiment(
            5,
            dir,
            sim_cfg.partition_fn,
            cd_alg,
            sim_cfg.graph_per_spec,
            "num_nodes",
            sim_cfg.num_nodes,
        )
