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
    exp_5 = Experiment(1)
    size_of_comm = 20
    num_communities = 500
    large_num_nodes = int(num_communities * size_of_comm)

    dir = "simulations/experiment_5_refactor_multi_algs_small_comms_20"
    sim_cfg = SimulationConfig(
        graph_per_spec=1,
        experiment_id=dir,
        partition_fn=[
            "modularity"
        ],  # , 'edge_cover', 'expansive_causal'],#, 'no_partition'],
        num_samples=[int(1e4)],
        graph_kind="erdos_renyi",
        num_nodes=[large_num_nodes],
        causal_learn_fn=["RFCI-PAG"],  # [ "NOTEARS", "GES", "PC","RFCI-PAG"],
        merge_fn=["screen"],
        partition_resolution=5,
        num_communities=num_communities,
        partition_best_n=num_communities,
        partition_cutoff=num_communities,
        comm_edge_prob=[num_communities * [0.5]],
        comm_pop=[num_communities * [1]],
        inter_edge_prob=[1e-10],
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
