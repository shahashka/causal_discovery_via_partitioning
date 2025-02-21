# Experiment 1: two community, scale free, default rho modularity (0.01),,
# num_nodes=50, num_trials=30, artificial superstructure with 10% extraneous edges,
# fusion + screen projections
# Sweep number of samples

from cd_v_partition.config import SimulationConfig
from cd_v_partition.experiment import Experiment
from cd_v_partition.vis_experiment import vis_experiment
import numpy as np
import copy

if __name__ == "__main__":
    exp_1 = Experiment(2)
    dir = "simulations/experiment_base"
    sim_cfg = SimulationConfig(
        graph_per_spec=1,
        experiment_id=dir,
        partition_fn=["no_partition", "modularity", "edge_cover", "expansive_causal"],
        num_samples=[10**i for i in np.arange(1, 2)],
        graph_kind="scale_free",
        num_nodes=25,
        num_communities=2,
        causal_learn_fn=["RFCI", "PC", "GES", "NOTEARS"],
        merge_fn=["screen"],
    )

    sim_cfg_pef = copy.copy(sim_cfg)

    sim_cfg_pef.partition_fn = ["PEF"]
    sim_cfg_pef.merge_fn = ["fusion"]
    sim_cfg_pef.merge_full_cand_set = [True]

    exp_1.run(sim_cfg, random_state=1)
    exp_1.run(sim_cfg_pef, random_state=1)
    for cd_alg in sim_cfg.causal_learn_fn:
        vis_experiment(
            1,
            dir,
            sim_cfg.partition_fn + sim_cfg_pef.partition_fn,
            cd_alg,
            sim_cfg.graph_per_spec,
            "num_samples",
            sim_cfg.num_samples,
        )
