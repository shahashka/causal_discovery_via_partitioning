# Experiment 4: two community, scale free, num samples 1e5, num_nodes=50,
# num_trials=30, default modularity (rho=0.01), fusion + screen projections
# Sweep PC alpha parameter which controls
# the density of the supsertructure
from cd_v_partition.config import SimulationConfig
from cd_v_partition.experiment import Experiment
from cd_v_partition.vis_experiment import vis_experiment
import numpy as np
import copy

if __name__ == "__main__":
    exp_4 = Experiment(16)
    dir = "simulations/experiment_4_refactor_multi_algs_new_exps_rfci_tmlr"
    sim_cfg = SimulationConfig(
        graph_per_spec=10,
        experiment_id=dir,
        partition_fn=["no_partition", "modularity", "edge_cover", "expansive_causal"],
        num_samples=[int(1e5)],
        use_pc_algorithm=[True],
        alpha=list(np.arange(0.1, 1, 0.05)),
        graph_kind="scale_free",
        num_nodes=50,
        num_communities=2,
        causal_learn_fn=["RFCI-PAG"],
        merge_fn=["screen"],
        merge_ss_subset_flag=False,
    )

    sim_cfg_pef = copy.copy(sim_cfg)

    sim_cfg_pef.partition_fn = ["PEF"]
    sim_cfg_pef.merge_fn = ["fusion"]
    sim_cfg_pef.merge_full_cand_set = [True]

    exp_4.run(sim_cfg, random_state=1)
    exp_4.run(sim_cfg_pef, random_state=1)
    for cd_alg in sim_cfg.causal_learn_fn:
        vis_experiment(
            4,
            dir,
            sim_cfg.partition_fn + sim_cfg_pef.partition_fn,
            cd_alg,
            sim_cfg.graph_per_spec,
            "alpha",
            sim_cfg.alpha,
        )
