# Experiment 6: Synthetic e.coli networks, num samples 1e5, artificial ss with frac_extraneoues = 0.5
# Modularity partitioning
from cd_v_partition.config import SimulationConfig
from cd_v_partition.experiment import Experiment
from cd_v_partition.vis_experiment import vis_experiment
import copy
import os

if __name__ == "__main__":
    exp_6 = Experiment(16)
    dir = "simulations/experiment_6_refactor_multi_algs_net_2"
    ecoli_data_dir = "./datasets/bionetworks/ecoli/synthetic_copies"
    sim_cfg = SimulationConfig(
        graph_per_spec=1,
        experiment_id=dir,
        partition_fn=["no_partition", "modularity", "edge_cover", "expansive_causal"],
        num_samples=[int(1e4)],
        graph_kind="ecoli",
        graph_load_path=[f"{ecoli_data_dir}/net_{i}.txt" for i in range(2, 3)],
        causal_learn_fn=["PC", "RFCI", "GES", "NOTEARS"],
        merge_fn=["screen"],
    )

    sim_cfg_pef = copy.copy(sim_cfg)

    sim_cfg_pef.partition_fn = ["PEF"]
    sim_cfg_pef.merge_fn = ["fusion"]
    sim_cfg_pef.merge_full_cand_set = [True]

    exp_6.run(sim_cfg, random_state=1)
    exp_6.run(sim_cfg_pef, random_state=1)
    for cd_alg in sim_cfg.causal_learn_fn:
        vis_experiment(
            6,
            dir,
            sim_cfg.partition_fn + sim_cfg_pef.partition_fn,
            cd_alg,
            sim_cfg.graph_per_spec,
            "ecoli_net",
            [os.path.basename(net) for net in sim_cfg.graph_kind],
        )
