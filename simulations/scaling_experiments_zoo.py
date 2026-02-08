from parsl.app.app import python_app
from parsl_setup import get_parsl_config, get_parsl_config_debug
import parsl
import sys
import os
@python_app
def run_experiment(worker_id, cd, p, repeat, seed, outdir): 
    import sys
    # caution: path[0] is reserved for script path (or '' in REPL)
    sys.path.insert(1, '/eagle/projects/FoundEpidem/shahashka/causal_discovery_via_partitioning/')
    from cd_v_partition.config import SimulationConfig
    from cd_v_partition.experiment import Experiment
    from cd_v_partition.vis_experiment import vis_experiment
    import numpy as np
    import copy
    import warnings
    import logging
    warnings.simplefilter("ignore") 
    logger = logging.getLogger(__name__)
    #outdir="/tmp"
    logging.basicConfig(
        filename=f"{outdir}/worker_{worker_id}.stderr",  # Specify the log file name
        level=logging.DEBUG,  # Set the logging level (e.g., INFO, DEBUG, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(levelname)s - %(message)s' # Define the log message format
    )

    logger.debug(f"Run experiment {p}, {cd}, {r}")

    exp = Experiment(1)
    
    sim_cfg = SimulationConfig(
        graph_per_spec=1,
        experiment_id=outdir,
        partition_fn=[ "no_partition"], 
        num_samples=[(int)(10*p)],
        graph_kind="hierarchical",
        num_nodes=[p],
        causal_learn_fn=[cd],
    )
    exp.run(sim_cfg, random_state=seed)

if __name__ == '__main__':
    #config = get_parsl_config_debug()
    config = get_parsl_config()
    # NOTE(MS): this is how you vary arg inputs into your expeirment
    args = []
    num_nodes = [10,100,1000,10000]
    #num_nodes=[10]
    cd_methods = ["GES", "PC", "NOTEARS"]
    num_repeats = 1
    worker_id = 0
    for r in range(num_repeats):
        for p in num_nodes:
            for cd in cd_methods:
                outdir = f"/eagle/projects/FoundEpidem/shahashka/causal_discovery_via_partitioning/{cd}_{p}_{r}"
                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                args.append({"worker_id":worker_id, "cd":cd, "p":p, "seed":p, "repeat":r, "outdir":outdir})
                worker_id += 1
                print(worker_id)

    with parsl.load(config):
        # Launch experiments as parsl tasks
        futures = [run_experiment(**arg) for arg in args]

        # Wait for tasks to return
        for future in futures:
            print(f'Waiting for {future}', file=sys.stderr)
            print(f'Got result {future.result()}',  file=sys.stderr)







