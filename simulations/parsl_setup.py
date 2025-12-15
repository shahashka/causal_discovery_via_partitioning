"""setting up Parsl utilities for experiment parallelization."""

from __future__ import annotations

from parsl.config import Config
from parsl.executors import HighThroughputExecutor
#from parsl.executors.ipp import IPyParallelExecutor
from parsl.launchers import MpiExecLauncher
from parsl.addresses import address_by_interface, address_by_hostname
from parsl.providers import PBSProProvider
from parsl.monitoring.monitoring import MonitoringHub
from parsl.launchers import SingleNodeLauncher
from parsl.providers import LocalProvider
from parsl.executors.high_throughput.executor import DEFAULT_LAUNCH_CMD
CONTAINERIZED_LAUNCH_CMD = "apptainer run --bind /eagle/projects/FoundEpidem/shahashka /eagle/projects/FoundEpidem/shahashka/causal_discovery_via_partitioning_main.sif " + DEFAULT_LAUNCH_CMD
def get_parsl_config() -> Config:
    """Initialize Parsl config.

    One experiment per GPU.
    Multiple experiment per node.
    """

    # NOTE(MS): replace these
    #env = "/eagle/projects/argonne_tpc/mansisak/ci-nn/env/"
    #run_dir = "/eagle/projects/argonne_tpc/mansisak/ci-nn/"
    env = "/eagle/projects/FoundEpidem/shahashka/env"
    run_dir = "/eagle/projects/FoundEpidem/shahashka/causal_discovery_via_partitioning/"
    user_opts = {
        "worker_init": f"""
module use /soft/modulefiles
module load conda
cd {run_dir} 
conda activate causal_discovery
# Print to stdout to for easier debugging
module list
nvidia-smi
which python
hostname
pwd""",
        "scheduler_options": "#PBS -l filesystems=home:eagle:grand",  # specify any PBS options here, like filesystems
        "account": "FoundEpidem",
        "queue": "preemptable",  # e.g.: "prod","debug, "preemptable", "debug-scaling" (see https://docs.alcf.anl.gov/polaris/running-jobs/)
        "walltime": "24:00:00", #HH:MM:SS
        "nodes_per_block": 3,  # think of a block as one job on polaris, so to run on the main queues, set this >= 10
        "available_accelerators": 4,  # Each Polaris node has 4 GPUs, setting this ensures one worker per GPU
    }
    provider=PBSProProvider(
                    launcher=MpiExecLauncher(
                        bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1"
                    ),
                    account=user_opts["account"],
                    queue=user_opts["queue"],
                    select_options="ngpus=4",
                    # PBS directives (header lines): for array jobs pass '-J' option
                    scheduler_options=user_opts["scheduler_options"],
                    # Command to be run before starting a worker, such as:
                    worker_init=user_opts["worker_init"],
                    # number of compute nodes allocated for each block
                    nodes_per_block=user_opts["nodes_per_block"],
                    init_blocks=1,
                    min_blocks=0,#0,
                    max_blocks=1,  # Can increase more to have more parallel jobs
                    # cpus_per_node=user_opts["cpus_per_node"],
                    walltime=user_opts["walltime"],
                )

    config = Config(
        executors=[
            HighThroughputExecutor(
                label='ci_results',
                available_accelerators=4,  # number of GPUs
                max_workers_per_node=4,
                provider=provider,
                #address=address_by_interface("bond0"),
                cpu_affinity="block-reverse",
                prefetch_capacity=0,
                launch_cmd=CONTAINERIZED_LAUNCH_CMD
            )
        ],
        monitoring=MonitoringHub(
            hub_address=address_by_hostname(),
            monitoring_debug=False,
            resource_monitoring_interval=10,
              ),
    )
    return config

def get_parsl_config_debug() -> Config:
    """Initialize Parsl config.

    One experiment per GPU.
    Multiple experiment per node.
    """

    # NOTE(MS): replace these
    #env = "/eagle/projects/argonne_tpc/mansisak/ci-nn/env/"
    #run_dir = "/eagle/projects/argonne_tpc/mansisak/ci-nn/"
    #env = "/eagle/projects/FoundEpidem/shahashka/env"
    run_dir = "/eagle/projects/FoundEpidem/shahashka/causal_discovery_via_partitioning/"
    user_opts = {
        "worker_init": f"""
module use /soft/modulefiles
module load conda
#conda activate causal_discovery
cd {run_dir} 
# Print to stdout to for easier debugging
module list
nvidia-smi
which python
hostname
pwd""",
        "scheduler_options": "#PBS -l filesystems=home:eagle:grand",  # specify any PBS options here, like filesystems
        "account": "FoundEpidem",
        "queue": "debug-scaling",  # e.g.: "prod","debug, "preemptable", "debug-scaling" (see https://docs.alcf.anl.gov/polaris/running-jobs/)
        "walltime": "00:10:00", #HH:MM:SS
        "nodes_per_block": 4,  # think of a block as one job on polaris, so to run on the main queues, set this >= 10
        "available_accelerators": 4,  # Each Polaris node has 4 GPUs, setting this ensures one worker per GPU
    
    }
    config = Config(
        executors=[
            HighThroughputExecutor(
            #IPyParallelExecutor(
            label='benchmark_scale',
            available_accelerators=4,  # number of GPUs
            max_workers_per_node=4,
            #container_image='/eagle/projects/FoundEpidem/shahashka/causal_discovery_via_partitioning_main.sif',
            provider=LocalProvider(
                    init_blocks=1,
                    max_blocks=1,
                    launcher=SingleNodeLauncher(), 
                    worker_init=user_opts['worker_init']
                ),
            cpu_affinity="block-reverse",
            launch_cmd=CONTAINERIZED_LAUNCH_CMD
                )
	],
    monitoring=MonitoringHub(
        hub_address=address_by_hostname(),
        monitoring_debug=False,
        resource_monitoring_interval=10,
    ),
)
    return config
