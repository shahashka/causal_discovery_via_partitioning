import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal
from omegaconf import OmegaConf, MISSING

ExecutorKind = Literal["parsl", "process", "thread"]


@dataclass
class ExecutorConfig:
    kind: ExecutorKind


@dataclass
# TODO update specs from Nathaniel's branch
class SimulationSpec:
    # Graph params
    graph_kind: str = MISSING
    num_nodes: int = MISSING
    num_samples: int = MISSING
    inter_edge_prob: float = MISSING
    edge_prob_alpha: float = MISSING
    comm_pop_alpha: float = MISSING
    comm_pop_coeff: float = MISSING
    num_communities: int = MISSING
    
    # Partition params
    partition_fn: str = MISSING
    partition_cutoff: int = MISSING 
    partition_best_n: int = MISSING
    partition_resolution: int = MISSING
    
    # Merge params
    merge_fn: str = MISSING
    merge_ss_subset_flag: bool = MISSING
    merge_finite_sample_flag: bool = MISSING 
    merge_full_cand_set: bool = MISSING
    
    # CD learn params
    causal_learn_fn: str = MISSING
    
    # Superstructure params
    alpha: float = MISSING
    frac_retain_direction: float = MISSING
    frac_extraneous: float = MISSING
    use_pc_algorithm: bool = MISSING


@dataclass
class SimulationConfig:
    # Parameters only used to initialize the executor used to launch jobs across compute.
    executor_kind: ExecutorKind = MISSING
    executor_args: dict[str, Any] = MISSING
    graph_per_spec: int = MISSING
    eval_algorithms: list[str] = MISSING # These are the partitioning functions, all need to be run for each spec
    experiment_id: str = MISSING
    sweep_param: str = MISSING
    sweep_values:list[float | int] = MISSING
    
    # Parameters included in a ``SimulationSpec`` instance.
    
    # Graph params
    graph_kind: str = MISSING
    num_nodes: int = MISSING
    num_samples: int = MISSING
    inter_edge_prob: float = MISSING
    edge_prob_alpha: float = MISSING
    comm_pop_alpha: float = MISSING
    comm_pop_coeff: float = MISSING
    num_communities: int = MISSING
    
    # Partition params
    partition_fn: str = MISSING
    partition_cutoff: int = MISSING
    partition_best_n: int = MISSING
    partition_resolution: int = MISSING
    
    # Merge params
    merge_fn: str = MISSING
    merge_ss_subset_flag: bool = MISSING
    merge_finite_sample_flag: bool = MISSING 
    merge_full_cand_set: bool = MISSING
    
    # CD learn params
    causal_learn_fn: str = MISSING
    
    # Superstructure params
    alpha: float = MISSING
    frac_retain_direction: float = MISSING
    frac_extraneous: float = MISSING
    use_pc_algorithm: bool = MISSING

    def __iter__(self) -> Iterator[SimulationSpec]:
        """
        Iterates through all the combinations of iterable items in config
        (see ``itertools.product``).

        Returns:
            Iterator item.
        """
        d = vars(self)

        lists = {key: val for key, val in d.items() if isinstance(val, list)}
        consts = {key: val for key, val in d.items() if not isinstance(val, list)}

        list_keys, list_values = zip(*lists.items())
        spec_iter = [dict(zip(list_keys, p)) for p in itertools.product(*list_values)]
        for spec in spec_iter:
            spec.update(**consts)
            del spec["executor_cfg"]
            # TODO: Double-check to make sure that this ^^ only deletes data
            #       for the `Spec` and not the entire `Config.
            yield SimulationSpec(**spec)

    def to_yaml(self, outfile: Path | str) -> None:
        """
        Saves config to yaml file.

        Args:
            outfile (Path | str): The path for where to save the yaml file.
        """
        cfg = OmegaConf.structured(self)
        OmegaConf.save(cfg, outfile)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "SimulationConfig":
        """Reads a `.yaml` file to instantiate a ``SimulationConfig`` object.

        Args:
            path (Path | str): Path of the `.yaml` file.

        Returns:
            An instance of ``SimulationConfig``.
        """
        data = OmegaConf.load(path)
        cfg = OmegaConf.structured(cls(**data))
        return OmegaConf.to_object(cfg)
