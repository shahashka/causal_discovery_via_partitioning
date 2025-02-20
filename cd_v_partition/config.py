import itertools
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from omegaconf import MISSING, OmegaConf

ExecutorKind = Literal["parsl", "process", "thread"]


@dataclass
class ExecutorConfig:
    kind: ExecutorKind


@dataclass
class SimulationSpec:
    # Graph params
    graph_kind: str = MISSING
    graph_load_path: str = MISSING
    num_nodes: int = MISSING
    num_samples: int = MISSING
    inter_edge_prob: float = MISSING
    comm_edge_prob: list[float] = MISSING
    comm_pop: list[float] = MISSING
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
    causal_learn_use_skel: bool = MISSING

    # Superstructure params
    alpha: float = MISSING
    frac_retain_direction: float = MISSING
    frac_extraneous: float = MISSING
    use_pc_algorithm: bool = MISSING
    use_corr_mat: bool = MISSING

    @classmethod
    def to_yaml(self, outfile: Path | str) -> None:
        """
        Saves spec to yaml file.

        Args:
            outfile (Path | str): The path for where to save the yaml file.
        """
        spec = OmegaConf.structured(self)
        OmegaConf.save(spec, outfile)


@dataclass
class SimulationConfig:
    # Parameters only used to initialize the executor used to launch jobs
    # across compute.
    executor_kind: ExecutorKind = MISSING
    executor_args: dict[str, Any] = MISSING
    graph_per_spec: int = 1

    # These are the partitioning functions, all need to be run for each spec
    # eval_algorithms: list[str] = MISSING
    experiment_id: str = MISSING

    # Parameters included in a ``SimulationSpec`` instance.

    # Graph params
    graph_kind: list[str] = field(default_factory=lambda: ["scale_free"])
    graph_load_path: list[str] = field(default_factory=lambda: [None])
    num_nodes: list[int] = field(default_factory=lambda: [25])
    num_samples: list[int] = field(default_factory=lambda: [int(10**4)])
    inter_edge_prob: list[float] = field(default_factory=lambda: [0.01])
    comm_edge_prob: list[list[float]] = field(
        default_factory=lambda: [[0.5, 0.5]],
    )
    comm_pop: list[list[float]] = field(default_factory=lambda: [[1, 2]])
    num_communities: int = field(default_factory=lambda: [2])

    # Partition params
    partition_fn: list[str] = field(default_factory=lambda: ["modularity"])
    partition_cutoff: list[int] = field(default_factory=lambda: [1])
    partition_best_n: list[int] = field(default_factory=lambda: [None])
    partition_resolution: list[int] = field(default_factory=lambda: [1])

    # Merge params
    merge_fn: list[str] = field(default_factory=lambda: ["screen"])
    merge_ss_subset_flag: list[bool] = field(default_factory=lambda: [True])
    merge_finite_sample_flag: list[bool] = field(
        default_factory=lambda: [False],
    )
    merge_full_cand_set: list[bool] = field(default_factory=lambda: [False])

    # CD learn params
    causal_learn_fn: list[str] = field(default_factory=lambda: ["GES"])
    causal_learn_use_skel: list[bool] = field(default_factory=lambda: ["True"])

    # Superstructure params
    alpha: list[float] = field(default_factory=lambda: [0.1])
    frac_retain_direction: list[float] = field(default_factory=lambda: [0.1])
    frac_extraneous: list[float] = field(default_factory=lambda: [0.1])
    use_pc_algorithm: list[bool] = field(default_factory=lambda: [False])
    use_corr_mat: list[bool] = field(default_factory=lambda: [False])

    def __iter__(self) -> Iterator[SimulationSpec]:
        """
        Iterates through all the combinations of iterable items in config
        (see ``itertools.product``).

        Returns:
            Iterator item.
        """
        d = vars(self)

        lists = {k: v for k, v in d.items() if isinstance(v, list)}
        consts = {k: v for k, v in d.items() if not isinstance(v, list)}

        list_keys, list_values = zip(*lists.items())

        spec_iter = []
        for p in itertools.product(*list_values):
            spec_iter.append(dict(zip(list_keys, p)))

        for spec in spec_iter:
            spec.update(**consts)
            del spec["executor_kind"]
            del spec["executor_args"]
            del spec["graph_per_spec"]
            del spec["experiment_id"]
            yield SimulationSpec(**spec)

    def __len__(self) -> int:
        return len(list(iter(self)))

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
