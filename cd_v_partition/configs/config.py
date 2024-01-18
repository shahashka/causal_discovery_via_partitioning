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
class SimulationSpec:
    graph_kind: str = MISSING
    num_nodes: int = MISSING
    edge_params: int = MISSING
    causal_learn_fn: str = MISSING
    partition_fn: str = MISSING
    merge_fn: str = MISSING


@dataclass
class SimulationConfig:
    # Parameters only used to initialize the executor used to launch jobs across compute.
    executor_kind: ExecutorKind = MISSING
    executor_args: dict[str, Any] = MISSING

    # Parameters included in a ``SimulationSpec`` instance.
    graph_kind: list[str] = MISSING
    num_nodes: list[int] = MISSING
    edge_params: list[int] = MISSING
    causal_learn_fn: list[str] = MISSING  # default to SPGIES
    partition_fn: list[str] = MISSING  # default to Adela's modularity partition
    merge_fn: list[str] = MISSING  # default to fusion

    def __iter__(self) -> Iterator[SimulationSpec]:
        """
        Iterates through all the combinations of iterable items in config (see ``itertools.product``).

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
