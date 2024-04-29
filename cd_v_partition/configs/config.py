import itertools
from pathlib import Path
from typing import Any, Iterator, Literal

from omegaconf import OmegaConf, MISSING

from dataclasses import dataclass, field

ExecutorKind = Literal["parsl", "process", "thread"]


@dataclass
class ExecutorConfig:
    kind: ExecutorKind


@dataclass
class Spec:
    graph_kind: str = MISSING
    num_nodes: int = MISSING
    num_samples: int = MISSING
    edge_params: int = MISSING
    causal_learn_fn: str = MISSING
    partition_fn: str = MISSING
    fusion_fn: str = MISSING
    inter_edge_prob: float = MISSING
    # Superstructure parameters
    alpha: float = MISSING
    """..."""

    # TODO: Parameters that NEED to be integrated.
    full_cand_set: bool = MISSING
    edge_prob_alpha: float = MISSING
    comm_pop_alpha: float = MISSING
    comm_pop_coeff: float = MISSING
    num_communities: int = MISSING
    extraneous: float = MISSING
    frac_retain_direction: float = MISSING
    use_pc_algorithm: bool = MISSING

    def to_yaml(self, outfile: Path | str) -> None:
        cfg = OmegaConf.structured(self)
        OmegaConf.save(cfg, outfile)


@dataclass
class Config:
    # Parameters only used to initialize the executor used to launch jobs across compute.
    graph_per_spec: int
    executor_kind: ExecutorKind = MISSING
    executor_args: dict[str, Any] = MISSING

    # Parameters included in a ``Spec`` instance.
    graph_kind: list[str] = field(default_factory=lambda: ["random"])
    num_nodes: list[int] = MISSING
    edge_params: list[int] = MISSING
    causal_learn_fn: list[str] = field(defaulf_factory=lambda: ["SPGIES"])  # default to SPGIES
    partition_fn: list[str] = MISSING  # default to Adela's modularity partition
    fusion_fn: list[str] = MISSING  # default to fusion

    # TODO: Parameters that NEED to be integrated.
    full_cand_set: list[bool] = MISSING
    edge_prob_alpha: list[float] = MISSING
    comm_pop_alpha: list[float] = MISSING
    comm_pop_coeff: list[float] = MISSING
    num_communities: list[int] = MISSING
    extraneous: list[float] = MISSING

    def __iter__(self) -> Iterator[Spec]:
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
            yield Spec(**spec)

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
    def from_yaml(cls, path: Path | str) -> "Config":
        """Reads a `.yaml` file to instantiate a ``Config`` object.

        Args:
            path (Path | str): Path of the `.yaml` file.

        Returns:
            An instance of ``Config``.
        """
        data = OmegaConf.load(path)
        cfg = OmegaConf.structured(cls(**data))
        return OmegaConf.to_object(cfg)
