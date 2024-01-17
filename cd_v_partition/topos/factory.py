import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, Literal, TypeAlias

import causaldag
import graphical_models.rand
import networkx as nx
import numpy as np
import pandas as pd
from numpy.random import RandomState

from cd_v_partition.utils import load_random_state

ArcList: TypeAlias = set[tuple[Hashable, Hashable]]
NodeList: TypeAlias = list[Hashable]
GraphKind = Literal["barabasi_albert", "erdos_renyi", "watts_strogatz"]


@dataclass
class GeneratedGraph:
    edge_ids: list[int]
    node_ids: list[int]
    bias: float
    variance: float
    adj_matrix: Any


class GraphFactory:
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes

    def create_graph(
        self, kind: GraphKind, random_state: RandomState | int | None, **kwargs
    ) -> graphical_models.DAG:
        """
        Create a DAG topology based on one of the 3 standard random graph models:
        Barabasi-Albert, Erdos-Renyi, and Watts Strogatz for scale-free, "truly" random,
        and small-world graphs (respectively).

        Args:
            kind (GraphKind): Which random graph model to use to generate the DAG topology.
            random_state (RandomState | int | None): RandomState for underlying random number generators.
            **kwargs: Additional kw-args for the random graph model.

        Returns:

        """
        random_state = load_random_state(random_state)
        seed = random_state.randint(low=0, high=2**32 - 1)

        k, m, p = 5, 3, 0.1  # FIXME

        match kind:
            # TODO: Incorporate the **kwargs here for each of the random models.
            case "barabasi_albert":
                dag = graphical_models.rand.directed_random_graph(
                    self.num_nodes,
                    random_graph_model=lambda n: nx.barabasi_albert_graph(
                        n, m=m, seed=seed
                    ),
                )
            case "erdos_renyi":
                dag = graphical_models.rand.directed_random_graph(
                    self.num_nodes,
                    random_graph_model=lambda n: nx.erdos_renyi_graph(
                        n, p=p, seed=seed
                    ),
                )
            case "watts_strogatz":
                dag = graphical_models.rand.directed_random_graph(
                    self.num_nodes,
                    random_graph_model=lambda n: nx.watts_strogatz_graph(
                        n, k=k, p=p, seed=seed
                    ),
                )
            case _:
                raise ValueError(
                    f"Illegal value for `kind` (`GraphKind`). Must be one of {GraphKind}."
                )

        return dag

    def create_observational_data(
        self,
        kind: GraphKind,
        num_observations: int,
        num_interventions: int,
        outdir: Path | str | None = None,
        random_state: RandomState | int | None = None,
    ) -> tuple[tuple[ArcList, NodeList, np.ndarray, np.ndarray], pd.DataFrame]:
        random_state = load_random_state(random_state)

        dag = self.create_graph(kind, random_state)
        node_ids = list(dag.nodes)
        bias = random_state.normal(0, 1, size=len(node_ids))
        variance = random_state.normal(0, 1, size=len(node_ids))
        variance = np.abs(variance)

        bn = graphical_models.GaussDAG(
            nodes=node_ids, arcs=dag.arcs, biases=bias, variances=variance
        )
        data = bn.sample(num_observations)
        df = pd.DataFrame(data=data, columns=node_ids)
        df["age"] = 0

        if num_interventions > 0:
            intervention_samples = []
            for idx, node in enumerate(node_ids):
                samples = bn.sample_interventional(
                    causaldag.Intervention(
                        {node: causaldag.ConstantIntervention(val=0)}
                    ),
                    num_interventions,
                )
                samples = pd.DataFrame(samples, columns=node_ids)
                samples["target"] = idx + 1
                intervention_samples.append(samples)
            df_intervention = pd.concat(intervention_samples)
            df = pd.concat([df, df_intervention])

        if outdir is not None:
            outdir = Path(outdir) if isinstance(outdir, str) else outdir
            if not outdir.is_dir():
                os.makedirs(outdir)
            df.to_csv(outdir / "data.csv", index=False)
            with open(outdir / "ground.txt", "w") as f:
                for arc in dag.arcs:
                    f.write(f"{arc}\n")

        return (dag.arcs, node_ids, bias, variance), df
