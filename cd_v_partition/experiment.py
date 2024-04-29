import datetime
import typing as t
from concurrent.futures import as_completed, ProcessPoolExecutor, Future
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tqdm
from numpy.random import RandomState

import cd_v_partition.utils as utils
from cd_v_partition.algorithms.causal_discovery import sp_gies, pc
from cd_v_partition.algorithms.fusion import fusion_basic, fusion, screen_projections
from cd_v_partition.algorithms.partitioning import (
    partition_problem,
    expansive_causal_partition,
    rand_edge_cover_partition,
)
from cd_v_partition.algorithms.typing import PartitioningAlgorithm, FusionAlgorithm
from cd_v_partition.configs.config import Config, Spec
from cd_v_partition.typing import GeneratedGraph, Partition, GraphKind, TrueGraph


def edge_list_to_df(edge_list: t.List[t.Tuple[int, int]]) -> pd.DataFrame:
    """This converts an edgelist to a dataframe.

    Notes:
        The edge list is assumed to be directed.

    Args:
        edge_list (t.List[t.Tuple[int, int]]): The edge list to convert.

    Returns:
        Dataframe for each
    """
    start_nodes = []
    end_nodes = []
    weight = []
    for edge in edge_list:
        start_nodes.append(edge[0])
        end_nodes.append(edge[1])
        try:
            weight.append(
                edge[2]["weight"]
            )  # TODO: If `KeyError` is the issue, use `get()`.
        except (
            KeyError
        ):  # NOTE: I took a guess that `KeyError` is the possible exception here.
            weight.append(1)
    return pd.DataFrame(zip(start_nodes, end_nodes, weight))


def _local_structure_learn(subproblem):
    skel, data = subproblem
    adj_mat = sp_gies(data, skel=skel, outdir=None)
    return adj_mat


class Experiment:
    def __init__(
        self,
        graph_factory,
        structure_learner,
        causal_learner,
        merge_fn,
        workers: int,
    ) -> None:
        self.workers = workers

    # ExecConfig
    def run(
        self,
        exec_cfg: dict[str, Any],
        cfg: Config,
        random_state: RandomState | int | None = None,
    ) -> None:
        if self.workers == 1:
            return self.run_serial(cfg, random_state)
        elif self.workers > 1:
            return self.run_concurrent()
        else:
            raise ValueError("Experiment member `workers` must be >= 1.")

    def run_concurrent(
        self, cfg: Config, random_state: RandomState | int | None = None
    ):
        # with ExecutorFactory.create(exec_cfg) as pool:
        #     futures = []
        #     for spec in cfg:
        #         fut = pool.submit(self.run_simulation, spec)
        #         futures.append(fut)
        #
        # dataframes = []
        # graphs = []
        # for fut in as_completed(futures):
        #     df, g = fut.result()
        #     dataframes.append(df)
        #     graphs.append(g)

        random_state = utils.load_random_state(trial)
        date = datetime.datetime.now()
        with ProcessPoolExecutor(self.workers) as executor:
            futures: dict[Future, tuple[int, int]] = {}
            for trial in range(cfg.graph_per_spec):
                seed = trial
                for i, spec in enumerate(cfg):
                    fut = executor.submit(self.run_simulation, spec, random_state=seed)
                    futures[fut] = (i, trial)

            progressbar = tqdm.tqdm(total=cfg.graph_per_spec * len(cfg))
            for fut in as_completed(futures):
                i, trial = futures[fut]
                result_df = fut.result()
                outdir = Path(f"out/{date}/spec_{i}/trial_{trial}/")
                if not outdir.exists():
                    outdir.mkdir(parents=True)

                result_df.to_feather(
                    outdir / "data_gen_true.feather",
                    header=["node_id", "bias", "var"],
                    index=False,
                )
                spec.to_yaml(outdir / "spec.yaml")
                progressbar.update()

    def run_serial(self, cfg: Config, random_state: RandomState | int | None = None):
        random_state = utils.load_random_state(random_state)
        date = datetime.datetime.now()
        progressbar = tqdm.tqdm(total=cfg.graph_per_spec * len(cfg))
        for i, spec in enumerate(cfg):
            seed = random_state.randint(0, 2**32 - 1)
            for trial in range(cfg.graph_per_spec):
                result_df = self.run_simulation(spec, random_state=seed)

                outdir = Path(f"out/{date}/spec_{i}/trial_{trial}/")
                if not outdir.exists():
                    outdir.mkdir(parents=True)

                result_df.to_feather(
                    outdir / "data_gen_true.feather",
                    header=["node_id", "bias", "var"],
                    index=False,
                )
                spec.to_yaml(outdir / "spec.yaml")
                progressbar.update()

    def run_simulation(
        self, spec: Spec, random_state: RandomState | int | None = None
    ) -> pd.DataFrame:
        random_state = utils.load_random_state(random_state)
        comm_popularity = random_state.dirichlet(
            [spec.comm_pop_alpha] * spec.num_communities
        )
        comm_popularity = (comm_popularity * spec.comm_pop_coeff).astype(int)

        edge_prob = random_state.dirichlet(
            [spec.edge_prob_alpha] * spec.num_communities
        )

        # GENERATE THE GRAPH AND DATA
        gen_graph: GeneratedGraph = Experiment.generate_graph(
            kind=spec.graph_kind,
            num_nodes=spec.num_nodes,
            num_samples=spec.num_samples,
            num_communities=spec.num_communities,
            comm_popularity=comm_popularity,
            edge_prob=edge_prob,
            inter_edge_prob=spec.inter_edge_prob,  # rho
        )

        # GENERATE THE SUPERSTRUCTURE
        if spec.use_pc_algorithm:
            super_struct, _ = pc(
                gen_graph.samples_to_numpy(), alpha=spec.alpha, outdir=..., num_cores=8
            )
        else:
            G_star = utils.edge_to_adj(list(gen_graph.edges), nodes=gen_graph.nodes)
            super_struct = utils.artificial_superstructure(
                G_star,
                frac_retain_direction=spec.frac_retain_direction,  # Leave const at 0.1 by default
                frac_extraneous=spec.extraneous,
            )

        if spec.partition_fn is None:
            out_adj = sp_gies(gen_graph.samples, skel=super_struct, outdir=None)
            num_nodes = spec.num_nodes
            edge_list = utils.adj_to_edge(out_adj, np.arange(num_nodes))
            df = edge_list_to_df(edge_list)
            df.to_csv(
                "./examples/edges_serial.csv",
                header=["node1", "node2", "weight"],
                index=False,
            )
        else:
            partition_alg = Experiment.get_partitioning_alg(spec.partition_fn)
            partition = partition_alg(super_struct)
            causal_partition = expansive_causal_partition(
                super_struct, partition
            )  # adapt the modularity partition
            edge_cover_partition = rand_edge_cover_partition(
                super_struct, partition
            )  # adapt the modularity partition randomly

            partition_schemes = {
                "default": gen_graph.default_partition,  # Fallback partition for analysis.
                "input_disjoint_partition": partition,  # comes from the Spec (PARAM)
                "causal": causal_partition,  # causal (FIXED)
                "edge_cover": edge_cover_partition,  # random edge covering (FIXED)
            }

            for part_name, partition in partition_schemes.items():
                self.single_partition(
                    spec=spec,
                    gen_graph=gen_graph,
                    partition=partition,
                    partition_name=part_name,
                    super_structure=super_struct,
                    A_X_v=None,
                )

        # Save
        edge_list_to_df(gen_graph.edges).to_csv(
            "./examples/edges_true.csv",
            header=["node1", "node2"],
            index=False,
        )

        tmp_data = list(
            zip(np.arange(len(gen_graph.bias)), gen_graph.bias, gen_graph.variance)
        )
        return pd.DataFrame(tmp_data)

    def single_partition(
        self,
        spec: Spec,
        gen_graph: GeneratedGraph,
        partition: Partition,
        partition_name: str,
        super_structure: np.ndarray,
        A_X_v: np.ndarray | None = None,
        # TODO: Add an outdir as an argument to simplify the outputting of data.
    ):
        print(partition_name)
        print(partition)
        subproblems = partition_problem(partition, super_structure, gen_graph.samples)
        # Visualize the partition
        superstructure_net = utils.adj_to_dag(
            super_structure
        )  # undirected edges in superstructure adjacency become bidirected
        utils.evaluate_partition(partition, superstructure_net, gen_graph.nodes)
        utils.create_partition_plot(
            superstructure_net,
            gen_graph.nodes,
            partition,
            f"./examples/{partition_name}_partition.png",
        )
        # Call the causal learner on subsets of the data F({A(X_s)}) and sub-structures
        num_partitions = 2
        nthreads = 2  # each thread handles one partition
        results: list[np.ndarray] = []
        chunksize = max(1, num_partitions // nthreads)

        with ProcessPoolExecutor(max_workers=nthreads) as executor:
            for result in executor.map(
                _local_structure_learn,
                spec.causal_learn_fn,
                subproblems,
                chunksize=chunksize,
            ):
                results.append(result)

        # Merge the subset learned graphs
        graph_star = utils.edge_to_adj(list(gen_graph.edges), nodes=gen_graph.nodes)
        fusion_algorithm = Experiment.get_fusion_alg(spec)
        fused_A_X_s = fusion_algorithm(partition, results, gen_graph.samples_to_numpy())
        utils.delta_causality(A_X_v, fused_A_X_s, graph_star)

        # Save
        edge_list_to_df(list(fused_A_X_s.edges(data=True))).to_csv(
            f"./examples/edges_{partition_name}_partition.csv",
            header=["node1", "node2", "weight"],
            index=False,
        )
        pd.DataFrame(list(zip(partition.keys(), partition.values()))).to_csv(
            f"./examples/{partition_name}_partition.csv",
            header=["comm id", "node list"],
            index=False,
        )

    @staticmethod
    def generate_graph(
        kind: GraphKind,
        num_nodes: int,
        num_samples: int,
        num_communities: int,
        comm_popularity: list[int],
        edge_prob: list[float],
        inter_edge_prob: float,
        random_state: RandomState | int | None = None,
    ) -> GeneratedGraph:
        random_state = utils.load_random_state(random_state)
        default_partition, comm_graph = utils.create_k_comms(
            graph_type=kind,
            n=int(num_nodes / num_communities),
            m_list=comm_popularity,
            p_list=edge_prob,
            k=num_communities,
            rho=inter_edge_prob,
            # TODO: Add `RandomState` here.
            random_state=random_state,
        )

        # Generate a random network and corresponding dataset
        (edges, nodes, bias, var), samples = utils.get_data_from_graph(
            list(np.arange(num_nodes)),
            list(comm_graph.edges()),
            nsamples=int(num_samples),
            iv_samples=0,
            bias=None,
            var=None,
        )

        return GeneratedGraph(
            true_graph=TrueGraph(nodes, edges),
            samples=samples,
            default_partition=default_partition,
            bias=bias,
            variance=var,
        )

    # TODO
    @staticmethod
    def get_partitioning_alg(self, spec: Spec, code: str) -> PartitioningAlgorithm:
        match spec.partition_fn:
            case "...":
                pass
            case ",,,":
                pass
        pass

    @classmethod
    def get_fusion_alg(cls, spec: Spec) -> FusionAlgorithm:
        match spec.fusion_fn:
            case "basic":
                return fusion_basic
            case "fusion":
                return fusion
            case "screen":
                return screen_projections
            case _:
                raise ValueError(f"`{spec.fusion_fn=}` is an illegal value.`")
