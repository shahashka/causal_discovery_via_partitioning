from concurrent.futures import as_completed, ProcessPoolExecutor

import numpy as np
import pandas as pd

import cd_v_partition.utils as utils
from cd_v_partition.algorithms.causal_discovery import sp_gies
from cd_v_partition.algorithms.fusion import fusion
from cd_v_partition.algorithms.partitioning import partition_problem
from cd_v_partition.configs.config import Config, Spec
from cd_v_partition.typing import GeneratedGraph, Partition


def edge_list_to_df(edge_list):
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


class Simulation:
    def __init__(
        self,
        graph_factory,
        structure_learner,
        causal_learner,
        merge_fn,
    ) -> None:
        pass

    def run(self, exec_cfg: ExecConfig, cfg: Config) -> None:
        with ExecutorFactory.create(exec_cfg) as pool:
            futures = []
            for spec in cfg:
                fut = pool.submit(self.run_simulation, spec)
                futures.append(fut)

        dataframes = []
        graphs = []
        for fut in as_completed(futures):
            df, g = fut.result()
            dataframes.append(df)
            graphs.append(g)

    def run_simulation(self, spec: Spec):
        comm_popularity = random_state.dirichlet(
            [spec.comm_pop_alpha] * spec.num_communities
        )
        comm_popularity = (comm_popularity * spec.comm_pop_coeff).astype(int)

        edge_prob = random_state.dirichlet(
            [spec.edge_prob_alpha] * spec.num_communities
        )

        # GENERATE THE GRAPH AND DATA
        # true_graph: tuple[list[Edge], list[Node]]
        # obs_samples: pd.DataFrame
        # default_partition: dict[int, set[Node]]
        # (
        #     true_graph,
        #     obs_samples,
        #     default_partition,
        #     bias,
        #     variance,
        # )
        gen_graph: GeneratedGraph = self.generate_graph(
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
            super_struct, _ = pc(...)
        else:
            super_struct = artificial_superstructure(...)

        if spec.partition_fn is None:
            out_adj = sp_gies(df, skel=super_struct, outdir=None)
        else:
            partition_alg = self.get_partitioning_alg(spec.partition_fn)
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
                    gen_graph,
                    partition,
                    part_name,
                    super_struct,
                    A_X_v=None,
                )

        # Save
        edge_list_to_df(utils.adj_to_edge(A_X_v, np.arange(num_nodes))).to_csv(
            "./examples/edges_serial.csv",
            header=["node1", "node2", "weight"],
            index=False,
        )
        edge_list_to_df(utils.adj_to_edge(G_star, np.arange(num_nodes))).to_csv(
            "./examples/edges_true.csv",
            header=["node1", "node2", "weight"],
            index=False,
        )
        pd.DataFrame(list(zip(np.arange(len(bias)), bias, var))).to_csv(
            "./examples/data_gen_true.csv",
            header=["node id", "bias", "var"],
            index=False,
        )

    def _run_simulation(self, spec: Spec):
        # true_graph = GraphFactory.create_graph()
        # superstructure = self.structure_learner.learn(true_graph)
        # parts = self.partition(superstructure)
        #
        # learned_structs = {}
        # for i, part in enumerate(parts):
        #     learned_structs[i] = self.causal_learner(part)
        #
        # predicted_graph = self.merge(learned_structs)
        # self.evaluate(predicted_graph)
        num_nodes = spec.num_nodes
        num_samples = spec.num_samples
        alpha = 0.5
        num_comms = 2

        hard_partition, comm_graph = utils.create_k_comms(
            graph_type="scale_free",
            n=int(num_nodes / num_comms),
            m_list=[1, 2],  # NOTE: community popularity
            p_list=num_comms * [0.2],
            k=num_comms,
            tune_mod=1,
        )
        # Generate a random network and corresponding dataset
        (edges, nodes, bias, var), df = utils.get_data_from_graph(
            list(np.arange(num_nodes)),
            list(comm_graph.edges()),
            nsamples=int(num_samples),
            iv_samples=0,
            bias=None,
            var=None,
        )
        G_star = utils.edge_to_adj(list(edges), nodes=nodes)

        # Find the 'superstructure'
        df_obs = df.drop(columns=["target"])
        data_obs = df_obs.to_numpy()
        superstructure, p_values = pc(data_obs, alpha=alpha, outdir=None)
        print("Found superstructure")

        # Call the causal learner on the full data A(X_v) and superstructure
        A_X_v = sp_gies(df, skel=superstructure, outdir=None)

        # Partition the superstructure and the dataset
        mod_partition = modularity_partition(superstructure)
        causal_partition = expansive_causal_partition(
            superstructure, mod_partition
        )  # adapt the modularity partition
        edge_cover_partition = rand_edge_cover_partition(
            superstructure, mod_partition
        )  # adapt the modularity partition randomly

        partition_schemes = {
            # "default": hard_partition,
            "input_disjoint_partition": mod_partition,  # comes from the Spec (PARAM)
            "causal": causal_partition,  # causal (FIXED)
            "edge_cover": edge_cover_partition,  # random edge covering (FIXED)
        }

        # For each partition scheme run parallel causal discovery
        for name, partition in partition_schemes.items():
            self.solve_single_partition(
                G_star,
                data_obs,
                df,
                name,
                nodes,
                partition,
                superstructure,
                A_X_v=A_X_v,
            )

        # Save
        edge_list_to_df(utils.adj_to_edge(A_X_v, np.arange(num_nodes))).to_csv(
            "./examples/edges_serial.csv",
            header=["node1", "node2", "weight"],
            index=False,
        )
        edge_list_to_df(utils.adj_to_edge(G_star, np.arange(num_nodes))).to_csv(
            "./examples/edges_true.csv",
            header=["node1", "node2", "weight"],
            index=False,
        )
        pd.DataFrame(list(zip(np.arange(len(bias)), bias, var))).to_csv(
            "./examples/data_gen_true.csv",
            header=["node id", "bias", "var"],
            index=False,
        )

    # def solve_single_partition(
    #     self, G_star, data_obs, name, nodes, partition, superstructure, A_X_v=None
    # ):
    def single_partition(
        self,
        gen_graph: GeneratedGraph,
        partition: Partition,
        partition_name: str,
        super_structure: np.ndarray,
        A_X_v: np.ndarray | None = None,
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
        results = []
        chunksize = max(1, num_partitions // nthreads)

        with ProcessPoolExecutor(max_workers=nthreads) as executor:
            for result in executor.map(
                _local_structure_learn, subproblems, chunksize=chunksize
            ):
                results.append(result)

        # Merge the subset learned graphs
        G_star = utils.edge_to_adj(list(gen_graph.edges), nodes=gen_graph.nodes)
        fused_A_X_s = fusion(partition, results, gen_graph.samples_to_numpy())
        utils.delta_causality(A_X_v, fused_A_X_s, G_star)
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
