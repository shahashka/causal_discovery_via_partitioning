from concurrent.futures import as_completed

from cd_v_partition.configs.base import SimulationConfig


class Simulation:
    def __init__(
        self,
        graph_factory,
        structure_learner,
        causal_learner,
        merge_fn,
    ) -> None:
        pass

    def run(self, exec_cfg: ExecConfig, cfg: SimulationConfig) -> None:
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
        """
        true_graph = GraphFactory.create_graph()
        superstructure = self.structure_learner.learn(true_graph)
        parts = self.partition(superstructure)

        learned_structs = {}
        for i, part in enumerate(parts):
            learned_structs[i] = self.causal_learner(part)

        predicted_graph = self.merge(learned_structs)
        self.evaluate(predicted_graph)
        """
