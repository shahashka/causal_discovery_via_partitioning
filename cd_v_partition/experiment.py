import datetime
from concurrent.futures import as_completed, ProcessPoolExecutor, Future
from pathlib import Path
from typing import Any, Callable
import time
import functools

import numpy as np
import pandas as pd
import tqdm
from numpy.random import RandomState
import matplotlib.pyplot as plt
import seaborn as sns

import cd_v_partition.utils as utils
from cd_v_partition.causal_discovery import pc, pc_local_learn, ges_local_learn, fci_local_learn
from cd_v_partition.fusion import fusion, screen_projections, no_partition_postprocess
from cd_v_partition.overlapping_partition import (
    partition_problem,
    expansive_causal_partition,
    rand_edge_cover_partition,
    modularity_partition,
    PEF_partition
)
from cd_v_partition.configs.config import SimulationConfig, SimulationSpec
from cd_v_partition.typing import GeneratedGraph, GraphKind, TrueGraph

class Experiment:
    def __init__(
        self,
        workers: int,
    ) -> None:
        self.workers = workers

    # ExecConfig
    def run(
        self,
        cfg: SimulationConfig,
        random_state: RandomState | int | None = None,
    ) -> None:
        if self.workers == 1:
            return self.run_serial(cfg, random_state)
        elif self.workers > 1:
            return self.run_concurrent()
        else:
            raise ValueError("Experiment member `workers` must be >= 1.")

    def run_concurrent(
        self, cfg: SimulationConfig, 
        random_state: RandomState | int | None = None
    ):
        random_state = utils.load_random_state(random_state)
        date = datetime.datetime.now()
        with ProcessPoolExecutor(self.workers) as executor:
            futures: dict[Future, tuple[int, int]] = {}
            for spec_id, spec in enumerate(cfg):
                seed = random_state.randint(0, 2**32 - 1) 
                # One process for each alg x spec, TODO check each spec should have the same random state, so the same set of graphs are created?
                fut = executor.submit(self.run_simulation, alg, spec_id, cfg.graph_per_spec, 
                                        cfg.experiment_id, date, spec, random_state=seed)
                futures[fut] = (spec_id, alg)

            for fut in as_completed(futures):
                spec_id, alg = futures[fut]
                outdir = Path(f"{cfg.experiment_id}/{date}/{alg}/spec_{spec_id}")
                spec.to_yaml(outdir / "spec.yaml")
        Experiment.read_visualize(date, cfg)

    def run_serial(self, cfg: SimulationConfig, random_state: RandomState | int | None = None):
        random_state = utils.load_random_state(random_state)
        date = datetime.datetime.now()
        #for alg in range(cfg.eval_algorithms):
        for spec_id, spec in enumerate(cfg):
            seed = random_state.randint(0, 2**32 - 1) 
            self.run_simulation(spec_id, cfg.graph_per_spec, 
                                    cfg.experiment_id, date, spec, random_state=seed)
            outdir = Path(f"{cfg.experiment_id}/{date}/{spec.partition_fn}/spec_{spec_id}")
            spec.to_yaml(outdir / "spec.yaml")
        Experiment.read_visualize(date, cfg)

    def run_simulation(
        self, spec_id: int,  num_trials: int, experiment_id: int, date: str, spec: SimulationSpec, random_state: RandomState | int | None = None
    ) -> pd.DataFrame:
        # for loop over trials with progress bar, no parallelism (for now)
        progressbar = tqdm.tqdm(total=num_trials)
        for trial in range(num_trials):
            progressbar.set_description(f"Running {spec.partition_fn} spec id {spec_id}")
            scores, time = Experiment.run_simulation_trial(spec, random_state)
            outdir = Path(f"{experiment_id}/{date}/{spec.partition_fn}/spec_{spec_id}/trial_{trial}/")
            if not outdir.exists():
                outdir.mkdir(parents=True)
                
            # Create checkpoint object and save
            out_data = np.zeros(5)
            out_data[0:5] = scores
            out_data[5] = time
            np.savetxt(outdir / "chkpoint.txt", out_data)
            progressbar.update()

    def run_simulation_trial(
        self, spec: SimulationSpec, random_state: RandomState | int | None = None
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
        gen_graph = Experiment.generate_graph(
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
                gen_graph.samples_to_numpy(), alpha=spec.alpha, outdir=None, num_cores=16
            )
        else:
            G_star = utils.edge_to_adj(list(gen_graph.edges), nodes=gen_graph.nodes)
            super_struct = utils.artificial_superstructure(
                G_star,
                frac_retain_direction=spec.frac_retain_direction,
                frac_extraneous=spec.frac_extraneous,
            )

        causal_discovery_alg = Experiment.get_causal_alg(spec.causal_learn_fn) 
        start = time.time()
        if spec.partition_fn is "no_partition":
            out_adj = causal_discovery_alg((super_struct, gen_graph.samples))
            out_adj = no_partition_postprocess(super_struct, out_adj, ss_subset=spec.ss_subset_flag)
        else:
            merge_alg = Experiment.get_merge_alg(spec.merge_fn) 
            partition_alg = Experiment.get_partitioning_alg(spec.partition_fn)
            
            # Partition
            partition = partition_alg(super_struct, cutoff=spec.partition_cutoff, 
                                      resolution=spec.partition_resolution, 
                                      best_n=spec.partition_best_n) 
            # Learn in parallel
            func_partial = functools.partial(causal_discovery_alg)
            results = []
            subproblems = partition_problem(partition, super_struct, gen_graph.samples)
            with ProcessPoolExecutor(max_workers=self.workers) as executor:
                for result in executor.map(func_partial, subproblems, chunksize=1):
                    results.append(result) 
            # Merge
            out_adj = merge_alg(ss=super_struct,partition=partition, local_cd_adj_mats=results,
                        data= gen_graph.samples_to_numpy(), 
                        ss_subset=spec.merge_ss_subset_flag, 
                        finite_lim=spec.merge_finite_sample_flag,
                        full_cand_set=spec.merge_full_cand_set
                        )
        total_time = time.time() - start
        scores = utils.get_scores([spec.partition_fn], [out_adj], G_star)
        return scores, total_time

    def read_visualize(date:Any, cfg: SimulationConfig):
        def sns_vis(cfg, results, ax, ind, label):
            data = [ s[:,:,ind] for s in results]
            data = [np.reshape(d, cfg.graph_per_spec * len(cfg.sweep_values)) for d in data]
            df = pd.DataFrame(data=np.column_stack(data), columns=cfg.eval_algorithms)
            df["samples"] = np.repeat(
                [cfg.sweep_values], cfg.graph_per_spec, axis=0
            ).flatten()  # samples go 1e2->1e7 1e2->1e7 etc
            df = df.melt(id_vars="samples", value_vars=cfg.eval_algorithms)
            df= df[df['value'] != 0] # Remove incomplete rows 
            x_order = np.unique(df["samples"])
            g = sns.boxplot(
                data=df,
                x="samples",
                y="value",
                hue="variable",
                order=x_order,
                hue_order=cfg.eval_algorithms,
                ax=axs[0],
                showfliers=False,
            )
            sns.move_legend(g, "center left", bbox_to_anchor=(1, 0.5), title="Algorithm")
            ax.set_xlabel(cfg.sweep_param)
            ax.set_ylabel(label)
            return g
        
        # Read checkpoints
        results_algs = []
        for alg in cfg.eval_algorithms:
            results = np.zeros((cfg.graph_per_spec, len(cfg.sweep_values), 6))
            for spec_id in range(len(cfg.sweep_values)):
                out_path = Path(f"{cfg.experiment_id}/{date}/{alg}/spec_{spec_id}")
                if out_path.exists():
                    for trial_id in range(cfg.graph_per_spec):
                        out_path = out_path / f"trial_{trial_id}/chkpoint.txt"
                        if out_path.exists():
                            results[trial_id][spec_id] = np.loadtxt(out_path)
            results_algs.append(results)
        
        # Save plots 
        shd_ind = 0
        tpr_ind = 3
        time_ind = 5
        save_path = Path(f"{cfg.experiment_id}/{date}")
        _, axs = plt.subplots(2, figsize=(10, 12), sharex=True)
        g = sns_vis(cfg, results_algs, axs[0],tpr_ind, "TPR")
        g = sns_vis(cfg, results_algs, axs[1],shd_ind, "SHD")
        plt.tight_layout()
        plt.savefig(save_path / "fig.png")

        plt.clf()
        _, ax = plt.subplots()
        g = sns_vis(cfg, results_algs, axs[1],time_ind, "Time to solution (s)")
        g.set(yscale="log")
        ax.set_ylabel("Time to solution (s)")
        plt.savefig(save_path/ "time.png")

        # Save score matrices
        for s, l in zip(results_algs, cfg.eval_algorithms):
            np.savetxt(
                save_path / f"scores_{l}.txt", s.reshape(cfg.graph_per_spec, -1)
            )
        
    @staticmethod
    def generate_graph(
        kind: GraphKind,
        num_nodes: int,
        num_communities: int,
        num_samples: int,
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
            random_state=random_state
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

    @staticmethod
    def get_partitioning_alg(self, spec: SimulationSpec, code: str) -> Callable:
        match spec.partition_fn:
            case "modularity":
                return modularity_partition
            case "edge_cover":
                return rand_edge_cover_partition
            case "expansive_causal":
                return expansive_causal_partition
            case "PEF":
                return PEF_partition
            case _:
                raise ValueError(f"`{spec.partition_fn=}` is an illegal value.`")
    
    @staticmethod
    def get_causal_discovery_alg(self, spec: SimulationSpec, code: str) -> Callable:
        match spec.causal_learn_fn:
            case "GES":
                return ges_local_learn
            case "PC":
                return pc_local_learn
            case "FCI":
                return fci_local_learn
            case "NO-TEARS":
                raise NotImplementedError(f"`{spec.causal_learn_fn=}` has not been implemented yet.`")
            case "GPS":
                raise NotImplementedError(f"`{spec.causal_learn_fn=}` has not beenn implemented yet.`")
            case _:
                raise ValueError(f"`{spec.causal_learn_fn=}` is an illegal value.`")

    @classmethod
    def get_merge_alg(cls, spec: SimulationSpec) -> Callable:
        match spec.fusion_fn:
            case "fusion":
                return fusion
            case "screen":
                return screen_projections
            case _:
                raise ValueError(f"`{spec.fusion_fn=}` is an illegal value.`")