from concurrent.futures import as_completed, ProcessPoolExecutor, Future
from pathlib import Path
from typing import Any, Callable
import time
import functools
import os
import numpy as np
import pandas as pd
import tqdm
from numpy.random import RandomState
import networkx as nx
from cd_v_partition.vis_partition import create_partition_plot
import cd_v_partition.utils as utils
from cd_v_partition.causal_discovery import pc, pc_local_learn, ges_local_learn, rfci_local_learn, rfci_pag_local_learn, damga_local_learn
from cd_v_partition.fusion import fusion, screen_projections, no_partition_postprocess, screen_projections_pag2cpdag
from cd_v_partition.overlapping_partition import (
    partition_problem,
    expansive_causal_partition,
    rand_edge_cover_partition,
    modularity_partition,
    PEF_partition
)
from cd_v_partition.config import SimulationConfig, SimulationSpec
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
            return self.run_concurrent(cfg, random_state)
        else:
            raise ValueError("Experiment member `workers` must be >= 1.")

    def run_concurrent(
        self, cfg: SimulationConfig, 
        random_state: RandomState | int | None = None
    ):
        random_state = utils.load_random_state(random_state)
        #date = datetime.datetime.now()
        with ProcessPoolExecutor(self.workers) as executor:
            futures: dict[Future, tuple[int, int, str, str, str]] = {}
            for trial in range(cfg.graph_per_spec):
                seed = trial
                for spec_id, spec in enumerate(cfg):
                    outdir = Path(f"{cfg.experiment_id}/{spec.partition_fn}/{spec.causal_learn_fn}/spec_{spec_id}/trial_{trial}/")                
                    if not outdir.exists():
                        outdir.mkdir(parents=True)
                    fut = executor.submit(self.run_simulation, spec, random_state=seed, save_vis=cfg.save_vis, save_path=outdir)
                    futures[fut] = (spec_id, trial, spec.partition_fn, spec.causal_learn_fn, outdir)
                    
            progressbar = tqdm.tqdm(total=cfg.graph_per_spec * len(cfg))
            for fut in as_completed(futures):
                spec_id, trial, p_alg, cd_alg, outdir = futures[fut]
                
                # Create checkpoint object and save
                np.savetxt(outdir / "chkpoint.txt", fut.result()[0])
                np.savetxt(outdir / "sizes.txt", fut.result()[1])

                # spec.to_yaml(outdir / '..'/ 'spec.yaml')         # TODO this is hanging  
                # print('done yaml save')
                progressbar.update()

    def run_serial(self, cfg: SimulationConfig, random_state: RandomState | int | None = None):
        random_state = utils.load_random_state(random_state)
        #date = datetime.datetime.now()
        progressbar = tqdm.tqdm(total=cfg.graph_per_spec * len(cfg), desc='Working on experiment configurations...')
        for trial in range(cfg.graph_per_spec):
            seed = trial
            for spec_id, spec in enumerate(cfg):
                outdir = Path(f"{cfg.experiment_id}/{spec.partition_fn}/{spec.causal_learn_fn}/spec_{spec_id}/trial_{trial}/")
                if not outdir.exists():
                    outdir.mkdir(parents=True)
                    
                scores = self.run_simulation(spec, random_state=seed, save_vis=cfg.save_vis, save_path=outdir)

                
                np.savetxt(outdir / "chkpoint.txt", scores[0])
                np.savetxt(outdir / "sizes.txt", scores[1])
               # spec.to_yaml(outdir / '..' / 'spec.yaml')  
                progressbar.update()         


    def run_simulation(
        self, spec: SimulationSpec, random_state: RandomState | int | None = None, save_vis: bool = False, save_path: Path | None = None
    ) -> np.ndarray:
        random_state = utils.load_random_state(random_state)
        # GENERATE THE GRAPH AND DATA
        gen_graph = Experiment.generate_graph(
            kind=spec.graph_kind,
            load_path=spec.graph_load_path,
            num_nodes=spec.num_nodes,
            num_samples=spec.num_samples,
            num_communities=spec.num_communities,
            comm_popularity=spec.comm_pop,
            edge_prob=spec.comm_edge_prob,
            inter_edge_prob=spec.inter_edge_prob,  # rho
            save_vis=save_vis,
            save_path=save_path
        )
        G_star = utils.edge_to_adj(list(gen_graph.edges), nodes=gen_graph.nodes)

        # GENERATE THE SUPERSTRUCTURE
        if spec.use_pc_algorithm:
            print(f"ALPHA {spec.alpha}")
            super_struct, _ = pc(
                gen_graph.samples, skel=np.ones((spec.num_nodes,spec.num_nodes)), alpha=spec.alpha, outdir=None, num_cores=16
            )
            print(f"Number of edges in ss {np.sum(super_struct)}")
        else:
            super_struct = utils.artificial_superstructure(
                G_star,
                frac_retain_direction=spec.frac_retain_direction,
                frac_extraneous=spec.frac_extraneous,
            )
        print("Generated graph")
        causal_discovery_alg = Experiment.get_causal_discovery_alg(spec) 
        start = time.time()
        partition_sizes=[]
        if spec.partition_fn == "no_partition":
            out_adj = causal_discovery_alg((super_struct, gen_graph.samples), spec.causal_learn_use_skel)
            print("CD Done")
            out_adj = no_partition_postprocess(super_struct, out_adj, ss_subset=spec.merge_ss_subset_flag)
        else:
            merge_alg = Experiment.get_merge_alg(spec) 
            partition_alg = Experiment.get_partitioning_alg(spec)
            
            # Partition
            partition = partition_alg(super_struct, data=gen_graph.samples, cutoff=spec.partition_cutoff, 
                                      resolution=spec.partition_resolution, 
                                      best_n=spec.partition_best_n) 
                    
            if save_vis:
                print("Saving graph as a gephy file with estimated partition")
                G = nx.from_numpy_array(G_star, create_using=nx.DiGraph)
                partition_by_node = {}
                for comm, nodes in partition.items():
                    for n in nodes:
                        partition_by_node[n] = comm # note that this doesn't include overlaps
                nx.set_node_attributes(G , partition_by_node, name='community')
                nx.write_gexf(G, f"{save_path}/{spec.partition_fn}_estimated_graph.gexf")
            
            
            
            # Learn in parallel
            func_partial = functools.partial(causal_discovery_alg, use_skel= spec.causal_learn_use_skel, params=spec.causal_learn_params) if spec.causal_learn_params \
                else functools.partial(causal_discovery_alg, use_skel= spec.causal_learn_use_skel)
            results = []
            subproblems = partition_problem(partition, super_struct, gen_graph.samples)
            workers = min(len(subproblems), os.cpu_count())
            # workers=1 # Serial for debuggign 
            print(f"Launching {workers} workers for partitioned run")
            
            partition_sizes = [len(p) for p in partition.values()]
            print(f"Biggest partition size {max(partition_sizes)}")
            print(partition_sizes)
            #return np.zeros(6), partition_sizes
            with ProcessPoolExecutor(max_workers=workers) as executor:
                results = list(tqdm.tqdm(executor.map(func_partial, subproblems, chunksize=1), total=len(subproblems)))
            print("CD done")
            # Merge
            out_adj = merge_alg(ss=super_struct,partition=partition, local_cd_adj_mats=results,
                        data= gen_graph.samples_to_numpy(), 
                        ss_subset=spec.merge_ss_subset_flag, 
                        finite_lim=spec.merge_finite_sample_flag,
                        full_cand_set=spec.merge_full_cand_set
                        )
            print('merge done')
        total_time = time.time() - start
        scores = utils.get_scores([spec.partition_fn], [out_adj], G_star)
        out_data = np.zeros(6)
        out_data[0:5] = scores
        out_data[5] = total_time

        return out_data, partition_sizes
    
        
    @staticmethod
    def generate_graph(
        kind: GraphKind,
        load_path: str,
        num_nodes: int,
        num_communities: int,
        num_samples: int,
        comm_popularity: list[int],
        edge_prob: list[float],
        inter_edge_prob: float,
        random_state: RandomState | int | None = None,
        save_vis: bool = False,
        save_path: Path |None = None
    ) -> GeneratedGraph:
        random_state = utils.load_random_state(random_state)
        
        # Generate graph topology 
        if kind == "hierarchical":
            graph = utils.directed_heirarchical_graph(num_nodes, random_state)
            default_partition = None
        elif kind == "ecoli":
            if load_path is None:
                raise ValueError("graph_load_path not set for ecoli graphs")
            adj_mat = np.loadtxt(load_path)
            graph =  nx.from_numpy_array(adj_mat, create_using=nx.DiGraph)
            num_nodes = adj_mat.shape[0]
            default_partition = None
        else:
            default_partition, graph = utils.create_k_comms(
                graph_type=kind,
                n=int(num_nodes / num_communities),
                m_list=comm_popularity,
                p_list=edge_prob,
                k=num_communities,
                rho=inter_edge_prob,
                random_state=random_state
            )
        # Generate corresponding dataset
        (edges, nodes, bias, var), samples = utils.get_data_from_graph(
            list(np.arange(num_nodes)),
            list(graph.edges()),
            nsamples=int(num_samples),
            iv_samples=0,
            bias=None,
            var=None,
        )
        if save_vis:
            print("Saving true graph as a gephy file")
            G = nx.DiGraph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)
            if default_partition:
                partition_by_node = {}
                for i, comm in default_partition.items():
                    for n in comm:
                        partition_by_node[n] = i
                nx.set_node_attributes(G , partition_by_node, name='community')
            nx.write_gexf(G, f"{save_path}/ground_truth.gexf")
        return GeneratedGraph(
            true_graph=TrueGraph(nodes, edges),
            samples=samples,
            default_partition=default_partition,
            bias=bias,
            variance=var,
        )

    @classmethod
    def get_partitioning_alg(cls, spec: SimulationSpec) -> Callable:
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
    
    @classmethod
    def get_causal_discovery_alg(cls, spec: SimulationSpec) -> Callable:
        match spec.causal_learn_fn:
            case "GES":
                return ges_local_learn
            case "PC":
                return pc_local_learn
            case "RFCI":
                return rfci_local_learn
            case "NOTEARS":
                return damga_local_learn
            case "GPS":
                raise NotImplementedError(f"`{spec.causal_learn_fn=}` has not beenn implemented yet.`")
            case "RFCI-PAG":
                return rfci_pag_local_learn
            case _:
                raise ValueError(f"`{spec.causal_learn_fn=}` is an illegal value.`")

    @classmethod
    def get_merge_alg(cls, spec: SimulationSpec) -> Callable:
        match spec.merge_fn:
            case "fusion":
                return fusion
            case "screen":
                if spec.causal_learn_fn == "RFCI-PAG":
                    return screen_projections_pag2cpdag
                else:
                    return screen_projections
            case _:
                raise ValueError(f"`{spec.merge_fn=}` is an illegal value.`")