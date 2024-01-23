import numpy as np
from cd_v_partition.causal_discovery import sp_gies
from cd_v_partition.overlapping_partition import rand_edge_cover_partition, expansive_causal_partition, modularity_partition, partition_problem, PEF_partition
from cd_v_partition.utils import get_data_from_graph, create_k_comms, artificial_superstructure, get_scores, edge_to_adj
from cd_v_partition.fusion import fusion, screen_projections
import functools
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import seaborn as sns
import pandas as pd
# Why does the random edge coverage outperform the causal partition?
# We suspect that the size of the overlap is the reason errors in the casual partition are more frequent than edge coverage
# This happens with a dense superstructure or bad conductance cuts

# In order to study we increase the density of the superstructure and measure the TPR, FPR and SHD of the causal partition
# versus the edge coverage partition
# We keep track and report the number of overlaps in the causal partition/edge coverage partition to see if this correlates
# with poor results

# If the superstructure is sparse then we expect edge coverage to converge to causal partition esp. for hierarchical networks
# where there aren't many potentially directed paths that flow out and back into a subset 


def _local_structure_learn(subproblem):
    skel, data = subproblem
    adj_mat = sp_gies(data, skel=skel, outdir=None)
    return adj_mat
    
def run_causal_discovery(superstructure, partition, df, G_star, full_cand_set=False):

    # Break up problem according to provided partition
    subproblems = partition_problem(partition, superstructure, df)

    # Local learn
    func_partial = functools.partial(_local_structure_learn)
    results = []
    num_partitions = len(partition)
    nthreads = 2
    chunksize = max(1, num_partitions // nthreads)
    print("Launching processes")

    with ProcessPoolExecutor(max_workers=nthreads) as executor:
        for result in executor.map(func_partial, subproblems, chunksize=chunksize):
            results.append(result)

    # Merge globally
    data_obs = df.drop(columns=["target"]).to_numpy()
    #est_graph_partition = fusion(partition, results, data_obs, full_cand_set=full_cand_set)
    est_graph_partition = screen_projections(partition, results)

    # Call serial method
    est_graph_serial = _local_structure_learn([superstructure, df])

    # Compare causal metrics
    # d_scores = delta_causality(est_graph_serial, est_graph_partition, G_star)
    scores_serial = get_scores(["CD-serial"], [est_graph_serial], G_star)
    scores_part = get_scores(["CD-partition"], [est_graph_partition], G_star)

    return scores_serial[-2:], scores_part[-2:]


def run_study():
    num_repeats = 30
    ns=1e5
    ss_density = np.arange(0, 1, 0.25)

    scores_serial = np.zeros((num_repeats, len(ss_density), 2))
    scores_edge_cover = np.zeros((num_repeats, len(ss_density), 2))
    scores_causal_partition = np.zeros((num_repeats, len(ss_density), 2))
    scores_mod_partition = np.zeros((num_repeats, len(ss_density), 2))
    scores_pef_partition = np.zeros((num_repeats, len(ss_density), 2))

    for i in range(num_repeats):
        for j,ss in enumerate(ss_density):
            init_partition, graph = create_k_comms(
                "scale_free", n=25, m_list=[1,2], p_list=[0.5,0.5], k=2
            )
            num_nodes = len(graph.nodes())
            bias = np.random.normal(0, 1, size=num_nodes)
            var = np.abs(np.random.normal(0, 1, size=num_nodes))
            # Generate data
            (edges, nodes, _, _), df = get_data_from_graph(
                list(np.arange(num_nodes)),
                list(graph.edges()),
                nsamples=int(ns),
                iv_samples=0,bias=bias, var=var
            )
            G_star = edge_to_adj(edges, nodes)
            superstructure = artificial_superstructure(G_star, frac_extraneous=ss)

            
            mod_partition = modularity_partition(superstructure, cutoff=1, best_n=None)
            
            ss, sp = run_causal_discovery(superstructure, mod_partition, df, G_star)
            scores_serial[i][j] = ss
            scores_mod_partition[i][j] = sp
            
            partition = rand_edge_cover_partition(superstructure, mod_partition)
            _, sp = run_causal_discovery(superstructure, partition, df, G_star)
            scores_edge_cover[i][j] = sp
            
            partition = expansive_causal_partition(superstructure, mod_partition)
            _, sp = run_causal_discovery(superstructure, partition, df, G_star)
            scores_causal_partition[i][j] = sp
        
            partition = PEF_partition(df)
            _, sp = run_causal_discovery(superstructure, partition, df, G_star, full_cand_set=True)
            scores_pef_partition[i][j] = sp


    plt.clf()
    fig, axs = plt.subplots(2, figsize=(10,8),sharex=True)
    plt.title("Edge coverage case study for 2 comm scale free network")

    data = [scores_serial[:,:,0], scores_pef_partition[:,:,0],scores_edge_cover[:,:,0], scores_causal_partition[:,:,0], scores_mod_partition[:,:,0]] 
    
    data = [np.reshape(d, num_repeats*len(ss_density)) for d in data]
    labels = [ 'serial', 'pef', 'edge_cover', 'expansive_causal', 'mod']
    df = pd.DataFrame(data=np.column_stack(data), columns=labels)
    df['samples'] = np.repeat(ss_density, num_repeats)
    df = df.melt(id_vars='samples', value_vars=labels)
    x_order = np.unique(df['samples'])
    g = sns.boxplot(data=df, x='samples', y='value', hue='variable', order=x_order, hue_order=labels, ax=axs[0])
    axs[0].set_xlabel("Fraction of Extraneous edges")
    axs[0].set_ylabel("TPR")
    
    
    data = [scores_serial[:,:,1], scores_pef_partition[:,:,1], scores_edge_cover[:,:,1], scores_causal_partition[:,:,1], scores_mod_partition[:,:,1]] 
    data = [np.reshape(d, num_repeats*len(ss_density)) for d in data]
    labels = [ 'serial', 'pef', 'edge_cover', 'expansive_causal', 'mod']
    df = pd.DataFrame(data=np.column_stack(data), columns=labels)
    df['samples'] = np.repeat(ss_density, num_repeats)
    df = df.melt(id_vars='samples', value_vars=labels)
    x_order = np.unique(df['samples'])
    sns.boxplot(data=df, x='samples', y='value', hue='variable', order=x_order, hue_order=labels, ax=axs[1], legend=None)
    axs[1].set_xlabel("Fraction of Extraneous edges")
    axs[1].set_ylabel("FPR")
    sns.move_legend(g, "center left", bbox_to_anchor=(1, .5), title='Algorithm')

    plt.tight_layout()
    plt.savefig(
        "./tests/empirical_tests/edge_coverage_study_screen_projections.png"
    )
    
if __name__ == "__main__":
    run_study()

