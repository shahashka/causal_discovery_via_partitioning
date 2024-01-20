# Imports
import networkx as nx

# Imports for code development
from tqdm import tqdm
from helpers import artificial_superstructure
from build_heirarchical_random_graphs import (
    directed_heirarchical_graph,
)

from cd_v_partition.utils import (
    get_data_from_graph,
)
from cd_v_partition.overlapping_partition import PEF_partition
from cd_v_partition.causal_discovery import pc
from cd_v_partition.vis_partition import create_partition_plot


def generate_heirarchical_instance(num_nodes=50, superstructure_mode="artificial"):
    ## Generate a random network with heirarchical structure and corresponding dataset
    G_star_graph = directed_heirarchical_graph(num_nodes=50)
    G_star_adj = nx.adjacency_matrix(G_star_graph)
    assert set(G_star_graph.nodes()) == set(range(G_star_graph.number_of_nodes()))
    nodes = list(range(G_star_graph.number_of_nodes()))

    ## Generate data
    df = get_data_from_graph(
        nodes,
        list(G_star_graph.edges()),
        nsamples=int(1e4),
        iv_samples=0,
    )[1]

    ## Find the 'superstructure'
    # artificially create superstructure
    if superstructure_mode == "artificial":
        superstructure = artificial_superstructure(G_star_adj, frac_extraneous=0.1)
    elif superstructure_mode == "PC":
        df_obs = df.drop(columns=["target"])
        data_obs = df_obs.to_numpy()
        superstructure, _ = pc(data_obs, alpha=0.1, outdir=None)
    return G_star_graph, G_star_adj, superstructure, df


# quick test:
print("Running PEF partitioning on 20 random heirarchical graphs.")
for _ in tqdm(range(20)):
    _, _, _, df = generate_heirarchical_instance()
    PEF_partition(df, 0.05)

print("Visualizing example PEF partition")
G_star_graph, G_star_adj, superstructure, df = generate_heirarchical_instance()
nodes = list(range(G_star_graph.number_of_nodes()))
partition = PEF_partition(df, 0.05)
print("PLOTTING PEF PARTITION IN G_STAR.")
create_partition_plot(
    G_star_graph,
    nodes,
    partition,
    "{}/PEF_partition_in_G_star.png".format("./"),
)
