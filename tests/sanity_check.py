import networkx as nx
import random
import itertools
import numpy as np
from cd_v_partition.causal_discovery import rfci_pag_local_learn
from cd_v_partition.fusion import screen_projections_pag2cpdag
from cd_v_partition.overlapping_partition import partition_problem
from cd_v_partition.utils import get_data_from_graph
import matplotlib.pyplot as plt


def is_inducing(path, subset, dag):
    inducing=True # default is it's inducing
    for i in np.arange(1,len(path)-1):
        if path[i] in subset: # for any mediators in the subset
            if dag.has_edge(path[i-1], path[i]) and dag.has_edge(path[i+1], path[i]): # if it's not a collider
                if path[i] in nx.ancestors(dag, path[0]) or path[i] in nx.ancestors(dag, path[-1]): # if it's an ancestor
                    print("NON TRIVIAL INDUCING PATH FOUND")
                    continue 
                else:
                    inducing = False
            else:
                inducing = False
    return inducing

results_rfci = []
results_my_pag_rep = []
for seed in np.arange(100):
    random.seed(seed)

    # (1) Sample random trees over 10 nodes
    num_nodes = 10
    G_undir = nx.random_labeled_tree(num_nodes, seed=42)
    G_dir = nx.DiGraph()
    G_dir.add_edges_from(G_undir.edges)

    # (2) Pick an edge that does not connect to a leaf node randomly
    #     Partition into 2 connected subsets
    edges_choice= list(G_dir.edges)
    cut_edge = random.choice(edges_choice)
    while len(list(G_dir.successors(cut_edge[1]))) == 0 and len(list(G_dir.predecessors(cut_edge[1]))) == 1:
        edges_choice.remove(cut_edge)
        cut_edge = random.choice(edges_choice)
    G_undir.remove_edge(cut_edge[0], cut_edge[1])
    G_S_undir = [G_dir.subgraph(c).copy() for c in nx.connected_components(G_undir)]
    S = [list(gs.nodes) for gs in G_S_undir ]

    # (3) Add an edge between subsets to create a more interesting structure (ensure acyclicity)
    new_edge = [random.choice(S[0]), random.choice(S[1])]
    while G_dir.has_edge(new_edge[0], new_edge[1]):
        new_edge = [random.choice(S[0]), random.choice(S[1])]

    G_dir.add_edge(new_edge[0], new_edge[1])
    try:
        cycle_list = nx.find_cycle(G_dir, orientation="original")
        while len(cycle_list) > 0:
            G_dir.remove_edge(new_edge[0], new_edge[1])
            new_edge = [random.choice(S[0]), random.choice(S[1])]
            while G_dir.has_edge(new_edge[0], new_edge[1]):
                new_edge = [random.choice(S[0]), random.choice(S[1])]
            G_dir.add_edge(new_edge[0], new_edge[1])
            try:
                cycle_list = nx.find_cycle(G_dir, orientation="original")
            except:
                break
    except:
        print("Found edge on first try")

    G_undir = nx.Graph()
    G_undir.add_edges_from(G_dir.edges)

    # (4) Create the causal partition from the disjoint partition 
    partition = dict(zip(np.arange(len(S)), S))
    causal_partition = dict()
    for idx, c in enumerate(list(partition.values())):
        outer_node_boundary = nx.node_boundary(G_undir, c)
        expanded_cluster = set(c).union(outer_node_boundary)
        causal_partition[idx] = list(expanded_cluster)
        
    # Vis stuff for debugging
    colors = ['blue', 'red', 'purple' ]
    color_map = dict()
    causal_partition_by_node = dict()
    for i, subset in causal_partition.items():
        for n in subset:
            if n in causal_partition_by_node.keys():
                causal_partition_by_node[n] += [i]
            else:
                causal_partition_by_node[n] = [i]
    for i, partition in causal_partition_by_node.items():
        if len(partition) == 1:
            color_map[i] = colors[partition[0]]
        else:
            color_map[i] = colors[-1]
        
    # (5) Find all inducing paths
    #     Find all paths between all pairs of nodes within a subset 
    #     A path between these nodes is inducing if all mediators are outside the path
    #     or any mediators inside the subset are colliders and ancestors of endpoints
    inducing_paths = [] # holds the path and the subset for which it is inducing
    for subset in causal_partition.values():
        for pair in itertools.permutations(subset, 2):
            for path in nx.all_simple_paths(G_undir, pair[0], pair[1]):
                #print(path)
                if is_inducing(path, subset, G_dir) and len(path) > 2:
                    inducing_paths.append((subset,path))

    # (6) Find the latent PAG over subsets 
    #     Add all adjacencies (i--j) in the DAG within a subset to the latent PAG
    #     For all inducing paths, add an adjacency (i o--o j)
    #     Add all unshielded colliders in the DAG to the latent PAG 
    #     Assign values to adjacency matrix according to PAG edge representation

    # pag[i,j] = 0 iff no edge btw i,j
    # pag[i,j] = 1 iff i *-o j
    # pag[i,j] = 2 iff i *-> j
    # pag[i,j] = 3 iff i *-- j

    # Find all unshielded colliders in the DAG
    unshielded_colliders = []
    dag_adj_mat = nx.adjacency_matrix(G_dir, nodelist=np.arange(num_nodes)).todense()
    for col in np.arange(dag_adj_mat.shape[1]):
        if sum(dag_adj_mat[:,col]) == 2:
            collider = [col] # the first node will be the 'colliding point'
            for row in np.arange(dag_adj_mat.shape[0]):
                if dag_adj_mat[row,col] == 1 :
                    collider.append(row)
            if dag_adj_mat[collider[1], collider[2]] == 0:
                unshielded_colliders.append(collider)
    latent_PAGs_subsets = []
    for i,s in causal_partition.items():
        local_pag_adj = np.zeros((len(s), len(s)))
        # add all adjacencies in DAG
        for node_local_ind_i in np.arange(len(s)):
            for node_local_ind_j in np.arange(len(s)):
                node_global_ind_i = s[node_local_ind_i]
                node_global_ind_j = s[node_local_ind_j]
                if G_dir.has_edge(node_global_ind_i, node_global_ind_j):
                    local_pag_adj[node_local_ind_i, node_local_ind_j] = 3
                    local_pag_adj[node_local_ind_j, node_local_ind_i] = 3
        # add projected edge between endpoints of inducing paths 
        for path_subset,path in inducing_paths:
            if path_subset == s:
                start = path[0]
                end = path[-1]
                # if start in s and end in s:
                local_i = s.index(start)
                local_j = s.index(end)
                local_pag_adj[local_i, local_j] = 1
                local_pag_adj[local_j, local_i] = 1
        # Add all directions corresponding to unshielded colliders 
        for uc in unshielded_colliders:
            if uc[0] in s and uc[1] in s and uc[2] in s:
                local_k = s.index(uc[0])
                local_i = s.index(uc[1] )
                local_j = s.index(uc[2] )
                local_pag_adj[local_i, local_k] = 2
                local_pag_adj[local_j, local_k] = 2
                
                local_pag_adj[local_k, local_i] = 3
                local_pag_adj[local_k, local_j] = 3
        latent_PAGs_subsets.append(local_pag_adj)
        
            
    # (7) Use RFCI to estimate latent PAG over subsets
    # Generate 1M samples, split into subproblems, learn locally

    _,df = get_data_from_graph(np.arange(10), list(G_dir.edges), nsamples=int(1e6), iv_samples=0)
    subproblems = partition_problem(causal_partition, np.ones((num_nodes, num_nodes)), df)
    rfci_pags = []
    for s in subproblems:
        pag = rfci_pag_local_learn(s, use_skel=True)
        rfci_pags.append(pag)

    # (8) Screen projections, compare to DAG MEC 

    cpdag_rfci = screen_projections_pag2cpdag(ss=np.ones((num_nodes, num_nodes)), partition=causal_partition,local_cd_adj_mats=rfci_pags, ss_subset=True, finite_lim=False, data=df, full_cand_set=False)
    cpdag_my_latent_rep = screen_projections_pag2cpdag(ss=np.ones((num_nodes, num_nodes)),partition=causal_partition, local_cd_adj_mats=latent_PAGs_subsets, ss_subset=True, finite_lim=False, data=df, full_cand_set=False)
    correct = (cpdag_rfci.edges==cpdag_my_latent_rep.edges)
    print(f"CPDAGs are consistent with each other? {correct}")
    if not correct:
        print(f"My cpdag \n {cpdag_my_latent_rep.edges}")
        print(f"RFCI cpdag \n {cpdag_rfci.edges}")
    #Compare to ground truth
    dag_mec_cpdag = np.zeros((num_nodes, num_nodes))
    for edge in list(G_dir.edges):
        part_of_collider = False
        dag_mec_cpdag[edge[0], edge[1]] = 1
        dag_mec_cpdag[edge[1], edge[0]] = 1
        for uc in unshielded_colliders:
            if edge[1] == uc[0] and (edge[0] == uc[1] or edge[0] == uc[2]):
                dag_mec_cpdag[edge[0], edge[1]] = 1
                dag_mec_cpdag[edge[1], edge[0]] = 0
    G_cpdag = nx.from_numpy_array(dag_mec_cpdag, create_using=nx.DiGraph)
    rfci_test = G_cpdag.edges == cpdag_rfci.edges
    my_test = G_cpdag.edges == cpdag_my_latent_rep.edges
    print(f"RFCI CPDAG is consistent w ground truth MEC? {rfci_test}")
    print(f"My CPDAG rep is consistent w ground truth MEC? {my_test}")   
    results_rfci.append(rfci_test)
    results_my_pag_rep.append(my_test)
    if not rfci_test or not my_test :
        # Visualize all three cpdags
        pos = nx.planar_layout(G_dir)
        fig, axs = plt.subplots(ncols=4)
        axs[0].set_title("True DAG")
        axs[1].set_title("True CPDAG")
        axs[2].set_title("RFCI CPDAG")
        axs[3].set_title("My latent rep CPDAG")

        nx.draw(G_dir,ax=axs[0], with_labels=True,node_color=[color_map[node] for node in list(G_dir.nodes)], pos=pos)
        nx.draw(G_cpdag,ax=axs[1], with_labels=True,node_color=[color_map[node] for node in list(G_cpdag.nodes)], pos=pos)
        nx.draw(cpdag_rfci,ax=axs[2], with_labels=True,node_color=[color_map[node] for node in list(cpdag_rfci.nodes)], pos=pos)
        nx.draw(cpdag_my_latent_rep,ax=axs[3], with_labels=True,node_color=[color_map[node] for node in list(cpdag_my_latent_rep.nodes)], pos=pos)

        plt.savefig(f"tests/sanity_check_failures_high_sample/trial_{seed}.png")
        
        
print(f"RFCI tests {sum(results_rfci)} passed out of {len(results_rfci)}")
print(f"My tests {sum(results_my_pag_rep)} passed out of {len(results_my_pag_rep)}")