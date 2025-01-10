import numpy as np
import pandas as pd
import scipy.stats as st
from cd_v_partition.experiment import Experiment
from cd_v_partition.overlapping_partition import partition_problem
import functools
from concurrent.futures import ProcessPoolExecutor
import tqdm 
from cd_v_partition.config import SimulationSpec
import os
import networkx as nx
def load_data_and_superstructure(data_file, superstructure_file='random_corr_coef.txt', use_corr=True, genes_are_symbols=False):
    radbio_data = pd.read_csv(data_file, sep=",")
    num_genes = radbio_data.shape[1] - 2
    gene_dictionary = pd.read_csv("ncbi_dataset.tsv", sep='\t', header=0)

    # If the dataset contains ncbi ids
    if not genes_are_symbols:
        gene_ids = [int(i) for i in radbio_data.columns[0:num_genes]]

        # Which genes are in the dataset and are also 'protein coding'
        gene_data = gene_dictionary.loc[gene_dictionary['NCBI GeneID'].isin(gene_ids)]
        protein_gene_data = gene_data.loc[gene_data['Gene Type']=='PROTEIN_CODING']

        # Construct the 'observational' data set by ignoring internventions for now
        data_ncbi_ids = [str(i) for i in protein_gene_data['NCBI GeneID']]
        final_data_set = radbio_data[data_ncbi_ids]
        data_ncbi_ids_w_dose = data_ncbi_ids + ["Dose"]
        final_data_set_w_condition = radbio_data[data_ncbi_ids_w_dose]
        final_data_set_w_condition = final_data_set_w_condition.rename(mapper={"Dose":"target"}, axis=1)
    # If the dataset contains gene sumbols
    else:
        gene_symbols = radbio_data.columns[0:num_genes]
        gene_data = gene_dictionary.loc[gene_dictionary['Symbol'].isin(gene_symbols)]
        protein_gene_data = gene_data.loc[gene_data['Gene Type']=='PROTEIN_CODING']

        # Construct the 'observational' data set by ignoring internventions for now
        data_symbol_ids = [i for i in protein_gene_data['Symbol']]
        data_ncbi_ids = [str(i) for i in protein_gene_data['NCBI GeneID']]

        final_data_set = radbio_data[data_symbol_ids]
        data_symbol_ids_w_dose = data_symbol_ids + ["Dose", "Dose Rate"]
        final_data_set_w_condition = radbio_data[data_symbol_ids_w_dose]
        final_data_set_w_condition = final_data_set_w_condition.rename(mapper={"Dose":"target"}, axis=1)
        
    # Load the superstructure
    num_genes = final_data_set.shape[1]
    if use_corr:
        corr_mat = final_data_set.corr('pearson').to_numpy()
        random_corr_coef = np.loadtxt(superstructure_file)

        ci_interval = st.t.interval(0.95, len(random_corr_coef)-1, 
                                    loc=np.mean(random_corr_coef), 
                                    scale=st.sem(random_corr_coef))
        cutoff = ci_interval[1]

        corr_mat[corr_mat<=cutoff] = 0
        corr_mat[corr_mat>cutoff] = 1
        super_struct = corr_mat
    else:
        df_ppi = pd.read_csv(superstructure_file, header=0)
        super_struct = np.zeros((num_genes, num_genes))
        for i,edge in df_ppi.iterrows():
            # some genes in the superstructure may not be in the dataset, ignore these
            try:
                start = data_ncbi_ids.index(str(edge['start']))
                end = data_ncbi_ids.index(str(edge['end']))
                super_struct[start,end] = 1
            except:
                continue
    
    # create a gene map that contains {<data_ind>: <symbol> }
    gene_map = dict()
    for i, id in enumerate(data_ncbi_ids):
        gene_map[i] = gene_data.loc[gene_data['NCBI GeneID']==int(id)]['Symbol'].values[0]
    print(f"Superstructure contains {np.sum(super_struct)} edges which is \
          {np.sum(super_struct)/(num_genes**2)} fraction of all possible edges")
    return final_data_set_w_condition, super_struct, gene_map


def get_de_genes(rad_level, rad_dose, dir, padj_value=0.05):
    level_to_file_map = {("0.168Gy",0.001) : "deseq2_0.168_W1vs0_by_week_results.csv",
                         ("0.336Gy",0.001): "deseq2_0.336_W2vs0_by_week_results.csv",
                         ("0.504Gy", 0.001): "deseq2_0.504_W3vs0_by_week_results.csv",
                         ("1.68Gy",0.01): "deseq2_1.68_W1vs0_by_week_results.csv",
                         ("3.36Gy",0.01): "deseq2_3.36_W2vs0_by_week_results.csv",
                         ("5.04Gy",0.01): "deseq2_5.04_W3vs0_by_week_results.csv",
                         ("16.8Gy",0.1) : "deseq2_16.8_W1vs0_by_week_results.csv",
                         ("33.6Gy",0.1): "deseq2_33.6_W2vs0_by_week_results.csv",
                         ("50.4Gy",0.1): "deseq2_50.4_W3vs0_by_week_results.csv",
                         ("168.0Gy",1): "deseq2_168_W1vs0_by_week_results.csv",
                         ("336.0Gy",1): "deseq2_336_W2vs0_by_week_results.csv",
                         ("504.0Gy",1): "deseq2_504_W3vs0_by_week_results.csv",
                         ("336.0Gy",2): "deseq2_336_W1vs0_by_week_results.csv",
                         ("672.0Gy",2): "deseq2_672_W2vs0_by_week_results.csv",
                         ("1008.0Gy",2): "deseq2_1008_W3vs0_by_week_results.csv"}
    de_df = pd.read_csv(f"{dir}/{level_to_file_map[(rad_level, rad_dose)]}", header=0)
    genes = de_df.loc[de_df['padj']<=padj_value].iloc[:,0].values
    print(f"NUMBER OF DE GENNES {len(genes)}")
    return genes

# This is a wrapper for the causal_discovery_alg that handles subselecting genes that are
# differentially expressed for a particular subset
def causal_discovery_alg_subselect(subproblem, causal_discovery_alg, de_genes, params, use_skel):

    # For the subproblem, remove rows/columns for non de_genes in the skeleton and the dataset
    # Keep track of gene indices in original data
    skel, data = subproblem
    print(data.columns)
    de_genes_inds = [i for i, gene in enumerate(data.columns) if gene in de_genes]
    data_de = data.iloc[:,de_genes_inds]
    data_de = data_de.assign(target=data['target'].values)
    skel_de = skel[de_genes_inds][:,de_genes_inds]
    # Call causal_discovery_alg on de_genes only
    de_adj_mat = causal_discovery_alg((skel_de, data_de), params=params, use_skel=use_skel)
    # Add rows/columns of 0s to adj_mat at indices corresponding to removed genes
    adj_mat = np.zeros(skel.shape)
    for node_1 in np.arange(skel.shape[0]):
        for node_2 in np.arange(skel.shape[1]):
            if node_1 in de_genes_inds and node_2 in de_genes_inds:
                subselect_id_1 = de_genes_inds.index(node_1)
                subselect_id_2 = de_genes_inds.index(node_2)
                adj_mat[node_1, node_2] = de_adj_mat[subselect_id_1, subselect_id_2]
    # Return new adj mat with these buffer regions 
    print(adj_mat.shape)
    return adj_mat
    
def run_causal_discovery_w_partition(data, superstructure, genes=None, dose_rate=0.001, level='0.Gy', partition=None,
                                     cd_alg="GES", params={'reg':0.5, 'maxDegree':100},
                                     partition_fn="modularity"): 
    spec = SimulationSpec(causal_learn_fn=cd_alg, 
                        merge_fn="screen", 
                        partition_fn=partition_fn)
    causal_discovery_alg = Experiment.get_causal_discovery_alg(spec)
    merge_alg = Experiment.get_merge_alg(spec) 
    partition_alg = Experiment.get_partitioning_alg(spec)

    # Extract the data corresponding to the radiation level (always including the control)
    # Treat as observational. (TODO add radiation as random variable)
    # For level 0, run cd on full dataset
    # if level != '0.0Gy':
    data = data.loc[data['target'].isin(['0.0Gy',level])]
    data = data.loc[data['Dose Rate'].isin(['Control',dose_rate])]
    # else:
    #     data_level = data.copy()
    data = data.assign(target=0)
    data = data.drop(columns='Dose Rate')
    #data_level.loc[:,'target'] = np.zeros(data_level.shape[0])
    print(data.columns)
    # Partition the superstructure 
    if partition is None:
        partition = partition_alg(superstructure, data=data, resolution=5) 
            
    # Learn in parallel
    func_partial = functools.partial(causal_discovery_alg_subselect, causal_discovery_alg=causal_discovery_alg, de_genes=genes,params=params, use_skel= True)
    results = []
    subproblems = partition_problem(partition, superstructure, data)
    workers = min(len(subproblems), os.cpu_count())
    workers=1
    print(f"Launching {workers} workers for partitioned run")

    partition_sizes = [len(p) for p in partition.values()]
    print(f"Biggest partition size {max(partition_sizes)}")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm.tqdm(executor.map(func_partial, subproblems, chunksize=1), total=len(subproblems)))

    print("CD done")
    # Merge
    graph = merge_alg(ss=superstructure,partition=partition, local_cd_adj_mats=results,
                data= data.to_numpy(), 
                ss_subset=True, 
                finite_lim=False,
                full_cand_set=False)
                            
    print(f"Number of estimated edges {len(graph.edges())}")
    return graph, partition

def get_ego_graphs(graph, important_genes):
    ego_graph_disjoint = nx.DiGraph()
    for gene in important_genes:
        subgraph = nx.ego_graph(graph, n=gene, radius=2, center=True, undirected=True)
        ego_graph_disjoint.add_nodes_from(subgraph.nodes())
        ego_graph_disjoint.add_edges_from(subgraph.edges())
    node_att_map = dict()
    for gene in ego_graph_disjoint.nodes():
        node_att_map[gene] = gene in important_genes
    nx.set_node_attributes(ego_graph_disjoint, node_att_map, name="partition")
    return ego_graph_disjoint

if __name__=="__main__":
    partition_fn = "modularity"
    ss_type = "B" #ppi
    ss_file = "edges_ppi_by_gene_id.csv"
    data_file = "rnaseq_lucid/data_w_rate.csv" #"de_radbio_data.csv"
    de_dir = "rnaseq_lucid/deseq2"
    important_genes = ["TP53",
                       "NFKB1", 
                       "PTGS2",
                        "DEPP1",
                        "MT2A",
                        "G0S2",
                        "SERPINB2",
                        "CXCL12",
                        "CXCL3",
                        "CXCL5",
                        "CXCL10",
                        "CXCL2",
                        "FGF2",
                        "GPR68",
                        "DHRS3",
                        "IL1A",
                        "IL1B",
                        "IL6",
                        "CXCL8",
                        "LIF",
                        "MMP1",
                        "MMP10",
                        "C15orf48",
                        "NEFM",
                        "ATF3",
                        "BCL2A1",
                        "JADE2",
                        "SOD2",
                        "HMOX1",
                        "MT1E",
                        "BMP2",
                        "INHBA",
                        "BIRC3",
                        "TNFAIP3",
                        "KYNU",
                        "LAMB3",
                        "RRAD"]
    
    dose_rate_mapping = {"0.0Gy":["Control"], "0.168Gy":[0.001], "0.336Gy":[0.001], "0.504Gy":[0.001], 
                         "1.68Gy":[0.01], "3.36Gy":[0.01], "5.04Gy":[0.01],
                         "16.8Gy":[0.1], "33.6Gy":[0.1], "50.4Gy":[0.1],
                         "168.0Gy":[1], "336.0Gy":[1,2], "504.0Gy":[1],
                         "672.0Gy":[2], "1008.0Gy":[2]}
    data_cd, ss, gene_map = load_data_and_superstructure(data_file, ss_file , use_corr=ss_type=='A', genes_are_symbols=True)
    radiation_levels = list(set(data_cd['target'].to_list()))
    partition=None
    graphs = dict()
    ego_graphs = dict()
    

    # For each radiation level, run CD using the same partition
    for i, level in enumerate(radiation_levels):
        for j, dose in enumerate(dose_rate_mapping[level]):
            print(level, dose)
            save_suffix = f"_{partition_fn}_{ss_type}"
            if level != "0.0Gy":
                genes = get_de_genes(rad_level=level, rad_dose=dose, dir=de_dir)
            else:
                genes = list(data_cd.columns)
                genes.remove('target')

            graph, partition = run_causal_discovery_w_partition(data_cd, ss, genes=genes,level=level,dose_rate=dose, partition=partition, partition_fn="modularity")
            #print(f"RUN CD {nx.get_edge_attributes(graph, 'weight')}")

            partition_symbol = dict()
            for id,comm in partition.items():
                for c in comm:
                    gene_symbol = gene_map[c]
                    partition_symbol[gene_symbol] = id
                    
            if i == 0 and j == 0:
                # Save partition as lists of gene symbols
                if not os.path.exists(f"./comms_{ss_type}"):
                    os.makedirs(f"./comms_{ss_type}")
                for id,comm in partition.items():
                    with open(f"./comms_{ss_type}/comm_{id}{save_suffix}.txt", "w") as f:
                        for c in comm:
                            f.write(gene_map[c])
                            f.write('\n')
                # Store the superstructure as a .gexf file with nodes named by gene symbol
                #   and node labels corresponding to the partition index
                ss_graph = nx.from_numpy_array(ss, create_using=nx.Graph)
                ss_graph = nx.relabel_nodes(ss_graph, mapping=gene_map )
                nx.set_node_attributes(ss_graph, partition_symbol, name='partition')
                nx.write_gexf(ss_graph, f"ss{save_suffix}.gexf")
                graphs['ss'] = ss_graph
                
                ego_graph_ss = get_ego_graphs(ss_graph, important_genes)
                nx.write_gexf(ego_graph_ss, f"ego_ss{save_suffix}.gexf")
                ego_graphs['ss'] = ego_graph_ss

        
            # Store the DAG as a .gexf file with nodes named by gene symbol
            #   node labels corresponding to partition index, AND edge labels corresponding
            #   to parameters estimated by CD algorithm
            graph  = nx.relabel_nodes(graph, mapping=gene_map)
            nx.set_node_attributes(graph, partition_symbol, name='partition')
            nx.write_gexf(graph, f"dag{save_suffix}_{dose}_{level}.gexf")
            graphs[level] = graph
            print(f"Number of edges: {len(graph.edges)}")
            
            # ego_graph = get_ego_graphs(graph, important_genes)
            # nx.write_gexf(ego_graph,  f"ego_dag{save_suffix}__{dose}_{level}.gexf")
            # nx.write_edgelist(ego_graph, f"ego_dag{save_suffix}_{dose}_{level}.csv")
            # ego_graphs[level] = ego_graph

    # In order to visualize the overlay of SS, DAG, DAG', DAG'' etc..
    # Store a .gexf file with all edges in these structures. Label the edge by which graph(s) it is a part of
    #   include partition index and label nodes by symbol
    vis_edges = dict()
    for graph_type, graph in graphs.items():
        print(len(graph.nodes))
        for edge in graph.edges():
            if edge in vis_edges.keys():
                vis_edges[edge][graph_type] = True
            else:
                vis_edges[edge] = dict(zip(graphs.keys(), len(graphs.keys())*[False]))
                vis_edges[edge][graph_type] = True

            if graph_type=='ss':
                edge_undir = (edge[1], edge[1])
                if edge_undir in vis_edges.keys():
                    vis_edges[edge][graph_type] = True
                else:
                    vis_edges[edge] = dict(zip(graphs.keys(), len(graphs.keys())*[False]))
                    vis_edges[edge][graph_type] = True    
    vis_graph = nx.DiGraph()
    print(len(gene_map.values()))
    vis_graph.add_nodes_from(gene_map.values())
    nx.set_node_attributes(vis_graph, partition_symbol, name='partition')
    
    important_gene_map = dict()
    for gene in vis_graph.nodes():
        important_gene_map[gene] = gene in important_genes
    nx.set_node_attributes(vis_graph, important_gene_map, name='important gene')
    vis_graph.add_edges_from(vis_edges.keys())
    nx.set_edge_attributes(vis_graph, vis_edges)
    nx.write_gexf(vis_graph, f"vis{save_suffix}.gexf")
    print(data_cd.shape, len(vis_graph.nodes))
        
    # same thing for ego graph
    # for level in radiation_levels:
    #     ego_graphs[level] = nx.read_gexf(f"ego_dag_modularity_B_{level}.gexf")
    ego_vis_edges = dict()
    for graph_type, graph in ego_graphs.items():
        print(len(graph.nodes))
        for edge in graph.edges():
            if edge in ego_vis_edges.keys():
                ego_vis_edges[edge][graph_type] = True
            else:
                ego_vis_edges[edge] = dict(zip(graphs.keys(), len(graphs.keys())*[False]))
                ego_vis_edges[edge][graph_type] = True

            if graph_type=='ss':
                edge_undir = (edge[1], edge[1])
                if edge_undir in ego_vis_edges.keys():
                    ego_vis_edges[edge][graph_type] = True
                else:
                    ego_vis_edges[edge] = dict(zip(graphs.keys(), len(graphs.keys())*[False]))
                    ego_vis_edges[edge][graph_type] = True    
    ego_vis_graph = nx.DiGraph()
    print(len(gene_map.values()))
    ego_vis_graph.add_nodes_from(gene_map.values())
    nx.set_node_attributes(ego_vis_graph, partition_symbol, name='partition')
    
    nx.set_node_attributes(ego_vis_graph, important_gene_map, name='important gene')
    ego_vis_graph.add_edges_from(ego_vis_edges.keys())
    nx.set_edge_attributes(ego_vis_graph, ego_vis_edges)
    nx.write_gexf(ego_vis_graph, f"ego_vis{save_suffix}.gexf")
    

        


                

            
    
    
    