from cd_v_partition.causal_discovery import ges_local_learn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from cd_v_partition.experiment import Experiment
from cd_v_partition.overlapping_partition import partition_problem
import functools
from concurrent.futures import ProcessPoolExecutor
import tqdm 
from cd_v_partition.config import SimulationSpec
import os
radbio_data = pd.read_csv("GSE43151_gs.csv")
num_genes = radbio_data.shape[1] - 1 
gene_ids = [int(i) for i in radbio_data.columns[0:num_genes]]
print(len(gene_ids))
gene_dictionary = pd.read_csv("ncbi_dataset.tsv", sep='\t', header=0)
print(gene_dictionary)

control = radbio_data[radbio_data['Dose'] == '0.Gy']
control_data = control.drop(columns=["Dose"])
for gene in np.arange(10):
    plt.hist(control_data.iloc[:,gene])

# Which genes are in the dataset and are also protein coding 
gene_data = gene_dictionary.loc[gene_dictionary['NCBI GeneID'].isin(gene_ids)]
protein_gene_data = gene_data.loc[gene_data['Gene Type']=='PROTEIN_CODING']
print(protein_gene_data)

final_gene_set = [str(i) for i in protein_gene_data['NCBI GeneID']]
final_data_set = radbio_data[final_gene_set]
final_column_set = final_gene_set + ["Dose"]
final_data_set_w_condition = radbio_data[final_column_set]

corr_mat = final_data_set.corr('pearson').to_numpy()
random_corr_coef = np.loadtxt('random_corr_coef.txt')

ci_interval = st.t.interval(0.95, len(random_corr_coef)-1, loc=np.mean(random_corr_coef), scale=st.sem(random_corr_coef))
cutoff = ci_interval[1]

corr_mat[corr_mat<=cutoff] = 0
corr_mat[corr_mat>cutoff] = 1
super_struct = corr_mat
print(f"Superstructure contains {np.sum(corr_mat)} edges which is {np.sum(corr_mat)/(corr_mat.shape[0]**2)} fraction of all possible edges")

final_data_set_w_condition = final_data_set_w_condition.rename(mapper={"Dose":"target"}, axis=1)
obs_final_data_set_w_condition = final_data_set_w_condition
obs_final_data_set_w_condition['target'] = np.zeros(radbio_data.shape[0])

spec = SimulationSpec(causal_learn_fn="GES", 
                      merge_fn="screen", 
                      partition_fn="modularity")
causal_discovery_alg = Experiment.get_causal_discovery_alg(spec)
merge_alg = Experiment.get_merge_alg(spec) 
partition_alg = Experiment.get_partitioning_alg(spec)

# Partition
partition = partition_alg(super_struct, data=obs_final_data_set_w_condition, resolution=5) 
# Learn in parallel
func_partial = functools.partial(ges_local_learn, maxDegree=100, use_skel= True)
results = []
subproblems = partition_problem(partition, super_struct, obs_final_data_set_w_condition)
workers = min(len(subproblems), os.cpu_count())
# workers=1 # Serial for debuggign 
print(f"Launching {workers} workers for partitioned run")

partition_sizes = [len(p) for p in partition.values()]
print(f"Biggest partition size {max(partition_sizes)}")
print(partition_sizes)
with ProcessPoolExecutor(max_workers=workers) as executor:
    results = list(tqdm.tqdm(executor.map(func_partial, subproblems, chunksize=1), total=len(subproblems)))

print("CD done")
# Merge
out_adj = merge_alg(ss=super_struct,partition=partition, local_cd_adj_mats=results,
            data= obs_final_data_set_w_condition.to_numpy(), 
            ss_subset=True, 
            finite_lim=False,
            full_cand_set=False)
                        
#est_dag = ges_local_learn((corr_mat, obs_final_data_set_w_condition), use_skel=True)
print(np.sum(out_adj))