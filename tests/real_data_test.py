from cd_v_partition.causal_discovery import ges_local_learn, damga_local_learn, pc_local_learn, rfci_local_learn
from cd_v_partition.utils import get_scores
import bnlearn as bn
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def vis_graph(adj_mat, layout, ax):
    G = nx.from_numpy_array(adj_mat)
    nx.draw(G, pos=layout, ax=ax)
    
df = pd.read_csv("./datasets/bionetworks/human/sachs/sachs_raw.csv", sep=',',header=0)

#df = pd.read_csv("./datasets/bionetworks/human/sachs.interventional.txt", sep=' ',header=0)
#df = pd.read_csv("./datasets/bionetworks/human/sachs.data.txt", sep='\t',header=0)
print(df.shape)
dag = bn.import_DAG('sachs')['adjmat'].to_numpy()
#df = df.rename(columns={"Raf":1,  "Mek":2, "Plcg":3,  "PIP2":4,  "PIP3":5,  "Erk":6,  "Akt":7,  "PKA":8,  "PKC":9,  "P38":10, "Jnk":11, "INT":"target"})
df = df.rename(columns={"praf":1,  "pmek":2, "plcg":3,  "PIP2":4,  "PIP3":5,  "p44/42":6,  "pakts473":7,  "PKA":8,  "PKC":9,  "P38":10, "pjnk":11})
#df['target'] = np.zeros(df.shape[0])
print(set(df['target'].to_numpy()))
#df = df.drop(df[df.target >0].index)
num_nodes = df.shape[1] - 1
         
est_dag_ges = ges_local_learn([np.ones((num_nodes, num_nodes)), df], use_skel=False)
est_dag_dagma = damga_local_learn([np.ones((num_nodes, num_nodes)), df], use_skel=False)
est_dag_pc = pc_local_learn([np.ones((num_nodes, num_nodes)), df], use_skel=False)
est_dag_rfci = rfci_local_learn([np.ones((num_nodes, num_nodes)), df], use_skel=False)


print(df.shape)
s = get_scores(['GES'], [est_dag_ges], dag)
s = get_scores(['DAGMA'], [est_dag_dagma], dag)
s = get_scores(['PC'], [est_dag_pc], dag)
s = get_scores(['RFCI'], [est_dag_rfci], dag)
print(np.sum(dag))


