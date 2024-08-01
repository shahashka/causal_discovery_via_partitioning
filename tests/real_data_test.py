from cd_v_partition.causal_discovery import sp_gies
from cd_v_partition.utils import get_scores
import bnlearn as bn
import numpy as np

df = bn.import_example('sachs')
print(df)
dag = bn.import_DAG('sachs')

df['target'] = np.ones(df.shape[0])
est_dag = sp_gies(df, skel=None,use_pc=False, outdir=None)

s = get_scores(['GES'], [est_dag], dag)
print(s)