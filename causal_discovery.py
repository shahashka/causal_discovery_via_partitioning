# Causal discovery methods: cuPC, SP-GIES, etc... each with a specific set of assumptions that are assumed to be satisfied on subgraph
# Runs local causal discovery on subgraphs to be merged later
import pandas as pd
import numpy as np
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
import logging
rpy2_logger.setLevel(logging.ERROR)   # will display errors, but not warnings
rpy2.robjects.numpy2ri.activate()
import os
pcalg = importr('pcalg')
base = importr('base')

def pc(data, alpha, outdir, num_cores=8):
    '''
      Python wrapper for PC.

               Parameters:
                       data (numpy ndarray): Observational data with dimensions n x p
                       alpha (float): significance threshold to trim edges
                       outdir (str): directory to save adjacency matrix to 
                       num_cores (int): Number of cpu cores to use during skeleton step of pc algorithm 

               Returns:
                       np.ndarray representing the adjancency matrix for the cpdag with dimensions p x p
                       np.ndarray representing the signifance level of each edge with dimensions p x p
       '''

    print("Running multicore CPU implementation of PC algorithm")
    ro.r.assign("data", data)
    rcode = 'cor(data)'
    corMat = ro.r(rcode)
    ro.r.assign("corrolationMatrix", corMat)

    p = data.shape[1]
    ro.r.assign("p",p)

    rcode = 'list(C = corrolationMatrix, n = nrow(data))'
    suffStat = ro.r(rcode)
    ro.r.assign("suffStat", suffStat)

    ro.r.assign("alpha", alpha)
    ro.r.assign("num_cores", num_cores)
    rcode = 'pc(suffStat,p=p,indepTest=gaussCItest,skel.method="stable.fast",alpha=alpha, numCores=num_cores)'
    pc_fit = ro.r(rcode)
    ro.r.assign("pc_fit", pc_fit)

    rcode = 'as(pc_fit@graph, "matrix")'
    pdag = ro.r(rcode)
    ro.r.assign("pdag", pdag)
    
    rcode = 'pc_fit@pMax'
    p_values = ro.r(rcode)

    rcode = "write.csv(pdag,row.names = FALSE, file = paste('{}/', 'pc-adj_mat.csv',sep = ''))".format(outdir)
    ro.r(rcode)
    return pdag, p_values 

def cu_pc(data, alpha, outdir):
    '''
      Python wrapper for cuPC. CUDA implementation of the PC algorithm

               Parameters:
                       data (numpy ndarray): Observational data with dimensions n x p
                       alpha (float): significance threshold to trim edges
                       outdir (str): directory to save adjacency matrix to 
               Returns:
                       np.ndarray representing the adjancency matrix for the cpdag with dimensions p x p
                       np.ndarray representing the signifance level of each edge with dimensions p x p    
    '''
    
    print("Running GPU implementation of PC algorithm")
    with open("./cupc/cuPC.R") as file:
        string = ''.join(file.readlines())
    cupc = SignatureTranslatedAnonymousPackage(string, "cupc")
    ro.r.assign("data", data)
    rcode = 'cor(data)'
    corMat = ro.r(rcode)
    ro.r.assign("corrolationMatrix", corMat)

    p = data.shape[1]
    ro.r.assign("p",p)

    rcode = 'list(C = corrolationMatrix, n = nrow(data))'
    suffStat = ro.r(rcode)
    ro.r.assign("suffStat", suffStat)

    cuPC_fit = cupc.cu_pc(ro.r['suffStat'],p=ro.r['p'],alpha=alpha, u2pd='rand')
    ro.r.assign("cuPC_fit", cuPC_fit)

    rcode = 'as(cuPC_fit@graph, "matrix")'
    skel = ro.r(rcode)
    ro.r.assign("skel", skel)
    
    rcode = 'as(cuPC_fit@pMax, "matrix")'
    p_values = ro.r(rcode)

    rcode = "write.csv(skel,row.names = FALSE, file = paste('{}/', 'cupc-adj_mat.csv',sep = ''))".format(outdir)
    ro.r(rcode)
    return skel, p_values


def sp_gies(data, outdir, skel=None, pc=True,  target_map=None, multifactor_targets=None, adaptive=True):
    '''
      Python wrapper for SP-GIES. Uses skeleton estimation to restrict edge set to GIES learner

               Parameters:
                       data (pandas DataFrame): DataFrame containing observational and interventional samples.
                                                Must contain a column named 'target' which specifies the index of the node
                                                that was intervened on to obtain the sample (assumes single interventions only).
                                                This indexes from 1 for R convenience.
                                                For observational samples the corresponding target should be 0
                       outdir (str): The directory to save the final adjacency matrix named sp-gies-adj_mat.csv 
                       skel (numpy ndarray): an optional initial skeleton with dimensions p x p
                       pc (bool): a flag to indicate if skeleton estimation should be done with the PC. If False
                                    and no skel is specified, then assumed no skeleton i.e. reverts to GIES algorithm.
                                    Will use the GPU accelerated version of the PC if avaiable, otherwise reverts to pcalg
                                    implementation of PC
                        target_map (dict): An optional dictionary to map the 'target' column of the input dataset to the indices
                                         in the dataframe. This is only needed for the parallel implementation of SP-GIES where the
                                        full graph is partitioned and indices need to be tracked
                        multifactor_targets (list): An optional list of lists for when there are multinode targets. In this case it is
                                                    assumed that the 'target' column of the data DataFrame contains the index into this list
               Returns:
                       np.ndarray representing the adjancency matrix for the final learned graph
       '''
    # When the dataset only has one node (possible when partitioning)
    if data.shape[1] == 2:
        adj_mat =  np.ones(1)
        df = pd.DataFrame(data=adj_mat)
        df.to_csv("{}/sp-gies-adj_mat.csv".format(outdir), header=False, index=False)
        return adj_mat
    if skel is None:
        obs_data = data.loc[data['target']==0]
        obs_data = obs_data.drop(columns=['target'])
        obs_data = obs_data.to_numpy(dtype=float)
        if pc:
            skel = pc(obs_data, outdir)
        else:
            skel = np.ones((data.shape[1], data.shape[1]))
    fixed_gaps = np.array((skel == 0), dtype=int)
    print("Fixed gaps {}".format(sum(sum(fixed_gaps))))
    target_index = data.loc[:, 'target'].to_numpy()
    if target_map is not None:
        target_index = np.array([0 if i == 0 else target_map[i] + 1 for i in target_index])
    targets = multifactor_targets if multifactor_targets else np.unique(target_index)[1:]  # Remove 0 the observational target
    target_index_R = target_index + 1  # R indexes from 1
    data = data.drop(columns=['target']).to_numpy(dtype=float)

    nr, nc = data.shape
    D = ro.r.matrix(data, nrow=nr, ncol=nc)
    ro.r.assign("data", D)
    def unroll_target(t):
        rcode  = ','.join(str(int(i)) for i in t)
        return rcode
    if len(targets) > 0:
        rcode  = ','.join('c({})'.format(unroll_target(i)) if type(i) == list else str(int(i)) for i in targets)
        rcode = 'append(list(integer(0)), list({}))'.format(rcode)
        T = ro.r(rcode)
        ro.r.assign("targets", T)
    else:
        rcode = 'list(integer(0))'
        T = ro.r(rcode)
        ro.r.assign("targets", T)

    TI = ro.IntVector(target_index_R)
    ro.r.assign("target_index", TI)

    nr, nc = fixed_gaps.shape
    FG = ro.r.matrix(fixed_gaps, nrow=nr, ncol=nc)
    ro.r.assign("fixed_gaps", FG)
    if data.shape[1] > 1:
        score = ro.r.new('GaussL0penIntScore', ro.r['data'], ro.r['targets'], ro.r['target_index'])
        ro.r.assign("score", score)
        if adaptive:
            result = pcalg.gies(ro.r['score'], fixedGaps=ro.r['fixed_gaps'], targets=ro.r['targets'], adaptive='triples')
        else: 
            result = pcalg.gies(ro.r['score'], fixedGaps=ro.r['fixed_gaps'], targets=ro.r['targets'])

        ro.r.assign("result", result)

        rcode = 'result$repr$weight.mat()'
        adj_mat = ro.r(rcode)
    else:
        adj_mat = np.zeros((1,1))
    ro.r.assign("adj_mat", adj_mat)
    rcode = 'write.csv(adj_mat, row.names = FALSE,' \
            ' file = paste("{}/", "sp-gies-adj_mat.csv", sep=""))'.format(outdir)
    ro.r(rcode)
    return adj_mat