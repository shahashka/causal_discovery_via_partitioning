from __future__ import annotations

# Causal discovery methods: cuPC, SP-GIES, etc... each with a specific set of
# assumptions that are assumed to be satisfied on subgraph. Runs local causal discovery on
# subgraphs to be merged later.
import itertools
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

CUPC_DIR = Path("./cupc/cuPC.R")

rpy2_logger.setLevel(logging.ERROR)  # will display errors, but not warnings
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr, SignatureTranslatedAnonymousPackage

rpy2.robjects.numpy2ri.activate()


pcalg = importr("pcalg")
base = importr("base")
GPU_AVAILABLE = os.path.exists("./Skeleton.so")


def pc(
    data: np.ndarray, alpha: float, outdir: Path | str, num_cores: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Python wrapper for the PC algorithm.

    Args:
        data (np.ndarray): Observational data with dimensions $n \times p$.
        alpha (float): Significance threshold to trim edges.
        outdir (Path | str): Directory to save adjacency matrix to.
        num_cores (int): Number of cpu cores to use during skeleton step of pc algorithm.

    Returns:
        A tuple containing two numpy arrays of dimensionality $p \times p$. The former
        `np.ndarray` represents the adjacency matrix for the CPDAG; the latter represents the
        significance level of each edge.
    """

    print("Running multicore CPU implementation of PC algorithm")
    ro.r.assign("data", data)
    rcode = "cor(data)"
    corMat = ro.r(rcode)
    ro.r.assign("correlationMatrix", corMat)

    p = data.shape[1]
    ro.r.assign("p", p)

    rcode = "list(C = correlationMatrix, n = nrow(data))"
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

    rcode = "pc_fit@pMax"
    p_values = ro.r(rcode)

    if outdir:
        d = str(outdir)
        rcode = f"write.csv(pdag,row.names = FALSE, file = paste('{d}/', 'pc-adj_mat.csv',sep = ''))"
        ro.r(rcode)
    return pdag, p_values


def cu_pc(
    data: np.ndarray, alpha: float, outdir: Path | str
) -> tuple[np.ndarray, np.ndarray] | None:
    r"""
    Python wrapper for cuPC. CUDA implementation of the PC algorithm

    Args:
        data (np.ndarray): Observational data with dimensions $n \times p$.
        alpha (float): Significance threshold to trim edges.
        outdir (Path | str): The directory to save adjacency matrix to.

    Returns:
        Tuple of numpy arrays of dimension $p \times p$. The former array is the adjacency matrix
        for the CPDAG; the latter represents the significance level of each edge.
    """
    if not GPU_AVAILABLE:
        print("No compiled Skeleton.so file")
        return
    print("Running GPU implementation of PC algorithm")
    with open(CUPC_DIR) as file:
        string = "".join(file.readlines())
    cupc = SignatureTranslatedAnonymousPackage(string, "cupc")
    ro.r.assign("data", data)
    rcode = "cor(data)"
    corMat = ro.r(rcode)
    ro.r.assign("corrolationMatrix", corMat)

    p = data.shape[1]
    ro.r.assign("p", p)

    rcode = "list(C = corrolationMatrix, n = nrow(data))"
    suffStat = ro.r(rcode)
    ro.r.assign("suffStat", suffStat)

    cuPC_fit = cupc.cu_pc(ro.r["suffStat"], p=ro.r["p"], alpha=alpha, u2pd="rand")
    ro.r.assign("cuPC_fit", cuPC_fit)

    rcode = 'as(cuPC_fit@graph, "matrix")'
    skel = ro.r(rcode)
    ro.r.assign("skel", skel)

    rcode = 'as(cuPC_fit@pMax, "matrix")'
    p_values = ro.r(rcode)

    if outdir:
        rcode = "write.csv(skel,row.names = FALSE, file = paste('{}/', 'cupc-adj_mat.csv',sep = ''))".format(
            outdir
        )
        ro.r(rcode)
    return skel, p_values


def sp_gies(
    data: pd.DataFrame,
    outdir: Path | str,
    alpha: float = 1e-3,
    skel: np.ndarray = None,
    use_pc: bool = True,
    multifactor_targets: list[list[Any]] = None,
    adaptive: bool = True,
):
    r"""
    Python wrapper for SP-GIES. Uses skeleton estimation to restrict edge set to GIES learner

    Args:
        data (pd.DataFrame): DataFrame containing observational and interventional samples.
            Must contain a column named 'target' which specifies the index of the node that
            was intervened on to obtain the sample (assumes single interventions only). This
            indexes from 1 for R convenience. For observational samples the corresponding
            target should be 0.
        outdir (Path | str): The directory to save the final adjacency matrix named
            `sp-gies-adj_mat.csv`. Set to None to skip saving files.
        alpha (float): Significance threshold to trim edges.
        skel (np.ndarray): An optional initial skeleton with dimensions $p \times p$.
        use_pc (bool): A flag to indicate if skeleton estimation should be done with the PC. If `False`
            and no skel is specified, then assumed no skeleton i.e., reverts to GIES algorithm.
            Will use the GPU accelerated version of the PC if available, otherwise reverts to pcalg
            implementation of PC.
        multifactor_targets (list[list[Any]]): An optional list of lists for when there are
            multinode targets. In this case it is assumed that the 'target' column of the
            data DataFrame contains the index into this list.
        adaptive (bool): # TODO

    Returns:
        Array representing the adjacency matrix for the final learned graph.
    """
    # When the dataset only has one node (possible when partitioning)
    if data.shape[1] == 2:
        adj_mat = np.ones(1)
        df = pd.DataFrame(data=adj_mat)
        if outdir:
            df.to_csv("{}/sp-gies-adj_mat.csv".format(outdir), header=False, index=False)
        return adj_mat

    if skel is None:
        obs_data = data.loc[data["target"] == 0]
        obs_data = obs_data.drop(columns=["target"])
        obs_data = obs_data.to_numpy(dtype=float)
        if use_pc:
            skel = pc(obs_data, alpha, outdir, num_cores=8)[
                0
            ]  # cu_pc(obs_data, alpha, outdir) if GPU_AVAILABLE else

        else:
            skel = np.ones((data.shape[1], data.shape[1]))

    fixed_gaps = np.array((skel == 0), dtype=int)
    target_index = data.loc[:, "target"].to_numpy()
    targets = (
        multifactor_targets if multifactor_targets else np.unique(target_index)[1:]
    )  # Remove 0 the observational target
    # TODO with interventional data do the names and target_ids match?
    target_index_R = target_index + 1  # R indexes from 1
    data = data.drop(columns=["target"]).to_numpy(dtype=float)

    nr, nc = data.shape
    D = ro.r.matrix(data, nrow=nr, ncol=nc)
    ro.r.assign("data", D)

    def unroll_target(t):
        rcode = ",".join(str(int(i)) for i in t)
        return rcode

    if len(targets) > 0:
        rcode = ",".join(
            "c({})".format(unroll_target(i)) if type(i) == list else str(int(i))
            for i in targets
        )
        rcode = "append(list(integer(0)), list({}))".format(rcode)
        T = ro.r(rcode)
        ro.r.assign("targets", T)
    else:
        rcode = "list(integer(0))"
        T = ro.r(rcode)
        ro.r.assign("targets", T)

    TI = ro.IntVector(target_index_R)
    ro.r.assign("target_index", TI)

    nr, nc = fixed_gaps.shape
    FG = ro.r.matrix(fixed_gaps, nrow=nr, ncol=nc)
    ro.r.assign("fixed_gaps", FG)
    if data.shape[1] > 1:
        score = ro.r.new(
            "GaussL0penIntScore", ro.r["data"], ro.r["targets"], ro.r["target_index"]
        )
        ro.r.assign("score", score)
        if adaptive:
            result = pcalg.gies(
                ro.r["score"],
                fixedGaps=ro.r["fixed_gaps"],
                targets=ro.r["targets"],
                adaptive="triples",
            )
        else:
            result = pcalg.gies(
                ro.r["score"], fixedGaps=ro.r["fixed_gaps"], targets=ro.r["targets"]
            )

        ro.r.assign("result", result)

        rcode = "result$repr$weight.mat()"  # weight: ith column contains the regression coefficients of the ith stuctural equation
        adj_mat = ro.r(rcode)
    else:
        adj_mat = np.zeros((1, 1))
    ro.r.assign("adj_mat", adj_mat)
    if outdir:
        rcode = (
            "write.csv(adj_mat, row.names = FALSE,"
            ' file = paste("{}/", "sp-gies-adj_mat.csv", sep=""))'.format(outdir)
        )
        ro.r(rcode)
    return adj_mat


def weight_colliders(adj_mat: np.ndarray, weight: int = 1):
    r"""
    Find and add weights to collider sets in a given adjacency matrix. Collider sets are x->y<-z
    when there is no edge between $(x,z)$.

    Args:
        adj_mat (np.ndarray): $p \times p$ adjacency matrix.
        weight (int): Edges that are part of a collider set are weighted with this weight.

    Returns:
        An array representing the weighted adjacency matrix.
    """
    weighted_adj_mat = adj_mat
    for col in np.arange(adj_mat.shape[1]):
        incident_nodes = np.argwhere(adj_mat[:, col] == 1).flatten()

        # For all edges incident on the node corresponding to this column
        for i, j in itertools.combinations(incident_nodes, 2):
            # Filter for directed edges
            if adj_mat[col, i] == 0 and adj_mat[col, j] == 0:
                # Check if a pair of source nodes is connected
                if adj_mat[i, j] == 0 and adj_mat[j, i] == 0:
                    # If not then this is a collider
                    weighted_adj_mat[i, col] = weight
                    weighted_adj_mat[j, col] = weight
    return weighted_adj_mat
