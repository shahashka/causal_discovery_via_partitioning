import logging
from pathlib import Path
from typing import Union, Any

import numpy as np
import pandas as pd
import rpy2.robjects
from rpy2 import robjects as ro
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

from cd_v_partition.algorithms.causal_discovery.pc import pc

rpy2_logger.setLevel(logging.ERROR)  # will display errors, but not warnings


def sp_gies(
    data: pd.DataFrame,
    outdir: Union[Path, str],
    alpha: float = 1e-3,
    skel: np.ndarray = None,
    pc_flag: bool = True,
    multifactor_targets: list[list[Any]] = None,
    adaptive: bool = True,
):
    """
    Python wrapper for SP-GIES. Uses skeleton estimation to restrict edge set to GIES learner

    Args:
        data (pd.DataFrame): DataFrame containing observational and interventional samples.
            Must contain a column named 'target' which specifies the index of the node that
            was intervened on to obtain the sample (assumes single interventions only). This
            indexes from 1 for R convenience. For observational samples the corresponding
            target should be 0.
        outdir (Union[Path, str]): The directory to save the final adjacency matrix named
            `sp-gies-adj_mat.csv`. Set to None to skip saving files.
        alpha (float): Significance threshold to trim edges.
        skel (np.ndarray): An optional initial skeleton with dimensions $p \\times p$.
        pc_flag (bool): A flag to indicate if skeleton estimation should be done with the PC. If `False`
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

    from rpy2.robjects.packages import importr

    rpy2.robjects.numpy2ri.activate()
    base = importr("base")
    pcalg = importr("pcalg")

    # When the dataset only has one node (possible when partitioning)
    if data.shape[1] == 2:
        adj_mat = np.ones(1)
        df = pd.DataFrame(data=adj_mat)
        df.to_csv("{}/sp-gies-adj_mat.csv".format(outdir), header=False, index=False)
        return adj_mat

    if skel is None:
        obs_data = data.loc[data["target"] == 0]
        obs_data = obs_data.drop(columns=["target"])
        obs_data = obs_data.to_numpy(dtype=float)
        if pc_flag:
            skel = pc(
                obs_data, alpha, outdir, num_cores=8
            )  # cu_pc(obs_data, alpha, outdir) if GPU_AVAILABLE else
        else:
            skel = np.ones((data.shape[1], data.shape[1]))

    fixed_gaps = np.array((skel == 0), dtype=int)
    try:
        print(data.head())
        print(data.loc[:, "target"].head())
    except KeyError as err:
        print(f"{data.columns=}")
        raise err
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
