import logging
from pathlib import Path
from typing import Union

import numpy as np
from rpy2 import robjects as ro
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

rpy2_logger.setLevel(logging.ERROR)  # will display errors, but not warnings


def pc(
    data: np.ndarray, alpha: float, outdir: Union[Path, str], num_cores: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Python wrapper for the PC algorithm.

    Args:
        data (np.ndarray): Observational data with dimensions $n \times p$.
        alpha (float): Significance threshold to trim edges.
        outdir (Union[Path, str]): Directory to save adjacency matrix to.
        num_cores (int): Number of cpu cores to use during skeleton step of pc algorithm.

    Returns:
        A tuple containing two numpy arrays of dimensionality $p \times p$. The former `np.ndarray` represents the
            adjacency matrix for the CPDAG; the latter represents the significance level of each edge.
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
