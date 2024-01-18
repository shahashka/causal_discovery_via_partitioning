import logging
import os
from pathlib import Path
from typing import Union, Optional

import numpy as np
from rpy2 import robjects as ro
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage

rpy2_logger.setLevel(logging.ERROR)  # will display errors, but not warnings

CUPC_DIR = Path("./cupc/cuPC.R")
GPU_AVAILABLE = os.path.exists("./Skeleton.so")


def cu_pc(
    data: np.ndarray, alpha: float, outdir: Union[Path, str]
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Python wrapper for cuPC. CUDA implementation of the PC algorithm

    Args:
        data (np.ndarray): Observational data with dimensions $n \\times p$.
        alpha (float): Significance threshold to trim edges.
        outdir (Union[Path, str]): The directory to save adjacency matrix to.

    Returns:
        Tuple of numpy arrays of dimension $p \\times p$. The former array is the adjacency matrix
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
