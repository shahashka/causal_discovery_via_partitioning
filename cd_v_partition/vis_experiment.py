import numpy as np
from pathlib import Path
import pandas as pd

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, TypeAlias

import matplotlib.pyplot as plt
import seaborn as sns
# Define metric constants.
NDIM = 6
SHD, SID, AUC, TPR, FPR, TIME = range(NDIM)
# SHD, AUC, TPR, FPR, TIME = range(NDIM - 1)
METRICS: dict[str, int] = {
    "SHD": SHD,
    "SID": SID,
    "AUC": AUC,
    "TPR": TPR,
    "FPR": FPR,
    "TIME": TIME,
}
METRIC_STRS: dict[int, str] = {val: key for key, val in METRICS.items()}

Record: TypeAlias = dict[str, Any]

# Vis formatting
ERROR_BAR = "ci"
ERROR_STYLE = "band"
ALG_MAP = {
    "expansive_causal": "Expansive Causal",
    "no_partition": "No Partition", "no partition": "No Partition",
    "PEF": "PEF",
    "modularity": "Disjoint",
    "edge_cover": "Edge Cover",
}
COLOR_MAP = {
    "Expansive Causal": "orange",
    "No Partition": "blue",
    "PEF": "green",
    "Disjoint": "grey",
    "Edge Cover": "red",
} 
MARKER_MAP = {
    "Expansive Causal": "X",
    "No Partition": "o",
    "PEF": "D",
    "Disjoint": "P",
    "Edge Cover": "s",
}
def load_iteration(data: np.ndarray, **kwargs) -> list[Record]:
    """Loads a single iteration (or Monte-Carlo) run from a result file.

    Args:
        data (np.ndarray): The ``numpy`` array that represents a single iteration of data.
            This is effectively a row-vector in terms of dimensionality.

    Returns:
        list[Record]: A lit of records (i.e., ``dict[str, Any]``), with the 5 metrics of interest
            (i.e., SHD, SId, AUC, TPR, and FPR) and the provided ``kwargs``.
    """
    shd = data[:, SHD]
    sid = data[:, SID]
    auc = data[:, AUC]
    tpr = data[:, TPR]
    fpr = data[:, FPR]
    time = data[:, TIME]

    records = []
    metrics = [shd, sid, auc, tpr, fpr, time]
    for name, idx in METRICS.items():
        if idx == SID:
            continue
        values = metrics[idx]
        for param, val in enumerate(values):
            rec = dict(metric=name, value=val, param=param, **kwargs)
            records.append(rec)

    return records

def read_chkpoints(dir: Path | str, eval_algs: list[str], cd_alg:str,
                   num_trials: int, 
                   save_sweep_values:Any) -> pd.DataFrame:
    """_summary_

    Args:
        dir (Path | str): _description_
        eval_algs (list[str]): _description_
        cd_alg (str): _description_
        num_trials (int): _description_
        save_sweep_values (Any): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # Read checkpoints
    records: list[Record] = []
    for alg in eval_algs:
        results = np.zeros((num_trials, len(save_sweep_values), NDIM))
        out_path = Path(f"{dir}/{alg}/{cd_alg}")
        if out_path.exists():
            for spec_id, spec_path in enumerate(out_path.iterdir()):
                for trial_id in range(num_trials):
                    out_path = spec_path / f"trial_{trial_id}/chkpoint.txt"
                    if out_path.exists():
                        results[trial_id][spec_id] = np.loadtxt(out_path)
        # Populate records for each trial across specs 
        for t, trial in enumerate(results):
            rec = load_iteration(trial, trial=t, method=alg)
            records.extend(rec)
    return pd.DataFrame.from_records(records)
        
def vis_experiment(experiment_id: int, dir: str, eval_algs: list[str], cd_alg:str,num_trials: int, 
                   save_sweep_param: str, save_sweep_values:Any):
    """Read checkpoints and visualize plots for scores from an experiment
    
    Plots the SHD, TPR and Time along the specified axis for the specified evaluation
    algorithms

    Args:
        experiment_id (int): index of experiment, used to specify format of plot
        dir (str): path to the save directory for the experiment
        eval_algs (list[str]): List of partitioning algorithms
        cd_alg (str): Name of the causal discovery algorithm e.g, GES
        num_trials (int): Number of trials/graphs per spec
        save_sweep_param (str): The name of the sweep parameter (x-axis label for plots)
        save_sweep_values (Any): The values for the sweep parameter (x-axis values for plots)
    """
    df = read_chkpoints(dir, eval_algs, cd_alg, num_trials, save_sweep_values )
    df = df.replace({
        "param": {
            i: val 
            for (i, val) in enumerate(save_sweep_values)
        }
    })
    df = df.rename(columns={"param": save_sweep_param})
    df = df[df.value != 0]
    df = df.reset_index()
    df.replace({"method": ALG_MAP}, inplace=True)
    hue_order, marker_order = dict(), dict()
    match experiment_id:
        case 1:
            vis_1(Path(dir), cd_alg, df, [ALG_MAP[e] for e in eval_algs]) 
        case _:
            raise ValueError(f"`{experiment_id=}` is an illegal value.`")

def vis_1(dir: Path | str, cd_alg:str, exp1: pd.DataFrame, eval_algs: list[str]): 
    tpr = exp1.query("metric == 'TPR'")
    shd = exp1.query("metric == 'SHD'")
    args = dict(
        x="num_samples", y="value", hue="method", style="method", 
        markers=MARKER_MAP, markersize=10, err_style=ERROR_STYLE, errorbar=ERROR_BAR,
        palette=COLOR_MAP, 
        hue_order=eval_algs, 
        style_order=eval_algs,
    )

    with sns.plotting_context("paper", font_scale=1.5):
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 5), sharex=True)
        sns.lineplot(tpr, ax=ax[0], **args)
        sns.lineplot(shd, ax=ax[1], **args)
        ax[0].set(xscale="log")
        ax[1].set(xscale="log")
        ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.6), ncol=3, title="Algorithm", frameon=False)
        ax[1].get_legend().remove()
        ax[0].set_ylabel("TPR", weight="bold")
        ax[1].set_ylabel("SHD", weight="bold")
        ax[1].set_xlabel("# Samples", weight="bold")
        plt.setp(ax[0].get_legend().get_title(), weight="bold")
        plt.subplots_adjust(hspace=0.05)
        plt.savefig(dir / f"{cd_alg}_tpr_shd.png", bbox_inches="tight")