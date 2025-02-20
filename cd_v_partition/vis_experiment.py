import warnings
from pathlib import Path
from typing import Any, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Define metric constants.
NDIM = 6
SHD, SID, AUC, TPR, FPR, TIME = range(NDIM)
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
    "no_partition": "No Partition",
    "no partition": "No Partition",
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
    """
    Loads a single iteration (or Monte-Carlo) run from a result file.

    Args:
        data (np.ndarray): The ``numpy`` array that represents a single
            iteration of data. This is effectively a row-vector in terms of
            dimensionality.

    Returns:
        A list of records (i.e., ``dict[str, Any]``), with the 5  metrics of
        interest (i.e., SHD, SId, AUC, TPR, and FPR) and the provided
        ``kwargs``.
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


def read_chkpoints(
    dir: Path | str,
    eval_algs: list[str],
    cd_alg: str,
    num_trials: int,
    save_sweep_values: Any,
) -> pd.DataFrame:
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
            inds, dirs = [], []
            # Make sure we iterate through specs in sequential orer
            for _, spec_path in enumerate(out_path.iterdir()):
                spec_id = int(
                    spec_path.name.split("_")[1]
                )  # grab the spec id from the folder
                inds.append(spec_id)
                dirs.append(spec_path)
            inds, dirs = zip(*sorted(zip(inds, dirs)))
            for spec_id, spec_path in enumerate(dirs):
                for trial_id in range(num_trials):
                    out_path = spec_path / f"trial_{trial_id}/chkpoint.txt"
                    if out_path.exists():
                        results[trial_id][spec_id] = np.loadtxt(out_path)
        # Populate records for each trial across specs
        for t, trial in enumerate(results):
            rec = load_iteration(trial, trial=t, method=alg)
            records.extend(rec)
    return pd.DataFrame.from_records(records)


def read_sizes(dir, eval_algs, cd_alg):
    trial_id = 0
    sizes_by_eval_alg = []
    for alg in eval_algs:
        out_path = Path(f"{dir}/{alg}/{cd_alg}")
        spec_path = next(out_path.iterdir())
        out_path = spec_path / f"trial_{trial_id}/sizes.txt"
        s = np.loadtxt(out_path)
        sizes_by_eval_alg.append(s)
    return sizes_by_eval_alg


def vis_experiment(
    experiment_id: int,
    dir: str,
    eval_algs: list[str],
    cd_alg: str,
    num_trials: int,
    save_sweep_param: str,
    save_sweep_values: Any,
):
    """Read checkpoints and visualize plots for scores from an experiment

    Plots the SHD, TPR and Time along the specified axis for the specified
    evaluation algorithms

    Args:
        experiment_id (int): index of experiment, used to specify format
            of plot
        dir (str): path to the save directory for the experiment
        eval_algs (list[str]): List of partitioning algorithms
        cd_alg (str): Name of the causal discovery algorithm e.g, GES
        num_trials (int): Number of trials/graphs per spec
        save_sweep_param (str): The name of the sweep parameter
            (x-axis label for plots)
        save_sweep_values (Any): The values for the sweep parameter
            (x-axis values for plots)
    """
    df = read_chkpoints(dir, eval_algs, cd_alg, num_trials, save_sweep_values)
    sizes = read_sizes(dir, eval_algs, cd_alg)
    for e, s in zip(eval_algs, sizes):
        plt.hist(s, label=e)
    plt.title("Partition sizes by algorithm")
    plt.legend()
    plt.savefig(f"{dir}/size_hist.png")
    plt.clf()

    df = df.replace(
        {"param": {i: val for (i, val) in enumerate(save_sweep_values)}}
    )
    df = df.rename(columns={"param": save_sweep_param})
    df = df[df.value != 0]
    df = df.reset_index()
    df.replace({"method": ALG_MAP}, inplace=True)
    df.to_csv(f"{dir}/scores_{experiment_id}_{cd_alg}.csv")
    match experiment_id:
        case 1:
            vis_gen(
                Path(dir),
                cd_alg,
                df,
                [ALG_MAP[e] for e in eval_algs],
                "num_samples",
                "# Samples",
                "log",
            )
        case 2:
            vis_gen(
                Path(dir),
                cd_alg,
                df,
                [ALG_MAP[e] for e in eval_algs],
                "inter_edge_prob",
                "Inter-community Edge Prob. ($\\rho$)",
                "linear",
            )
        case 3:
            vis_gen(
                Path(dir),
                cd_alg,
                df,
                [ALG_MAP[e] for e in eval_algs],
                "frac_extraneous_edges",
                "Fraction of Extraneous Edges",
                "linear",
            )
        case 4:
            vis_gen(
                Path(dir),
                cd_alg,
                df,
                [ALG_MAP[e] for e in eval_algs],
                "alpha",
                "$\\alpha$",
                "linear",
            )
        case 5:
            vis_5(Path(dir), cd_alg, df, [ALG_MAP[e] for e in eval_algs])
        case 6:
            print(df)
        case _:
            raise ValueError(f"`{experiment_id=}` is an illegal value.`")


def vis_gen(
    dir: Path | str,
    cd_alg: str,
    exp: pd.DataFrame,
    eval_algs: list[str],
    x_param: str,
    x_label: str,
    x_scale: str,
):
    tpr = exp.query("metric == 'TPR'")
    shd = exp.query("metric == 'SHD'")
    args = dict(
        x=x_param,
        y="value",
        hue="method",
        style="method",
        markers=MARKER_MAP,
        markersize=10,
        err_style=ERROR_STYLE,
        errorbar=ERROR_BAR,
        palette=COLOR_MAP,
        hue_order=eval_algs,
        style_order=eval_algs,
    )

    with sns.plotting_context("paper", font_scale=1.5):
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 5), sharex=True)
        sns.lineplot(tpr, ax=ax[0], **args)
        sns.lineplot(shd, ax=ax[1], **args)
        ax[0].set(xscale=x_scale)
        ax[1].set(xscale=x_scale)
        ax[0].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.6),
            ncol=3,
            title="Algorithm",
            frameon=False,
        )

        try:
            ax[1].get_legend().remove()
        except BaseException:
            pass

        ax[0].set_ylabel("TPR", weight="bold")
        ax[1].set_ylabel("SHD", weight="bold")
        ax[1].set_xlabel(x_label, weight="bold")
        plt.setp(ax[0].get_legend().get_title(), weight="bold")
        plt.subplots_adjust(hspace=0.05)
        plt.savefig(dir / f"{cd_alg}_tpr_shd.png", bbox_inches="tight")
        plt.clf()


def vis_5(
    dir: Path | str, cd_alg: str, exp5: pd.DataFrame, eval_algs: list[str]
):
    time = exp5.query("metric == 'TIME'")  # and num_nodes >= 10")
    args = dict(
        x="num_nodes",
        y="value",
        hue="method",
        style="method",
        markers=MARKER_MAP,
        markersize=10,
        err_style=ERROR_STYLE,
        errorbar=ERROR_BAR,
        palette=COLOR_MAP,
        hue_order=eval_algs,
        style_order=eval_algs,
    )
    with sns.plotting_context("paper", font_scale=1.5):
        ax = sns.lineplot(time, **args)
        ax.set(xscale="log", yscale="log")
        plt.ylabel("Runtime (sec.)", weight="bold")
        plt.xlabel("# Nodes", weight="bold")
        plt.legend(bbox_to_anchor=(1.0, 1.0), frameon=False, title="Algorithm")
        plt.setp(ax.get_legend().get_title(), weight="bold")
        plt.savefig(dir / f"{cd_alg}_timing.png", bbox_inches="tight")
        plt.clf()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # tmp = exp5.query("metric == 'TIME' and num_nodes == 10_000")
        # tpr = exp5.query("metric == 'TPR'  and num_nodes == 10_000")

        tmp = exp5.query("metric == 'TIME' and num_nodes == 1000")
        tpr = exp5.query("metric == 'TPR'  and num_nodes == 1000")

        tmp["hours"] = tmp.value.to_numpy() / 60 / 60  # Convert to hours
        tmp["TPR"] = tpr.value.to_numpy()

        gb = tmp.groupby(by=["method"])
        print(gb[["hours", "TPR"]].mean().to_latex())
