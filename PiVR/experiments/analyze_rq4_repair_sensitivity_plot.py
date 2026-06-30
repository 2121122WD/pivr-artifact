from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(r"D:\paper_methods\CP_Repair_with_baseline\CCBR\PiVR\experiments_birdnn\results\rq4_repair_stage_stability_result.csv")
DEFAULT_OUT_DIR = Path(r"D:\paper_methods\CP_Repair_with_baseline\CCBR\PiVR\experiments_birdnn\results")
DEFAULT_FIG_DIR = Path(r"D:\paper_methods\CP_Repair_with_baseline\CCBR\ACM_Conference_Proceedings_Primary_Article_Template\figures")

TASK_ORDER = {"Backdoor": 0, "Safety": 1, "Fairness": 2}
METRIC_LABEL = {"Backdoor": "ASR", "Safety": "VR", "Fairness": "IDR"}
DEFAULT_PARAM_VALUE = {"lambda_clean": 0.5}


def safe_name(s: str) -> str:
    s = str(s).replace(",", "_").replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def format_param_label(param_name: str) -> str:
    if param_name == "lambda_clean":
        return r"$\lambda_{clean}$"
    if param_name == "lambda_task":
        return r"$\lambda_{task}$"
    return param_name


def setting_id(row_or_df) -> str:
    if isinstance(row_or_df, pd.DataFrame):
        row = row_or_df.iloc[0]
    else:
        row = row_or_df
    attr = row.get("protected_attr", "none")
    if pd.isna(attr) or attr in ("", "none"):
        return f"{row['model']}/{row['dataset']}"
    return f"{row['model']}/{row['dataset']}-{attr}"


def _parse_float(x) -> float:
    if x is None or x == "":
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def load_and_normalize(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)
        for raw in reader:
            if len(raw) < 17:
                continue
            if len(raw) >= 18 and raw[5] == "full":
                # Legacy ablation-like format.
                task = raw[1]
                rows.append({
                    "timestamp": raw[0],
                    "task": task,
                    "model": raw[2],
                    "dataset": raw[3],
                    "protected_attr": raw[4] or "none",
                    "seed": 0,
                    "param_name": raw[6],
                    "param_value": _parse_float(raw[7]),
                    "repair_metric_name": METRIC_LABEL.get(task, "Metric"),
                    "repair_metric_before": _parse_float(raw[8]),
                    "repair_metric_after": _parse_float(raw[9]),
                    "acc_before": _parse_float(raw[10]),
                    "acc_after": _parse_float(raw[11]),
                    "loc_time": _parse_float(raw[14]),
                    "ver_time": _parse_float(raw[15]),
                    "repair_time": _parse_float(raw[16]),
                    "total_time": _parse_float(raw[17]),
                })
            else:
                # Standard RQ5 format without drawdown.
                rows.append({
                    "timestamp": raw[0],
                    "task": raw[1],
                    "model": raw[2],
                    "dataset": raw[3],
                    "protected_attr": raw[4] or "none",
                    "seed": int(float(raw[5])),
                    "param_name": raw[6],
                    "param_value": _parse_float(raw[7]),
                    "repair_metric_name": raw[8],
                    "repair_metric_before": _parse_float(raw[9]),
                    "repair_metric_after": _parse_float(raw[10]),
                    "acc_before": _parse_float(raw[11]),
                    "acc_after": _parse_float(raw[12]),
                    "loc_time": _parse_float(raw[13]),
                    "ver_time": _parse_float(raw[14]),
                    "repair_time": _parse_float(raw[15]),
                    "total_time": _parse_float(raw[16]),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No valid rows loaded from {path}")

    for task in df["task"].dropna().unique():
        mask = df["task"] == task
        for col in ["repair_metric_before", "repair_metric_after", "acc_before", "acc_after"]:
            vals = df.loc[mask, col].dropna().astype(float)
            if not vals.empty and vals.abs().max() <= 1.5:
                df.loc[mask, col] = df.loc[mask, col] * 100.0

    df["setting"] = df.apply(setting_id, axis=1)
    df["task_order"] = df["task"].map(TASK_ORDER).fillna(99)
    df = df.sort_values(["task_order", "setting", "param_name", "param_value", "seed"])
    return df.drop(columns=["task_order"])


def summarize(df: pd.DataFrame, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "rq4_repair_stage_normalized.csv", index=False)

    group_cols = ["task", "model", "dataset", "protected_attr", "setting", "param_name", "param_value"]
    setting_summary = (
        df.groupby(group_cols)
        .agg(
            metric_mean=("repair_metric_after", "mean"),
            metric_std=("repair_metric_after", "std"),
            acc_mean=("acc_after", "mean"),
            acc_std=("acc_after", "std"),
            n_runs=("repair_metric_after", "count"),
        )
        .reset_index()
    )
    setting_summary.to_csv(out_dir / "rq4_repair_stage_setting_summary.csv", index=False)

    task_summary = (
        setting_summary.groupby(["task", "param_name", "param_value"])
        .agg(
            metric_mean=("metric_mean", "mean"),
            metric_std=("metric_mean", "std"),
            acc_mean=("acc_mean", "mean"),
            acc_std=("acc_mean", "std"),
            n_settings=("setting", "nunique"),
        )
        .reset_index()
    )
    task_summary.to_csv(out_dir / "rq4_repair_stage_task_summary.csv", index=False)

    stability_rows = []
    for (task, param), sub in setting_summary.groupby(["task", "param_name"]):
        metric_min, metric_max = sub["metric_mean"].min(), sub["metric_mean"].max()
        acc_min, acc_max = sub["acc_mean"].min(), sub["acc_mean"].max()

        default_value = DEFAULT_PARAM_VALUE.get(param, sub["param_value"].median())
        stable_count = 0
        total_count = 0
        for _, ss in sub.groupby("setting"):
            default_idx = (ss["param_value"] - default_value).abs().idxmin()
            default_metric = float(ss.loc[default_idx, "metric_mean"])
            default_acc = float(ss.loc[default_idx, "acc_mean"])
            metric_tol = max(0.5, 0.25 * max(default_metric, 1.0))
            acc_tol = 2.0
            ok = (ss["metric_mean"] <= default_metric + metric_tol) & (ss["acc_mean"] >= default_acc - acc_tol)
            stable_count += int(ok.sum())
            total_count += int(len(ok))
        stable_rate = 100.0 * stable_count / total_count if total_count else np.nan

        metric_span = metric_max - metric_min
        acc_span = acc_max - acc_min
        if stable_rate >= 80 and metric_span <= 2.0 and acc_span <= 2.0:
            observation = "Stable"
        elif stable_rate >= 60:
            observation = "Mostly stable"
        else:
            observation = "Parameter-sensitive"

        stability_rows.append({
            "Task": task,
            "Parameter": param,
            "Settings": sub["setting"].nunique(),
            "Metric Range": f"{metric_min:.2f}--{metric_max:.2f}",
            "ACC Range": f"{acc_min:.2f}--{acc_max:.2f}",
            "Stable Configs": f"{stable_rate:.1f}%",
            "Observation": observation,
        })
    stability_table = pd.DataFrame(stability_rows)
    stability_table.to_csv(out_dir / "rq4_repair_stage_stability_table.csv", index=False)
    return setting_summary, task_summary, stability_table


def _plot_param_panel(ax, ax2, sub: pd.DataFrame, task: str, param: str, title: str):
    sub = sub.sort_values("param_value")
    x = np.arange(len(sub))
    labels = [f"{v:g}" for v in sub["param_value"]]
    metric = sub["metric_mean"].to_numpy(float)
    metric_std = sub["metric_std"].fillna(0).to_numpy(float)
    acc = sub["acc_mean"].to_numpy(float)
    acc_std = sub["acc_std"].fillna(0).to_numpy(float)

    line1 = ax.plot(x, metric, marker="o", linewidth=1.7, markersize=4, label=f"{METRIC_LABEL.get(task, 'Metric')} after")
    if np.any(metric_std > 0):
        ax.fill_between(x, metric - metric_std, metric + metric_std, alpha=0.15)
    line2 = ax2.plot(x, acc, marker="s", linestyle="--", linewidth=1.5, markersize=3.5, label="ACC")
    if np.any(acc_std > 0):
        ax2.fill_between(x, acc - acc_std, acc + acc_std, alpha=0.10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel(format_param_label(param), fontsize=9)
    ax.set_ylabel(f"{METRIC_LABEL.get(task, 'Metric')} ↓", fontsize=9)
    ax2.set_ylabel("ACC ↑", fontsize=9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.set_title(title, fontsize=9)
    lines = line1 + line2
    ax.legend(lines, [l.get_label() for l in lines], fontsize=7, loc="best", frameon=True)


def plot_task_average(task_summary: pd.DataFrame, fig_dir: Path, out_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(8.2, 7.2), constrained_layout=True)
    for i, task in enumerate(["Backdoor", "Safety", "Fairness"]):
        for j, param in enumerate(["lambda_clean", "lambda_task"]):
            ax = axes[i, j]
            ax2 = ax.twinx()
            sub = task_summary[(task_summary["task"] == task) & (task_summary["param_name"] == param)]
            if sub.empty:
                ax.set_visible(False)
                ax2.set_visible(False)
                continue
            _plot_param_panel(ax, ax2, sub, task, param, f"{task} / {format_param_label(param)}")
    out = fig_dir / "rq4_repair_stage_stability_taskavg.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / out.name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_selected_settings(setting_summary: pd.DataFrame, fig_dir: Path, out_dir: Path):
    preferred = {
        "Backdoor": ["NN2/MNIST", "NN2/CIFAR-10"],
        "Safety": ["ACAS_Xu/N2,9", "ACAS_Xu/N1,9"],
        "Fairness": ["NN9/bank-age", "NN10/credit-age", "NN10/credit-gender"],
    }
    selected = []
    for task, sub in setting_summary.groupby("task"):
        available = list(sub["setting"].unique())
        chosen = [s for s in preferred.get(task, []) if s in available]
        if not chosen:
            chosen = available[:1]
        for s in chosen[:2]:
            selected.append((task, s))

    if not selected:
        return
    fig, axes = plt.subplots(len(selected), 2, figsize=(8.2, max(2.2 * len(selected), 4.0)), constrained_layout=True)
    if len(selected) == 1:
        axes = np.array([axes])
    for i, (task, setting) in enumerate(selected):
        for j, param in enumerate(["lambda_clean", "lambda_task"]):
            ax = axes[i, j]
            ax2 = ax.twinx()
            sub = setting_summary[(setting_summary["task"] == task) & (setting_summary["setting"] == setting) & (setting_summary["param_name"] == param)]
            if sub.empty:
                ax.set_visible(False)
                ax2.set_visible(False)
                continue
            _plot_param_panel(ax, ax2, sub, task, param, f"{setting} / {format_param_label(param)}")
    out = fig_dir / "rq4_repair_stage_stability_selected.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / out.name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze PiVR RQ4 repair-stage parameter stability results.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_normalize(args.input)
    setting_summary, task_summary, stability_table = summarize(df, args.out_dir)
    plot_task_average(task_summary, args.fig_dir, args.out_dir)
    plot_selected_settings(setting_summary, args.fig_dir, args.out_dir)

    print(f"[INFO] Normalized CSV: {args.out_dir / 'rq4_repair_stage_normalized.csv'}")
    print(f"[INFO] Setting summary: {args.out_dir / 'rq4_repair_stage_setting_summary.csv'}")
    print(f"[INFO] Task summary: {args.out_dir / 'rq4_repair_stage_task_summary.csv'}")
    print(f"[INFO] Stability table: {args.out_dir / 'rq4_repair_stage_stability_table.csv'}")
    print(f"[INFO] Task-average figure: {args.fig_dir / 'rq4_repair_stage_stability_taskavg.png'}")
    print(f"[INFO] Selected-setting figure: {args.fig_dir / 'rq4_repair_stage_stability_selected.png'}")
    print("\nStability table:")
    print(stability_table.to_string(index=False))


if __name__ == "__main__":
    main()
