import os
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fmt_mean_std(mean, std, digits=3):
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description="Analyze CP-Repair RQ3 ablation results")
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join("cp_repair", "experiments", "results", "rq4_k_result.csv"),
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=os.path.join("cp_repair", "experiments", "results", "rq3_ablation_summary.csv"),
    )
    parser.add_argument(
        "--table_tex",
        type=str,
        default=os.path.join("cp_repair", "experiments", "results", "rq3_ablation_table.tex"),
    )
    parser.add_argument(
        "--figure_png",
        type=str,
        default=os.path.join("cp_repair", "experiments", "results", "rq3_ablation_tradeoff.png"),
    )
    args = parser.parse_args()

    import csv

    rows = []
    with open(args.input, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        _ = next(reader, None)
        for row in reader:
            if not row:
                continue
            if len(row) < 19:
                continue
            rows.append({
                "timestamp": row[0],
                "task": row[1],
                "model": row[2],
                "dataset": row[3],
                "protected_attr": row[4],
                "ablation": row[5],
                "seed": row[6],
                "repair_metric_name": row[7],
                "repair_metric_before": row[8],
                "repair_metric_after": row[9],
                "acc_before": row[10],
                "acc_after": row[11],
                "drawdown": row[12],
                "modified_params": row[13],
                "modified_params_ratio": row[14],
                "loc_time": row[15],
                "ver_time": row[16],
                "repair_time": row[17],
                "total_time": row[18],
            })

    df = pd.DataFrame(rows)
    for col in ["repair_metric_before", "repair_metric_after", "acc_before", "acc_after", "drawdown", "modified_params_ratio", "loc_time", "ver_time", "repair_time", "total_time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    group_cols = ["task", "model", "dataset", "protected_attr", "ablation"]
    agg_cols = ["repair_metric_after", "acc_after", "drawdown", "modified_params_ratio", "total_time"]

    summary = (
        df.groupby(group_cols, as_index=False)[agg_cols]
        .agg(["mean", "std"])
    )
    summary.columns = ["_".join([c for c in col if c]).strip("_") for col in summary.columns.to_flat_index()]

    summary.to_csv(args.summary_csv, index=False)

    # Prepare table rows
    task_metric_name = {"Backdoor": "ASR", "Safety": "VR", "Fairness": "IDR"}
    rows = []
    for task in ["Backdoor", "Safety", "Fairness"]:
        task_df = summary[summary["task"] == task].copy()
        if task_df.empty:
            continue
        full_row = task_df[task_df["ablation"] == "full"]
        best_metric_idx = task_df["repair_metric_after_mean"].idxmin() if len(task_df) else None
        best_drawdown_idx = task_df["drawdown_mean"].idxmin() if len(task_df) else None
        for _, r in task_df.iterrows():
            variant = r["ablation"]
            metric = fmt_mean_std(r["repair_metric_after_mean"], r["repair_metric_after_std"] if pd.notna(r["repair_metric_after_std"]) else 0.0)
            acc = fmt_mean_std(r["acc_after_mean"], r["acc_after_std"] if pd.notna(r["acc_after_std"]) else 0.0)
            drawdown = fmt_mean_std(r["drawdown_mean"], r["drawdown_std"] if pd.notna(r["drawdown_std"]) else 0.0)
            modp = fmt_mean_std(100.0 * r["modified_params_ratio_mean"], 100.0 * (r["modified_params_ratio_std"] if pd.notna(r["modified_params_ratio_std"]) else 0.0))

            if variant == "full":
                variant = r"\textbf{Full CP-Repair}"
            if task_df.index.get_loc(r.name) == task_df.index.get_loc(best_metric_idx):
                metric = r"\textbf{" + metric + "}"
            if task_df.index.get_loc(r.name) == task_df.index.get_loc(best_drawdown_idx):
                drawdown = r"\textbf{" + drawdown + "}"

            rows.append([task, variant, metric, acc, drawdown, modp])

    table_lines = []
    table_lines.append(r"\begin{table*}[t]")
    table_lines.append(r"\centering")
    table_lines.append(r"\caption{RQ3 ablation results. Repair Metric denotes ASR for backdoor repair, VR for safety repair, and IDR for fairness repair.}")
    table_lines.append(r"\label{tab:rq3_ablation}")
    table_lines.append(r"\begin{tabular}{llcccc}")
    table_lines.append(r"\toprule")
    table_lines.append(r"Task & Variant & Repair Metric $\downarrow$ & ACC $\uparrow$ & Drawdown $\downarrow$ & Modified Params (\%) $\downarrow$ \\")
    table_lines.append(r"\midrule")
    current_task = None
    for task, variant, metric, acc, drawdown, modp in rows:
        if task != current_task:
            current_task = task
            table_lines.append(rf"\multicolumn{{6}}{{l}}{{\textbf{{{task}}}}} \\")
        table_lines.append(f"{task} & {variant} & {metric} & {acc} & {drawdown} & {modp} \\")
    table_lines.append(r"\bottomrule")
    table_lines.append(r"\end{tabular}")
    table_lines.append(r"\end{table*}")

    with open(args.table_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(table_lines))

    # Trade-off plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    tasks = ["Backdoor", "Safety", "Fairness"]
    colors = {
        "full": "black",
        "no_localization": "tab:blue",
        "no_verification": "tab:orange",
        "no_pathway_constraint": "tab:green",
    }
    markers = {"full": "*", "default": "o"}

    for ax, task in zip(axes, tasks):
        task_df = summary[summary["task"] == task]
        if task_df.empty:
            ax.set_title(task)
            continue
        for _, r in task_df.iterrows():
            ablation = r["ablation"]
            x = r["repair_metric_after_mean"]
            y = r["acc_after_mean"]
            ax.scatter(
                x,
                y,
                s=150 if ablation == "full" else 70,
                marker=markers["full"] if ablation == "full" else markers["default"],
                color=colors.get(ablation, "gray"),
                edgecolors="black" if ablation == "full" else "none",
                linewidths=1.2 if ablation == "full" else 0.0,
                zorder=3 if ablation == "full" else 2,
            )
            ax.annotate(
                ablation.replace("no_", "w/o ").replace("full", "Full"),
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
        ax.set_title(task)
        ax.set_xlabel("Repair Metric After (lower is better)")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_xlim(left=0)
    axes[0].set_ylabel("ACC After (higher is better)")
    plt.tight_layout()
    plt.savefig(args.figure_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved summary CSV to {args.summary_csv}")
    print(f"Saved LaTeX table to {args.table_tex}")
    print(f"Saved trade-off figure to {args.figure_png}")


if __name__ == "__main__":
    main()
