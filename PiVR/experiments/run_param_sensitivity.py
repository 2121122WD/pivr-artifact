"""Unified localization-stage parameter stability runner for PiVR.

This script runs one-parameter-at-a-time or small-grid sweeps over localization
parameters and records comparable metrics.  It is intended for RQ4, where the
question is parameter stability rather than global hyperparameter tuning.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    script_dir = os.path.dirname(__file__)
    for candidate in [os.path.abspath(path), os.path.abspath(os.path.join(script_dir, path))]:
        if os.path.exists(candidate):
            return candidate
    return os.path.abspath(path)


def load_grid(path: str) -> Dict[str, List[object]]:
    path = resolve_path(path)
    with open(path, "r", encoding="utf-8") as f:
        grid = json.load(f)
    if not isinstance(grid, dict):
        raise ValueError("Grid JSON must be an object mapping parameter names to lists")
    normalized: Dict[str, List[object]] = {}
    for key, values in grid.items():
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(f"Grid entry {key!r} must be a non-empty list")
        normalized[key] = values
    return normalized


def cartesian_product(grid: Dict[str, List[object]]):
    keys = list(grid.keys())
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo))


TASK_ALLOWED_OVERRIDE_KEYS = {
    "safety": {"k"},
    "backdoor": {"k"},
    "fairness": {"k"},
}

TASK_KEY_PARAMS = {
    "safety": ["k"],
    "backdoor": ["k"],
    "fairness": ["k"],
}

TASK_METRIC_FIELDS = {
    "safety": ["repair_metric_name", "metric_before", "metric_after", "acc_before", "acc_after", "drawdown", "loc_time", "repair_time", "total_time"],
    "backdoor": ["repair_metric_name", "metric_before", "metric_after", "acc_before", "acc_after", "drawdown", "loc_time", "repair_time", "total_time"],
    "fairness": ["repair_metric_name", "metric_before", "metric_after", "acc_before", "acc_after", "drawdown", "loc_time", "repair_time", "total_time"],
}


def build_cmd(script: str, base_args: List[str], overrides: Dict[str, object], task: str) -> List[str]:
    cmd = [sys.executable, script]
    cmd.extend(base_args)
    allowed = TASK_ALLOWED_OVERRIDE_KEYS.get(task, set())
    for key, value in overrides.items():
        if value is None or key not in allowed:
            continue
        cmd.extend([f"--{key}", str(value)])
    return cmd


def _maybe_percent(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return value * 100.0 if abs(value) <= 1.5 else value


def extract_metrics_from_stdout(task: str, stdout: str) -> Dict[str, Optional[float]]:
    """Parse common summary lines from the target scripts and normalize to %.

    The experiment entrypoints do not print exactly the same format.  This parser
    keeps all tasks in the same output schema: metric_after is ASR/VR/IDR, acc is
    the preservation accuracy, and drawdown is acc_before - acc_after.
    """
    out = {
        "repair_metric_name": {"backdoor": "ASR", "safety": "VR", "fairness": "IDR"}[task],
        "metric_before": None,
        "metric_after": None,
        "acc_before": None,
        "acc_after": None,
        "drawdown": None,
        "loc_time": None,
        "repair_time": None,
        "total_time": None,
    }

    if task == "backdoor":
        acc = re.search(r"ACC:\s*([0-9.]+)%\s*->\s*([0-9.]+)%", stdout)
        sr = re.search(r"(?:SR|ASR):\s*([0-9.]+)%\s*->\s*([0-9.]+)%", stdout)
        if acc:
            out["acc_before"] = float(acc.group(1))
            out["acc_after"] = float(acc.group(2))
        if sr:
            out["metric_before"] = float(sr.group(1))
            out["metric_after"] = float(sr.group(2))
    elif task == "safety":
        vr = re.search(r"Counter VR\s*:\s*before=([0-9.]+)\s*after=([0-9.]+)", stdout)
        acc = re.search(r"Positive ACC\s*:\s*before=([0-9.]+)\s*after=([0-9.]+)", stdout)
        if vr:
            out["metric_before"] = _maybe_percent(float(vr.group(1)))
            out["metric_after"] = _maybe_percent(float(vr.group(2)))
        if acc:
            out["acc_before"] = _maybe_percent(float(acc.group(1)))
            out["acc_after"] = _maybe_percent(float(acc.group(2)))
    elif task == "fairness":
        acc = re.search(r"ACC:\s*([0-9.]+)%\s*->\s*([0-9.]+)%", stdout)
        idr = re.search(r"IDR:\s*([0-9.]+)%\s*->\s*([0-9.]+)%", stdout)
        if acc:
            out["acc_before"] = float(acc.group(1))
            out["acc_after"] = float(acc.group(2))
        if idr:
            out["metric_before"] = float(idr.group(1))
            out["metric_after"] = float(idr.group(2))

    time_line = re.search(r"Time:\s*loc=([0-9.]+)s,\s*repair=([0-9.]+)s,\s*total=([0-9.]+)s", stdout)
    if time_line:
        out["loc_time"] = float(time_line.group(1))
        out["repair_time"] = float(time_line.group(2))
        out["total_time"] = float(time_line.group(3))

    if out["acc_before"] is not None and out["acc_after"] is not None:
        out["drawdown"] = out["acc_before"] - out["acc_after"]
    return out


def write_csv_rows(path: str, fieldnames: List[str], rows: List[Dict[str, object]]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _lower_is_better(metric: Optional[float], acc: Optional[float]) -> tuple:
    return (
        float("inf") if metric is None else metric,
        float("inf") if acc is None else -acc,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified PiVR localization-stage parameter stability runner")
    parser.add_argument("--task", choices=["safety", "backdoor", "fairness"], required=True)
    parser.add_argument("--script", required=True, help="Target experiment entrypoint script")
    parser.add_argument("--grid", required=False, help="JSON parameter grid file")
    parser.add_argument("--base-args", default="", help="Extra args passed to the target script")
    parser.add_argument("--output", default="", help="CSV output path")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--single-run", action="store_true", help="Run once without sweeping")
    parser.add_argument("--top-k", type=int, default=5, help="Print top-k ranked runs after sweep")
    args = parser.parse_args()

    script = resolve_path(args.script)
    base_args = args.base_args.split() if args.base_args else []

    if args.single_run:
        cmd = build_cmd(script, base_args, {}, args.task)
        print("Executing single run:\n" + " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
        return

    if not args.grid:
        raise ValueError("--grid is required unless --single-run is provided")

    grid = load_grid(args.grid)
    combinations = list(cartesian_product(grid))
    output = args.output or os.path.join(os.getcwd(), f"rq4_localization_stability_{args.task}.csv")

    rows = []
    for idx, overrides in enumerate(combinations, start=1):
        cmd = build_cmd(script, base_args, overrides, args.task)
        print(f"\n[{idx}/{len(combinations)}] overrides={overrides}")
        print("CMD:", " ".join(cmd))
        if args.dry_run:
            continue
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        print(stdout)
        if proc.returncode != 0:
            print(stderr, file=sys.stderr)

        metrics = extract_metrics_from_stdout(args.task, stdout)
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task": args.task,
            "script": os.path.basename(script),
            "returncode": proc.returncode,
            "stdout_tail": stdout.splitlines()[-1] if stdout else "",
            "stderr_tail": stderr.splitlines()[-1] if stderr else "",
        }
        for key in TASK_KEY_PARAMS[args.task]:
            row[key] = overrides.get(key, "")
        for key, value in metrics.items():
            row[key] = "" if value is None else (value if isinstance(value, str) else f"{value:.6f}")
        rows.append(row)

    if args.dry_run:
        return

    fieldnames = ["timestamp", "task", "script", "returncode", "stdout_tail", "stderr_tail"] + TASK_KEY_PARAMS[args.task] + TASK_METRIC_FIELDS[args.task]
    write_csv_rows(output, fieldnames, rows)

    ranked = sorted(rows, key=lambda r: _lower_is_better(
        float(r["metric_after"]) if r.get("metric_after") not in (None, "") else None,
        float(r["acc_after"]) if r.get("acc_after") not in (None, "") else None,
    ))
    print(f"\nSaved sweep summary to {output}")
    print("Top ranked runs, lower repair metric and higher ACC are better:")
    for i, row in enumerate(ranked[: max(1, args.top_k)], start=1):
        params = ", ".join(f"{k}={row.get(k)}" for k in TASK_KEY_PARAMS[args.task] if row.get(k) not in (None, ""))
        print(f"  #{i}: metric={row.get('metric_after')}, acc={row.get('acc_after')}, {params}")


if __name__ == "__main__":
    main()
