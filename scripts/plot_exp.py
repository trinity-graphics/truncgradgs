#!/usr/bin/env python3

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def extract_x(run_name: str):
    """
    Extract last integer in run name, e.g.
    baseline-50-to-47 -> 47
    """
    nums = re.findall(r"\d+", run_name)
    return int(nums[-1]) if nums else None


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def collect_metrics(data):
    """
    Strict version:
    - If ANY metric in a run entry is None -> discard entire run
    - Otherwise include all metrics
    """

    metrics = defaultdict(lambda: defaultdict(list))

    for entry in data:
        run_name = entry["run_name"]
        x = extract_x(run_name)

        if x is None:
            continue

        values = entry["metrics"]

        # --- strict null check ---
        if any(v is None for v in values.values()):
            continue

        for m, v in values.items():
            metrics[m][x].append(v)

    return metrics

def average_metrics(metrics_dict):
    """
    Convert list values into means:
      metrics[m][x] -> scalar
    """
    out = {}

    for m, xdict in metrics_dict.items():
        out[m] = {}
        for x, vals in xdict.items():
            if len(vals) > 0:
                out[m][x] = float(np.mean(vals))

    return out


def compute_file_averages(avg_metrics):
    """
    Average over ALL x for each metric
    """
    results = {}

    for m, xdict in avg_metrics.items():
        values = list(xdict.values())
        results[m] = float(np.mean(values)) if values else None

    return results


def plot_metric(metric_name, all_series, output_dir):
    plt.figure()

    for label, series in all_series.items():
        xs = sorted(series.keys())
        ys = [series[x] for x in xs]

        plt.plot(xs, ys, marker="o", label=label)

    plt.xlabel("Run index (last number in run name)")
    plt.ylabel(metric_name)
    plt.title(metric_name)
    plt.legend()
    plt.grid(True)

    safe_name = metric_name.replace("/", "_").replace(" ", "_")
    out_path = output_dir / f"{safe_name}.png"

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_files", nargs="+")
    parser.add_argument("--outdir", default="plots")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    all_file_metrics = {}
    file_averages = {}

    # -------- load all files --------
    for path_str in args.json_files:
        if path_str.endswith("averages.json") or not path_str.endswith(".json"):
            continue
        path = Path(path_str)
        data = load_json(path)

        metrics = collect_metrics(data)
        avg_metrics = average_metrics(metrics)

        all_file_metrics[path.stem] = avg_metrics
        file_averages[path.stem] = compute_file_averages(avg_metrics)

    # -------- print averages --------
    print("\n=== AVERAGES PER FILE ===")
    for file, metrics in file_averages.items():
        print(f"\n[{file}]")
        for m, v in metrics.items():
            print(f"  {m}: {v}")

    # Dump averages to JSON:
    avg_out_path = outdir / "averages.json"
    with open(avg_out_path, "w") as f:
        json.dump(file_averages, f, indent=2)
    print(f"\nSaved file averages to {avg_out_path}")

    # -------- plot per metric --------
    all_metrics = set()
    for fm in all_file_metrics.values():
        all_metrics.update(fm.keys())

    for metric in all_metrics:
        series_by_file = {}

        for file, metrics in all_file_metrics.items():
            if metric in metrics:
                series_by_file[file] = metrics[metric]

        if series_by_file:
            plot_metric(metric, series_by_file, outdir)


if __name__ == "__main__":
    main()
