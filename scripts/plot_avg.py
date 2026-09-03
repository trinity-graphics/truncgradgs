#!/usr/bin/env python3

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def extract_x(run_name: str):
    nums = re.findall(r"\d+", run_name)
    return int(nums[-1]) if nums else None


def label_from_stem(stem: str):
    """Use the last dash-separated part as the x-axis label."""
    parts = stem.split("-")
    return parts[-1] if parts else stem


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def collect_metrics(data):
    metrics = defaultdict(lambda: defaultdict(list))

    for entry in data:
        run_name = entry["run_name"]
        x = extract_x(run_name)

        if x is None:
            x = 0

        values = entry["metrics"]

        if any(v is None for v in values.values()):
            continue

        for m, v in values.items():
            metrics[m][x].append(v)

    return metrics


def average_metrics(metrics_dict):
    out = {}

    for m, xdict in metrics_dict.items():
        out[m] = {}
        for x, vals in xdict.items():
            if len(vals) > 0:
                out[m][x] = float(np.mean(vals))

    return out


def compute_file_averages(avg_metrics):
    results = {}

    for m, xdict in avg_metrics.items():
        values = list(xdict.values())
        results[m] = float(np.mean(values)) if values else None

    return results


def plot_metric(metric_name, all_series, output_dir):
    plt.figure()

    all_labels = set()
    for series in all_series.values():
        all_labels.update(series.keys())

    # Try numeric sort, fall back to alphabetical
    try:
        sorted_labels = sorted(all_labels, key=int)
        is_numeric = True
    except (ValueError, TypeError):
        sorted_labels = sorted(all_labels)
        is_numeric = False

    for curve_label, series in all_series.items():
        ys = [series.get(x) for x in sorted_labels]
        # Filter out None
        valid = [(x, y) for x, y in zip(sorted_labels, ys) if y is not None]
        if valid:
            xpos = [sorted_labels.index(v[0]) for v in valid]
            plt.plot(xpos, [v[1] for v in valid], marker="o", label=curve_label)

    x_ticks = range(len(sorted_labels))
    plt.xticks(x_ticks, sorted_labels, rotation=45, ha="right")

    plt.xlabel("Configuration" if not is_numeric else "Run index")
    plt.ylabel(metric_name)
    plt.title(metric_name)
    plt.legend()
    plt.grid(True)

    safe_name = metric_name.replace("/", "_").replace(" ", "_")
    out_path = output_dir / f"{safe_name}.png"

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_files", nargs="+")
    parser.add_argument("--outdir", default="plots")
    parser.add_argument("--prefixes", nargs="+", required=True,
                        help="Prefixes to group and average files by")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    # Group files by prefix, extract label from filename stem
    prefix_points = defaultdict(list)  # prefix -> [(label, {metric: avg})]

    for path_str in args.json_files:
        if path_str.endswith("averages.json") or not path_str.endswith(".json"):
            continue
        path = Path(path_str)
        stem = path.stem

        for prefix in args.prefixes:
            if stem.startswith(prefix):
                label = label_from_stem(stem)

                data = load_json(path)
                metrics = collect_metrics(data)
                avg_metrics = average_metrics(metrics)
                file_avgs = compute_file_averages(avg_metrics)
                prefix_points[prefix].append((label, file_avgs))
                break

    # Build per-prefix, per-metric series (average across files sharing same x)
    prefix_series = {}

    for prefix, points in prefix_points.items():
        # Group by x, collect per-metric values
        by_x = defaultdict(lambda: defaultdict(list))
        for x, file_avgs in points:
            for m, v in file_avgs.items():
                if v is not None:
                    by_x[m][x].append(v)

        prefix_series[prefix] = {}
        prefix_averages = {}

        for m, xdict in by_x.items():
            series = {}
            for x, vals in xdict.items():
                series[x] = float(np.mean(vals))

            prefix_series[prefix][m] = series

            all_vals = list(series.values())
            prefix_averages[m] = float(np.mean(all_vals)) if all_vals else None

        # Print averages
        print(f"\n[{prefix}] ({len(points)} files)")
        for m, v in prefix_averages.items():
            print(f"  {m}: {v}")

        # Save per-prefix-per-metric series and averages to JSON
        series_path = outdir / f"{prefix}_series.json"
        with open(series_path, "w") as f:
            json.dump(prefix_series[prefix], f, indent=2)
        print(f"  Saved series to {series_path}")

    # Dump overall averages to JSON
    avg_out_path = outdir / "averages.json"
    overall = {}
    for prefix, points in prefix_points.items():
        by_x = defaultdict(lambda: defaultdict(list))
        for x, file_avgs in points:
            for m, v in file_avgs.items():
                if v is not None:
                    by_x[m][x].append(v)
        overall[prefix] = {}
        for m, xdict in by_x.items():
            all_vals = [v for vals in xdict.values() for v in vals]
            overall[prefix][m] = float(np.mean(all_vals)) if all_vals else None

    with open(avg_out_path, "w") as f:
        json.dump(overall, f, indent=2)
    print(f"\nSaved prefix averages to {avg_out_path}")

    # Plot per metric
    all_metrics = set()
    for ps in prefix_series.values():
        all_metrics.update(ps.keys())

    for metric in all_metrics:
        series_by_prefix = {}

        for prefix, metrics in prefix_series.items():
            if metric in metrics:
                series_by_prefix[prefix] = metrics[metric]

        if series_by_prefix:
            plot_metric(metric, series_by_prefix, outdir)


if __name__ == "__main__":
    main()
