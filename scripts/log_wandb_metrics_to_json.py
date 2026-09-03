#!/usr/bin/env python3

"""
Parse multiple local Weights & Biases runs and export selected final metrics to JSON.

Expected local structure (default wandb offline/online cache layout):

wandb/
    run-xxxx/
    run-yyyy/
    latest-run -> run-yyyy

Usage:
    python export_wandb_final_metrics.py \
        --wandb-dir ./wandb \
        --metrics eval/loss train/loss accuracy \
        --output final_metrics.json

Optional:
    --include-config learning_rate batch_size
    --strict

Behavior:
- Finds all runs under wandb/
- Reads the history for each run
- Extracts the LAST non-NaN value for requested metrics
- Saves aggregated results to JSON
"""

import json
import math
import os
import re
import shutil
import warnings
from pathlib import Path

import pandas as pd
import typer

app = typer.Typer()

# =========================
# HARD-CODE METRICS HERE
# =========================

METRICS = [
    "test/loss_viewpoint - psnr",
    "test/loss_viewpoint - ssim",
    "test/loss_viewpoint_motion - psnr",
    "test/loss_viewpoint_motion - ssim",
    "total_points",
]

CONFIG_KEYS = []


# ============================================================
# Helpers
# ============================================================


def extract_x(run_name: str):
    nums = re.findall(r"\d+", run_name)
    return int(nums[-1]) if nums else None


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# ============================================================
# Metrics command
# ============================================================


def is_valid(x):
    if x is None:
        return False
    if isinstance(x, float) and math.isnan(x):
        return False
    return True


def load_run_metadata(run_dir: Path):
    meta_path = run_dir / "files" / "wandb-metadata.json"
    if not meta_path.exists():
        return {}

    with open(meta_path, "r") as f:
        return json.load(f)


def find_run_dirs(wandb_dir: Path, name_match: str = None, run_group: str = None):
    run_dirs = []

    for p in wandb_dir.iterdir():
        if not p.is_dir():
            continue

        if not (p.name.startswith("run-") or p.name.startswith("offline-run-")):
            continue

        if name_match is None:
            run_dirs.append(p)
            continue

        meta = load_run_metadata(p)

        if "args" not in meta or not isinstance(meta["args"], list):
            warnings.warn(f"Missing or invalid metadata for run {p.name}")
            continue
        else:
            run_name = (
                meta["args"][meta["args"].index("--exp_name") + 1]
                if "--exp_name" in meta["args"]
                else ""
            )
            this_run_group = (
                meta["args"][meta["args"].index("--wandb_group") + 1]
                if "--wandb_group" in meta["args"]
                else ""
            )

        if run_name == name_match and (
            this_run_group == run_group if run_group else True
        ):
            run_dirs.append((run_name, meta["startedAt"], p))

    return [
        (name, p)
        for (name, _, p) in (
            sorted(run_dirs, key=lambda x: x[1]) if name_match else sorted(run_dirs)
        )
    ][-1:]


def load_history(run_dir: Path):
    """
    W&B stores history in different formats depending on version.
    We try a few common locations.
    """

    # Common history file locations
    candidates = [
        run_dir / "files" / "wandb-history.jsonl",
        run_dir / "files" / "media" / "table",
    ]

    history_file = None

    for c in candidates:
        if c.exists() and c.is_file():
            history_file = c
            break

    if history_file is None:
        return None

    # Standard history format
    if history_file.name == "wandb-history.jsonl":
        rows = []

        with open(history_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        if not rows:
            return None

        return pd.DataFrame(rows)

    return None


def load_summary(run_dir: Path):
    summary_path = run_dir / "files" / "wandb-summary.json"

    if summary_path.exists():
        with open(summary_path, "r") as f:
            return json.load(f)

    return {}


def load_config(run_dir: Path):
    config_path = run_dir / "files" / "config.yaml"

    if not config_path.exists():
        return {}

    try:
        import yaml

        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)

        # wandb config format:
        # key:
        #   value: ...
        config = {}

        for k, v in raw.items():
            if isinstance(v, dict) and "value" in v:
                config[k] = v["value"]
            else:
                config[k] = v

        return config

    except Exception:
        return {}


def get_last_metric(df: pd.DataFrame, metric_name: str):
    if metric_name not in df.columns:
        return None

    series = df[metric_name]

    for val in reversed(series.tolist()):
        if is_valid(val):
            return val

    return None


def extract_run_data(run_dir: Path):
    summary = load_summary(run_dir)
    config = load_config(run_dir)

    result = {
        "run_name": run_dir.name,
        "metrics": {},
        "config": {},
    }

    for metric in METRICS:
        result["metrics"][metric] = summary.get(metric, None)

    for key in CONFIG_KEYS:
        result["config"][key] = config.get(key, None)

    return result


@app.command()
def export_metrics(
    run_name: str,
    experiment_group: str,
    wandb_dir: str = "./wandb",
    output: str = "final_metrics.json",
):
    wandb_dir = Path(wandb_dir)
    run_dirs = find_run_dirs(wandb_dir, name_match=run_name, run_group=experiment_group)
    if not run_dirs:
        raise RuntimeError(
            f"No W&B runs found for run_name={run_name} and experiment_group={experiment_group} in {wandb_dir}"
        )

    all_results = []

    for run_name, run_dir in run_dirs:
        try:
            run_data = extract_run_data(run_dir=run_dir)
            run_data["run_name"] = run_name
            run_data["run_group"] = experiment_group

            all_results.append(run_data)

            print(
                f"[OK] {run_dir.name}: *{run_data['run_group']}:* {run_data['run_name']}"
            )

        except Exception as e:
            print(f"[FAILED] {run_dir.name}: {e}")

    if not os.path.isdir(os.path.dirname(output)):
        os.makedirs(os.path.dirname(output), exist_ok=True)

    with open(output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved {len(all_results)} runs to {output}")


@app.command()
def extract_images(
    run_name: str,
    category: str,
    section: str,
    wandb_dir: str = "./wandb",
    outdir: str = "extracted_images",
):
    """
    Extract latest image matching a section/tag from each run.

    Example:
        python wandb_tools.py extract-images ./wandb render
    """

    wandb_dir = Path(wandb_dir)
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    run_dirs = find_run_dirs(wandb_dir, name_match=run_name)
    if not run_dirs:
        raise RuntimeError(f"No W&B runs found in {wandb_dir}")

    # run_dirs = sorted([
    #     p for p in wandb_dir.iterdir()
    #     if p.is_dir()
    #     and (
    #         p.name.startswith("run-")
    #         or p.name.startswith("offline-run-")
    #     )
    # ])

    for run_name, run_dir in run_dirs:
        media_dir = run_dir / "files" / "media" / "images" / category

        if not media_dir.exists():
            continue

        # find matching images
        matches = sorted(media_dir.glob(f"{section}*"))

        if not matches:
            continue

        # latest image = newest modified file
        latest = max(matches, key=lambda p: p.stat().st_mtime)

        outname = f"{run_name}_{latest.name}"

        shutil.copy(latest, outdir / outname)

        print(f"[OK] {run_dir.name} -> {outname}")


if __name__ == "__main__":
    app()
