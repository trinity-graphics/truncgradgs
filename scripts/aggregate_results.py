#!/usr/bin/env python3

import json
from collections import defaultdict
from pathlib import Path

import typer

app = typer.Typer()


def read_gaussians_count(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except Exception:
        return None


def read_results_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


@app.command()
def main(root: str = typer.Argument(..., help="Root directory containing scene/method/experiment structure")):
    root = Path(root)
    scenes = sorted(p for p in root.iterdir() if p.is_dir())

    # Structure: method -> experiment -> list of (scene, metrics dict, gaussians count)
    data: dict[str, dict[str, list[tuple[str, dict, int | None]]]] = \
        defaultdict(lambda: defaultdict(list))

    for scene_dir in scenes:
        scene = scene_dir.name
        for method_dir in sorted(scene_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            method = method_dir.name
            for exp_dir in sorted(method_dir.iterdir()):
                if not exp_dir.is_dir():
                    continue
                exp = exp_dir.name

                results = read_results_json(exp_dir / "results.json")
                gcount = read_gaussians_count(exp_dir / "gaussians_count.txt")

                data[method][exp].append((scene, results or {}, gcount))

    print("=" * 72)
    print(f"Aggregated results from {len(scenes)} scenes")
    print("=" * 72)

    for method in sorted(data):
        print(f"\n{'─' * 72}")
        print(f"  Method: {method}")
        print(f"{'─' * 72}")

        for exp in sorted(data[method]):
            entries = data[method][exp]
            n = len(entries)
            scenes_list = [e[0] for e in entries]

            print(f"\n  ┌─ Experiment: {exp}  ({n} scenes: {', '.join(scenes_list)})")

            # ---- Aggregate gaussians_count ----
            gcounts = [e[2] for e in entries if e[2] is not None]
            if gcounts:
                avg_g = sum(gcounts) / len(gcounts)
                print(f"  │  gaussians_count: avg={avg_g:.1f}  min={min(gcounts)}  max={max(gcounts)}")

            # ---- Aggregate results.json metrics ----
            all_metrics: dict[str, list[float]] = defaultdict(list)
            for _scene, res, _g in entries:
                if not res:
                    continue
                for key, val in res["ours_30000"].items():
                    if isinstance(val, (int, float)):
                        all_metrics[key].append(val)

            if all_metrics:
                print(f"  │  Metrics (averaged across {n} scenes):")
                for key in sorted(all_metrics):
                    vals = all_metrics[key]
                    avg = sum(vals) / len(vals)
                    print(f"  │    {key}: {avg:.6f}  (min={min(vals):.6f}, max={max(vals):.6f})")
            else:
                print(f"  │  (no numeric metrics in results.json)")

            print(f"  └─")


if __name__ == "__main__":
    app()
