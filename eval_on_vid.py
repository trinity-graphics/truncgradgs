import argparse
import json
import os
import subprocess
from typing import Optional, Tuple

from tqdm import tqdm


def main(
    models_root: str,
    n_frames: int,
    frames_range: Tuple[int, int],
    save_video: bool,
    iteration: int,
    rerun_metrics: bool = False,
    dynamic_masks_path: Optional[str] = None,
):
    if not os.path.exists(models_root):
        raise FileNotFoundError(f"Model directory {models_root} does not exist.")
    if frames_range[1] == -1:
        frames_range = (frames_range[0], n_frames)
    else:
        frames_range = (frames_range[0], frames_range[1] + 1)
    if frames_range[0] < 0 or frames_range[1] > n_frames:
        raise ValueError(
            f"Invalid frame range {frames_range}. Must be between 0 and {n_frames}."
        )
    per_frame_metrics = {"3DGS baseline": {}}
    per_frame_metrics_masked  = {"3DGS baseline": {}}
    for i in range(*frames_range):
        print(f"Rendering frame {i + 1}/{frames_range[1]}")
        model_path = f"{models_root}/frame_{i:03d}_3dgs_model"
        if not os.path.exists(os.path.join(model_path, "results.json")):
            process_args = [
                "python",
                "render.py",
                "--skip_train",
                "--eval",
                f"-m={model_path}",
            ]
            process = subprocess.run(process_args)
            if process.returncode != 0:
                raise RuntimeError(f"3DGS render.py failed to run on frame {i}.")
            process_args = [
                "python",
                "metrics.py",
                f"-m={model_path}",
                f"--dynamic_masks_path={dynamic_masks_path}",
            ]
            process = subprocess.run(process_args)
            if process.returncode != 0:
                raise RuntimeError(f"3DGS metrics.py failed to run on frame {i}.")
        elif rerun_metrics:
            process_args = [
                "python",
                "metrics.py",
                f"-m={model_path}",
                f"--dynamic_masks_path={dynamic_masks_path}",
            ]
            process = subprocess.run(process_args)
            if process.returncode != 0:
                raise RuntimeError(f"3DGS metrics.py failed to run on frame {i}.")
        with open(f"{model_path}/results.json", "r") as f:
            metrics = json.load(f)
        with open(f"{model_path}/results_masked.json", "r") as f:
            metrics_masked = json.load(f)
        assert len(metrics_masked.keys()) == 1, (
            "Multiple models found in results.json. What should we do?"
        )
        model_name = list(metrics.keys())[0]
        for metric_name, val in metrics[model_name].items():
            if metric_name not in per_frame_metrics["3DGS baseline"]:
                per_frame_metrics["3DGS baseline"][metric_name] = {}
            per_frame_metrics["3DGS baseline"][metric_name][str(i)] = val
        for metric_name, val in metrics_masked[model_name].items():
            if metric_name not in per_frame_metrics_masked["3DGS baseline"]:
                per_frame_metrics_masked["3DGS baseline"][metric_name] = {}
            per_frame_metrics_masked["3DGS baseline"][metric_name][str(i)] = val
    # Save per-frame metrics to a JSON file
    with open(f"{models_root}/per_frame.json", "w") as f:
        print(f"Saving to {models_root}/per_frame.json")
        json.dump(per_frame_metrics, f, indent=2)
    with open(f"{models_root}/per_frame_masked.json", "w") as f:
        print(f"Saving to {models_root}/per_frame_masked.json")
        json.dump(per_frame_metrics_masked, f, indent=2)
    # Print average metrics
    metric_names = per_frame_metrics["3DGS baseline"].keys()
    avg_metrics = {"3DGS baseline": {}}
    avg_metrics_masked = {"3DGS baseline": {}}
    print("========= Average metrics ==========")
    for name in metric_names:
        avg = sum(val for val in per_frame_metrics["3DGS baseline"][name].values())
        avg /= len(per_frame_metrics["3DGS baseline"][name].keys())
        avg_metrics["3DGS baseline"][name] = avg
        print(f"\t {name}: {avg:.4f}")
        avg = sum(val for val in per_frame_metrics_masked["3DGS baseline"][name].values())
        avg /= len(per_frame_metrics_masked["3DGS baseline"][name].keys())
        avg_metrics_masked["3DGS baseline"][name] = avg
        print(f"\t {name} (masked): {avg:.4f}")
    print("===================================")
    with open(f"{models_root}/results.json", "w") as f:
        print(f"Saving to {models_root}/results.json")
        json.dump(avg_metrics, f, indent=2)
    with open(f"{models_root}/results_masked.json", "w") as f:
        print(f"Saving to {models_root}/results_masked.json")
        json.dump(avg_metrics_masked, f, indent=2)
    if save_video:
        n_cams = None
        os.makedirs(f"{models_root}/videos", exist_ok=True)
        for i in tqdm(range(*frames_range), desc="Saving video frames"):
            path = (
                f"{models_root}/frame_{i:03d}_3dgs_model/test/ours_{iteration}/renders"
            )
            if n_cams is None:
                n_cams = len(os.listdir(path))
                for j in range(n_cams):
                    os.makedirs(f"{models_root}/videos/cam_{j}", exist_ok=True)
            for j in range(n_cams):
                try:
                    os.link(
                        os.path.join(path, f"{j:05d}.png"),
                        os.path.join(
                            models_root, "videos", f"cam_{j}", f"frame_{i:03d}.png"
                        ),
                    )
                except FileExistsError:
                    pass
        assert n_cams is not None
        print("Generating videos...")
        for j in range(n_cams):
            os.system(
                f"ffmpeg -y -framerate 30 -i {models_root}/videos/cam_{j}/frame_%03d.png -c:v libx264 -pix_fmt yuv420p {models_root}/videos/cam_{j}_output.mp4"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run 3DGS per-frame on a sequence of blender datasets."
    )
    parser.add_argument(
        "root", type=str, help="Root directory containing the per-frame models."
    )
    parser.add_argument("--n-frames", type=int, help="Number of frames to process.")
    parser.add_argument(
        "--range",
        nargs=2,
        type=int,
        default=(0, -1),
        help="Range of frames to process. Default is all frames. This allows running multiple instances simultaneously.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save the output video per test camera.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Iteration to save the output video.",
    )
    parser.add_argument(
        "--rerun-metrics",
        action="store_true",
    )
    parser.add_argument(
        "--dynamic-masks-path",
        required=False,
        default=None,
        help="Path to dynamic masks for metrics computation.",
    )
    args = parser.parse_args()
    if args.save_video and args.iteration is None:
        raise ValueError("Must specify --iteration when using --save-video.")
    main(
        args.root,
        args.n_frames,
        args.range,
        args.save_video,
        args.iteration,
        args.rerun_metrics,
        args.dynamic_masks_path,
    )
