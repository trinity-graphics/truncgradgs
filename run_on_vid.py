import argparse
import os
import shutil
import subprocess
from random import randint
from typing import Tuple
import warnings


def get_colmap_dataset_path(root: str, frame_no: int) -> str:
    dataset_path = os.path.join(root, f"colmap_frame_{frame_no:03d}")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(root, f"colmap_{frame_no:03d}")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(root, f"colmap_{frame_no}")
    return dataset_path


def get_colmap_images_path(root: str, frame_no: int) -> str:
    images_path = os.path.join(root, f"frame_{frame_no:03d}_images")
    if not os.path.exists(images_path):
        images_path = os.path.join(root, f"colmap_{frame_no:03d}", "images")
    if not os.path.exists(images_path):
        images_path = os.path.join(root, f"colmap_{frame_no}", "images")
    return images_path


def main(
    root: str,
    model_path: str,
    n_frames: int,
    n_iterations: int,
    frames_range: Tuple[int, int],
    test_iterations,
    copy_models: bool = False,
    resolution: int = 1,
    constant_set: bool = False,
    white_bg: bool = False,
    cpu: bool = False,
    rerun: bool = False,
):
    if not os.path.exists(root):
        raise FileNotFoundError(f"Root directory {root} does not exist.")
    if copy_models:
        os.makedirs(os.path.join(root, "3dgs_frames"), exist_ok=True)
    if frames_range[1] == -1:
        frames_range = (frames_range[0], n_frames)
    else:
        frames_range = (frames_range[0], frames_range[1] + 1)
    if frames_range[0] < 0 or frames_range[1] > n_frames:
        raise ValueError(
            f"Invalid frame range {frames_range}. Must be between 0 and {n_frames}."
        )
    final_model_path = None
    initial_port = randint(10000, 60000)
    for i in range(*frames_range):
        print(f"Processing frame {i + 1}/{frames_range[1]}")
        colmap_dataset_root = get_colmap_dataset_path(root, i)
        colmap_images_root = get_colmap_images_path(root, i)
        output_path = os.path.join(model_path, f"frame_{i:03d}_3dgs_model")
        if (
            os.path.exists(
                os.path.join(
                    output_path,
                    "point_cloud",
                    f"iteration_{n_iterations}",
                    "point_cloud.ply",
                )
            )
            and not rerun
        ):
            continue
        process_args = [
            "python",
            "train.py",
            f"-s={colmap_dataset_root}",
            f"-i={colmap_images_root}",
            f"-m={output_path}",
            f"--iterations={n_iterations}",
            f"--save_iterations={n_iterations}",
            "--eval",
            f"--resolution={resolution}",
            f"--port={initial_port + i}",
            f"--data_device={'cpu' if cpu else 'cuda'}",
            f"--position_lr_max_steps={n_iterations}",
            f"--densify_until_iter={round(5 * n_iterations / 6)}",
            f"--opacity_reset_interval={3000 if not constant_set else 100_000}",
        ]
        process_args.append("--test_iterations")
        process_args += list(map(str, test_iterations))
        if white_bg:
            process_args.append("--white_background")
        if constant_set:
            process_args.append("--num_extra_pts=200_000")
        if i > 0:
            if constant_set:
                assert final_model_path is not None
                process_args.append(f"--initial_model_path={final_model_path}")
                process_args += [
                    "--position_lr_init=0.0005",
                    # "--feature_lr=0.005",
                    # "--opacity_lr=0.04",
                    "--scaling_lr=0.01",
                    "--rotation_lr=0.002",
                    "--densify_until_iter=-1",  # NOTE: This disables densification and pruning altogether.
                    "--add_noise_to_initial_model",
                ]
            else:
                process_args += [
                    # "--position_lr_init=0.00009",
                    # "--feature_lr=0.002",
                    # "--opacity_lr=0.035",
                    # "--scaling_lr=0.0035",
                    # "--rotation_lr=0.0005",
                    # "--densify_from_iter=1_500",
                    # "--opacity_reset_from_iter=3_500",
                    # "--acc_stats_from_iter=1_000",
                    # "--num_extra_pts=100_000",
                    # "--add_noise_to_initial_model",
                ]
        else:
            # final_model_path = os.path.join(
            #     output_path,
            #     "point_cloud",
            #     f"iteration_{n_iterations}",
            #     "point_cloud.ply",
            # )
            # continue
            process_args += [
                "--densify_from_iter=500",
            ]
        process = subprocess.run(process_args)
        if process.returncode != 0:
            warnings.warn(f"3DGS failed to run on frame {i}. Skipping frame.", RuntimeWarning)
            continue
            # raise RuntimeError(f"3DGS failed to run on frame {i}.")
        final_model_path = os.path.join(
            output_path,
            "point_cloud",
            f"iteration_{n_iterations}",
            "point_cloud.ply",
        )
        if copy_models:
            dest_path = os.path.join(root, "3dgs_frames", f"point_cloud_{i:03d}.ply")
            shutil.copyfile(
                final_model_path,
                dest_path,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run 3DGS per-frame on a sequence of blender datasets."
    )
    parser.add_argument(
        "root", type=str, help="Root directory containing the blender datasets."
    )
    parser.add_argument(
        "model_path", type=str, help="Root directory for the output model."
    )
    parser.add_argument("--n-frames", type=int, help="Number of frames to process.")
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=30000,
        help="Number of iterations to run 3DGS for each frame.",
    )
    parser.add_argument(
        "--range",
        nargs=2,
        type=int,
        default=(0, -1),
        help="Range of frames to process. Default is all frames. This allows running multiple instances simultaneously.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run 3DGS on CPU instead of GPU. Useful for running multiple instances simultaneously.",
    )
    parser.add_argument(
        "--test-iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000]
    )
    parser.add_argument(
        "--constant-set",
        action="store_true",
        help="Use constant set of Gaussians for all frames.",
    )
    parser.add_argument(
        "--white-bg",
        action="store_true",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Whether to rerun 3DGS on frames that already have output models.",
    )
    parser.add_argument("--resolution", type=int, default=1, help="Resolution ratio.")
    args = parser.parse_args()
    main(
        args.root,
        args.model_path,
        args.n_frames,
        args.n_iterations,
        args.range,
        args.test_iterations,
        False,
        args.resolution,
        args.constant_set,
        args.white_bg,
        cpu=args.cpu,
        rerun=args.rerun,
    )
