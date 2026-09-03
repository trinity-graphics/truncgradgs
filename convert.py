#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import logging
import subprocess
from argparse import ArgumentParser
import shutil

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--gpu_index", default=0, type=int,
                    help="CUDA device index for SIFT (use 0 for first GPU)")
parser.add_argument("--matcher", choices=("sequential", "exhaustive"),
                    default="sequential",
                    help="sequential: fast for ordered captures (MipNeRF360); "
                         "exhaustive: all pairs, very slow on large scenes")
parser.add_argument("--sequential_overlap", type=int, default=15,
                    help="Neighbor window for sequential_matcher")
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
colmap_bin = args.colmap_executable if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0
gpu_index = args.gpu_index


def _colmap_extraction_gpu_prefix():
    """COLMAP 4.x uses FeatureExtraction.*; COLMAP 3.x uses SiftExtraction.*."""
    try:
        result = subprocess.run(
            [colmap_bin, "feature_extractor", "-h"],
            capture_output=True,
            text=True,
            check=False,
        )
        help_text = result.stdout + result.stderr
    except OSError:
        return "SiftExtraction"
    if "FeatureExtraction.use_gpu" in help_text:
        return "FeatureExtraction"
    return "SiftExtraction"


_extraction_gpu_prefix = _colmap_extraction_gpu_prefix()


def _run_colmap(cmd):
    logging.info("Running: %s", cmd)
    return os.system(cmd)


def _extraction_gpu_flags():
    return (
        f"--{_extraction_gpu_prefix}.use_gpu {use_gpu} "
        f"--{_extraction_gpu_prefix}.gpu_index {gpu_index}"
    )


def _matching_gpu_flags():
    if use_gpu:
        return (
            f"--FeatureMatching.use_gpu {use_gpu} "
            f"--FeatureMatching.gpu_index {gpu_index} "
            "--SiftMatching.cpu_brute_force_matcher 0"
        )
    return f"--FeatureMatching.use_gpu {use_gpu}"


def _rename_points3d_to_colmap(directory):
    """COLMAP writes points3D.bin; we keep points3D_colmap.bin for MODEL_INIT=COLMAP."""
    bin_path = os.path.join(directory, "points3D.bin")
    colmap_path = os.path.join(directory, "points3D_colmap.bin")
    if not os.path.isfile(bin_path):
        return
    if os.path.isfile(colmap_path):
        os.remove(colmap_path)
    os.rename(bin_path, colmap_path)


def _rename_all_points3d_colmap(source_path):
    for sub in ("sparse", os.path.join("distorted", "sparse")):
        base = os.path.join(source_path, sub)
        if not os.path.isdir(base):
            continue
        for dirpath, _, filenames in os.walk(base):
            if "points3D.bin" in filenames:
                _rename_points3d_to_colmap(dirpath)


# MipNeRF360 / tankntemple use images/; raw COLMAP captures use input/.
if os.path.isdir(os.path.join(args.source_path, "input")):
    image_dir = "input"
elif os.path.isdir(os.path.join(args.source_path, "images")):
    image_dir = "images"
else:
    logging.error(
        f"No image folder found under {args.source_path} "
        "(expected 'input' or 'images')."
    )
    exit(1)

if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/" + image_dir + " \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " " \
        + _extraction_gpu_flags()
    exit_code = _run_colmap(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    db_path = args.source_path + "/distorted/database.db"
    if args.matcher == "sequential":
        feat_matching_cmd = colmap_command + " sequential_matcher \
            --database_path " + db_path + " \
            --SequentialMatching.overlap " + str(args.sequential_overlap) + " \
            --SequentialMatching.quadratic_overlap 1 \
            " + _matching_gpu_flags()
    else:
        feat_matching_cmd = colmap_command + " exhaustive_matcher \
            --database_path " + db_path + " \
            " + _matching_gpu_flags()
    if use_gpu:
        print(
            "Note: nvidia-smi may show ~0% GPU during matching — SIFT uses the GPU "
            "in short bursts; geometric verification (RANSAC) runs on CPU."
        )
    exit_code = _run_colmap(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path "  + args.source_path + "/" + image_dir + " \
        --output_path "  + args.source_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = _run_colmap(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + "/" + image_dir + " \
    --input_path " + args.source_path + "/distorted/sparse/0 \
    --output_path " + args.source_path + "\
    --output_type COLMAP")
exit_code = _run_colmap(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)

files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
# Copy each file from the source directory to the destination directory
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

_rename_all_points3d_colmap(args.source_path)

if(args.resize):
    print("Copying and resizing...")

    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    # Get the list of files in the source directory
    files = os.listdir(args.source_path + "/images")
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)

        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

print("Done.")
