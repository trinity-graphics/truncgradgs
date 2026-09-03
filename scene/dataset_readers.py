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

import json
import os
import sys
from pathlib import Path
from typing import NamedTuple, Optional, Union

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_points3D_binary,
)
from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2
from utils.sh_utils import SH2RGB


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool
    motion_mask: Optional[Union[str, np.ndarray]]


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    is_nerf_synthetic: bool
    ply_path: Optional[str]


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(
    cam_extrinsics,
    cam_intrinsics,
    depths_params,
    images_folder,
    depths_folder,
    motion_masks_path,
    test_cam_names_list,
    no_motion_mask=False,
):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, (
                "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
            )

        n_remove = len(extr.name.split(".")[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = (
            os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png")
            if depths_folder != ""
            else ""
        )
        if no_motion_mask:
            print(f"Using no motion mask for camera {extr.name} (uid {uid})")
            motion_mask = np.ones((height, width), dtype=np.bool_)
        else:
            print(
                f"Looking for motion mask for camera {extr.name} (uid {uid}) in '{motion_masks_path}'..."
            )
            motion_masks_dir = os.path.join(motion_masks_path, extr.name)
            if not os.path.isdir(motion_masks_dir):
                raise FileNotFoundError(
                    f"Motion masks directory not found for camera {extr.name} at path '{motion_masks_dir}'."
                    + " If this is intentional, pass the no_motion_mask=True flag."
                )

            frame_idx = int(os.path.split(images_folder)[0].split("colmap_")[1])
            mask_path = os.path.join(
                motion_masks_dir, sorted(os.listdir(motion_masks_dir))[frame_idx]
            )
            # print(f"\nLoading motion mask for camera {extr.name} with frame idx '{frame_idx}' from '{mask_path}'")
            if not mask_path.endswith(".npz"):
                raise ValueError(
                    f"Motion mask file for camera {extr.name} does not have .npz extension: '{mask_path}'"
                )
            motion_mask = np.load(mask_path)["arr_0"]
            # motion_mask = cv2.resize(motion_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            motion_mask = (motion_mask > 0).astype(np.bool_).squeeze()
            assert motion_mask.shape == (height, width), (
                f"Motion mask shape {motion_mask.shape} does not match expected image dimensions {(height, width)} for camera {extr.name}"
            )

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            depth_params=depth_params,
            image_path=image_path,
            image_name=image_name,
            depth_path=depth_path,
            width=width,
            height=height,
            is_test=image_name in test_cam_names_list,
            motion_mask=motion_mask,
        )
        cam_infos.append(cam_info)

    sys.stdout.write("\n")
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(
    path, images, depths, eval, train_test_exp, no_motion_mask, llffhold=8
):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array(
                [depths_params[key]["scale"] for key in depths_params]
            )
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(
                f"Error: depth_params.json file not found at path '{depth_params_file}'."
            )
            sys.exit(1)
        except Exception as e:
            print(
                f"An unexpected error occurred when trying to open depth_params.json file: {e}"
            )
            sys.exit(1)

    if eval:
        has_test_file = os.path.isfile(os.path.join(path, "sparse/0", "test.txt"))
        if "360" in path:
            llffhold = 8
        if llffhold and not has_test_file:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [
                name for idx, name in enumerate(cam_names) if idx % llffhold == 0
            ]
        elif has_test_file:
            with open(os.path.join(path, "sparse/0", "test.txt"), "r") as file:
                test_cam_names_list = [line.strip() for line in file]
                print(f"Using test cameras: {','.join(test_cam_names_list)}")
        else:
            raise RuntimeError("no test cameras found.")
    else:
        test_cam_names_list = []

    reading_dir = "images" if images is None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir),
        depths_folder=os.path.join(path, depths) if depths != "" else "",
        motion_masks_path=os.path.join(path, "..", "motion_masks"),
        test_cam_names_list=test_cam_names_list,
        no_motion_mask=no_motion_mask,
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    MODEL_INIT = os.getenv("MODEL_INIT", "TRUE").upper()
    is_random, ply_name, bin_name = False, None, None
    if MODEL_INIT == "TRUE":
        ply_name = "points3D.ply"
        bin_name = "points3D.bin"
    elif MODEL_INIT == "COLMAP":
        ply_name = "points3D_colmap.ply"
        bin_name = "points3D_colmap.bin"
    elif MODEL_INIT == "COLMAP_DENSE":
        ply_name = "points3D_colmap_dense.ply"
        bin_name = "points3D_colmap_dense.bin"
    elif MODEL_INIT == "RANDOM_OURS":
        ply_name = "points3D.ply"
        bin_name = "points3D.bin"
        is_random = True
    elif MODEL_INIT in ["RANDOM", "RANDOM_MCMC"]:
        ply_name = None
        bin_name = None
        is_random = True
        ply_path = None
    else:
        raise ValueError(
            "MODEL_INIT env var must be one of: true, colmap, colmap_dense, random, random_ours, random_mcmc"
        )

    if ply_name is not None and bin_name is not None:
        ply_path = os.path.join(path, "sparse/0/", ply_name)
        bin_path = os.path.join(path, "sparse/0/", bin_name)
        if not os.path.exists(ply_path):
            print(
                "Converting point3d.bin to .ply, will happen only the first time you open the scene."
            )
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                raise Exception("point cloud bin file not found")
            # if xyz.shape[0] > 100_000:
            #     print(
            #         f"Downsampling point cloud from {xyz.shape[0]} points to 100,000 points for faster loading..."
            #     )
            #     indices = np.random.permutation(xyz.shape[0])[:100_000]
            #     xyz = xyz[indices]
            #     rgb = rgb[indices]
            storePly(ply_path, xyz, rgb)

        pcd = fetchPly(ply_path)
        # if pcd.points.shape[0] > 100_000:
        #     print(
        #         f"Downsampling point cloud from {pcd.points.shape[0]} points to 100,000 points for faster loading..."
        #     )
        #     indices = np.random.permutation(pcd.points.shape[0])[:100_000]
        #     xyz = pcd.points[indices]
        #     rgb = pcd.colors[indices]
        #     normals = pcd.normals[indices]
        #     pcd = BasicPointCloud(points=xyz, colors=rgb, normals=normals)
    if is_random:
        # Since this data set has no colmap data, we start with random points
        num_pts = 150_000
        print(f"Generating random point cloud ({num_pts})...")

        if MODEL_INIT == "RANDOM":
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
            rgb = SH2RGB(shs)
        elif MODEL_INIT == "RANDOM_MCMC":
            # https://github.com/ubc-vision/3dgs-mcmc/blob/7b4fc9f76a1c7b775f69603cb96e70f80c7e6d13/scene/dataset_readers.py
            xyz = np.random.random((num_pts, 3)) * nerf_normalization[
                "radius"
            ] * 3 * 2 - (nerf_normalization["radius"] * 3)
            shs = np.random.random((num_pts, 3)) / 255.0
            rgb = SH2RGB(shs)
        elif MODEL_INIT == "RANDOM_OURS":
            scene_bounds = pcd.points.max(axis=0) - pcd.points.min(axis=0)
            print(f"Scene bounds: {scene_bounds}")
            unit_cube_pts = np.random.uniform(-1, 1, (num_pts, 3))
            xyz = unit_cube_pts * scene_bounds / 2 + pcd.points.mean(axis=0)
            shs = np.random.random((num_pts, 3)) / 255.0
            rgb = SH2RGB(shs)
        elif MODEL_INIT == "RANDOM_UNIT_SPHERE":
            scene_bounds = pcd.points.max(axis=0) - pcd.points.min(axis=0)
            print(f"Scene bounds: {scene_bounds}")
            unit_cube_pts = np.random.uniform(-1, 1, (num_pts, 3))
            unit_cube_pts = unit_cube_pts / np.linalg.norm(
                unit_cube_pts, axis=1, keepdims=True
            )
            # unit_ball_norms = np.linalg.norm(unit_ball_points, axis=1, keepdims=False)
            # unit_ball_points[unit_ball_norms > 1] = (
            #     unit_ball_points[unit_ball_norms > 1] / unit_ball_norms[unit_ball_norms >
            #         1][:, None]
            # )
            xyz = unit_cube_pts * scene_bounds / 4 + pcd.points.mean(axis=0)
            rgb = np.random.random((num_pts, 3))  # *  255.0
        pcd = BasicPointCloud(points=xyz, colors=rgb, normals=np.zeros((num_pts, 3)))

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        is_nerf_synthetic=False,
    )
    return scene_info


# def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
#     cam_infos = []
#
#     with open(os.path.join(path, transformsfile)) as json_file:
#         contents = json.load(json_file)
#         fovx = contents["camera_angle_x"]
#
#         frames = contents["frames"]
#         for idx, frame in enumerate(frames):
#             cam_name = os.path.join(path, frame["file_path"] + extension)
#
#             # NeRF 'transform_matrix' is a camera-to-world transform
#             c2w = np.array(frame["transform_matrix"])
#             # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
#             c2w[:3, 1:3] *= -1
#
#             # get the world-to-camera transform and set R, T
#             w2c = np.linalg.inv(c2w)
#             R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
#             T = w2c[:3, 3]
#
#             image_path = os.path.join(path, cam_name)
#             image_name = Path(cam_name).stem
#             image = Image.open(image_path)
#
#             im_data = np.array(image.convert("RGBA"))
#
#             bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
#
#             norm_data = im_data / 255.0
#             arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
#             image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
#
#             fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
#             FovY = fovy
#             FovX = fovx
#
#             depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""
#
#             cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
#                             image_path=image_path, image_name=image_name,
#                             width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
#
#     return cam_infos


def readCamerasFromTransforms(
    path, transformsfile, depth_folder, white_background, is_test, extension=".png"
):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (
                1 - norm_data[:, :, 3:4]
            )
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovx = contents["camera_angle_x"] if "camera_angle_x" in contents else None
            w, h = contents["w"], contents["h"]
            assert w == image.size[0] and h == image.size[1]
            if fovx is not None:
                fovy = focal2fov(fov2focal(fovx, w), h)
                FovY = fovy
                FovX = fovx
            else:
                FovY = focal2fov(contents["fl_y"], h)
                FovX = focal2fov(contents["fl_x"], w)

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image_path=image_path,
                    image_name=image_name,
                    width=w,
                    height=h,
                    depth_path="",
                    depth_params=None,
                    is_test=is_test,
                )
            )

    return cam_infos


def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):
    depths_folder = os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", depths_folder, white_background, False, extension
    )
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", depths_folder, white_background, True, extension
    )

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(
            points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
        )

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        is_nerf_synthetic=True,
    )
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
}
