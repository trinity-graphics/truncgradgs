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
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
from tqdm import tqdm

from lpipsPyTorch.modules.lpips import LPIPS
from utils.loss_utils import  ssim as og_ssim

try:
    from fused_ssim import fused_ssim

    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False



def psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def ssim(pred_img, gt_img, train=True):
    if FUSED_SSIM_AVAILABLE:
        pred_img = pred_img[None, ...] if len(pred_img.shape) < 4 else pred_img
        gt_img = gt_img[None, ...] if len(gt_img.shape) < 4 else gt_img
        return fused_ssim(pred_img, gt_img, train=train)
    else:
        return og_ssim(pred_img, gt_img).item()


ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)


def msssim(rgb, gts):
    # assert (rgb.max() <= 1.05 and rgb.min() >= -0.05)
    # assert (gts.max() <= 1.05 and gts.min() >= -0.05)
    return ms_ssim(rgb, gts).item()


def readImages(renders_dir, gt_dir, slice_):
    renders = []
    gts = []
    image_names = []
    for fname in sorted(os.listdir(renders_dir))[slice_]:
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def load_test_cam_motion_masks(dynamic_masks_path: str, test_cam_list: List[str]):
    print(f"Loading motion masks from {dynamic_masks_path}")
    masks = []
    for cam in tqdm(
        sorted(test_cam_list), desc="Loading motion masks for test cameras"
    ):
        masks_path = os.path.join(dynamic_masks_path, cam.strip())
        if not os.path.exists(masks_path):
            raise FileNotFoundError(
                f"Motion masks for camera {cam.strip()} not found in {dynamic_masks_path}"
            )
        masks.extend(
            [
                (
                    torch.from_numpy(np.load(os.path.join(masks_path, f))["arr_0"])
                    / 255.0
                ).float()
                for f in sorted(os.listdir(masks_path))
                if f.endswith(".npz")
            ]
        )
    return masks


def evaluate(
    model_paths: str,
    n_frames_per_camera: int,
    is_oracle: bool,
    dynamic_masks_path: Optional[str],
    frame_id: Optional[int],
):
    full_dict = {}
    full_dict_masked = {}
    per_frame_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    lpips_model_alex = LPIPS(net_type="alex").to("cuda")
    lpips_model_vgg = LPIPS(net_type="vgg").to("cuda")

    test_masks = None
    if len(model_paths) > 1:
        raise NotImplementedError(
            "Multiple model paths provided. This is not currently supported."
        )
    with open(os.path.join(model_paths[0], "cfg_args"), "r") as f:
        # Namespace is called in eval()
        Namespace = lambda **kwargs: {k: v for k, v in kwargs.items()}
        config = eval(f.read())
        data_path = config["source_path"]
        if dynamic_masks_path is not None and os.path.exists(dynamic_masks_path):
            colmap_path = (
                os.path.join(
                    os.path.split(data_path)[0],
                    os.path.basename(data_path).split("_")[0] + "_0",
                )
                if os.path.basename(data_path).startswith("colmap_")
                else os.path.join(data_path, "colmap_0")
            )
            with open(os.path.join(colmap_path, "sparse", "0", "test.txt"), "r") as f:
                test_cam_list = f.readlines()
            test_masks = load_test_cam_motion_masks(dynamic_masks_path, test_cam_list)
        else:
            print(
                f"dynamic_masks_path={dynamic_masks_path}. Exists? {os.path.exists(dynamic_masks_path) if dynamic_masks_path is not None else False}"
            )

    for scene_dir in model_paths:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        full_dict_masked[scene_dir] = {}
        per_frame_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}

        test_dir = Path(scene_dir) / "test"

        for method in os.listdir(test_dir):
            print("Method:", method)

            full_dict[scene_dir][method] = {}
            full_dict_masked[scene_dir][method] = {}
            per_frame_dict[scene_dir][method] = {}
            full_dict_polytopeonly[scene_dir][method] = {}
            per_view_dict_polytopeonly[scene_dir][method] = {}

            method_dir = test_dir / method
            gt_dir = method_dir / "gt"
            renders_dir = method_dir / "renders"

            ssims, masked_ssims = [], []
            msssims, masked_msssims = [], []
            psnrs, masked_psnrs = [], []
            lpipss_vgg, masked_lpipss_vgg = [], []
            lpipss_alex, masked_lpipss_alex = [], []
            step = 200
            n_images = len(os.listdir(renders_dir))
            for start_idx in range(0, n_images, step):
                renders, gts, _ = readImages(
                    renders_dir, gt_dir, slice_=slice(start_idx, start_idx + step)
                )

                for idx in tqdm(
                    range(len(renders)),
                    desc=f"Metric evaluation progress (chunk {start_idx // step})",
                ):
                    ssims.append(ssim(renders[idx], gts[idx], train=False))
                    msssims.append(msssim(renders[idx].cpu(), gts[idx].cpu()))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss_vgg.append(lpips_model_vgg(renders[idx], gts[idx]))
                    lpipss_alex.append(lpips_model_alex(renders[idx], gts[idx]))
                    # TODO: VMAF metric using our Y4M conversion script and the VMAF
                    # utility
                    if (
                        test_masks is not None and idx > 0
                    ):  # Skip frame 0 as there's no motion
                        if frame_id is None:
                            raise ValueError(
                                "frame_id must be provided if test_masks is provided."
                            )
                        mask = test_masks[frame_id].to(renders[idx].device)
                        mask = mask.transpose(2, 0).transpose(1, 2)[None]
                        if not mask.max().item() > 0:
                            raise ValueError(
                                f"Mask for frame {idx} is empty. This should not happen."
                            )
                        masked_gt = mask * gts[idx]
                        masked_render = mask * renders[idx]
                        masked_ssims.append(ssim(masked_render, masked_gt, train=False))
                        masked_msssims.append(
                            msssim(
                                masked_render.cpu(),
                                masked_gt.cpu(),
                            )
                        )
                        masked_psnrs.append(psnr(masked_render, masked_gt))
                        masked_lpipss_vgg.append(
                            lpips_model_vgg(masked_render, masked_gt)
                        )
                        masked_lpipss_alex.append(
                            lpips_model_alex(masked_render, masked_gt)
                        )

            print(
                "SSIM : {:>12.7f} (std={:>12.7f})".format(
                    torch.tensor(ssims).mean(),
                    torch.tensor(ssims).std(),
                )
            )
            print(
                "MSSSIM : {:>12.7f} (std={:>12.7f})".format(
                    torch.tensor(msssims).mean(),
                    torch.tensor(msssims).std(),
                )
            )
            print(
                "PSNR : {:>12.7f} (std={:>12.7f})".format(
                    torch.tensor(psnrs).mean(),
                    torch.tensor(psnrs).std(),
                )
            )
            print(
                "LPIPS (VGG): {:>12.7f} (std={:>12.7f})".format(
                    torch.tensor(lpipss_vgg).mean(),
                    torch.tensor(lpipss_vgg).std(),
                )
            )
            print(
                "LPIPS (ALEX): {:>12.7f} (std={:>12.7f})".format(
                    torch.tensor(lpipss_alex).mean(),
                    torch.tensor(lpipss_alex).std(),
                )
            )
            print("")
            if test_masks is not None:
                print("-------------- Masked metrics ----------------")
                print(
                    "SSIM : {:>12.7f} (std={:>12.7f})".format(
                        torch.tensor(masked_ssims).mean(),
                        torch.tensor(masked_ssims).std(),
                    )
                )
                print(
                    "MSSSIM : {:>12.7f} (std={:>12.7f})".format(
                        torch.tensor(masked_msssims).mean(),
                        torch.tensor(masked_msssims).std(),
                    )
                )
                print(
                    "PSNR : {:>12.7f} (std={:>12.7f})".format(
                        torch.tensor(masked_psnrs).mean(),
                        torch.tensor(masked_psnrs).std(),
                    )
                )
                print(
                    "LPIPS (VGG): {:>12.7f} (std={:>12.7f})".format(
                        torch.tensor(masked_lpipss_vgg).mean(),
                        torch.tensor(masked_lpipss_vgg).std(),
                    )
                )
                print(
                    "LPIPS (ALEX): {:>12.7f} (std={:>12.7f})".format(
                        torch.tensor(masked_lpipss_alex).mean(),
                        torch.tensor(masked_lpipss_alex).std(),
                    )
                )
                print("")

            full_dict[scene_dir][method].update(
                {
                    "SSIM": torch.tensor(ssims).mean().item(),
                    "SSIM std": torch.tensor(ssims).std().item(),
                    "MSSSIM": torch.tensor(msssims).mean().item(),
                    "PSNR": torch.tensor(psnrs).mean().item(),
                    "PSNR std": torch.tensor(psnrs).std().item(),
                    "LPIPS (VGG)": torch.tensor(lpipss_vgg).mean().item(),
                    "LPIPS (VGG) std": torch.tensor(lpipss_vgg).std().item(),
                    "LPIPS (ALEX)": torch.tensor(lpipss_alex).mean().item(),
                    "LPIPS (ALEX) std": torch.tensor(lpipss_alex).std().item(),
                }
            )
            full_dict_masked[scene_dir][method].update(
                {
                    "SSIM": torch.tensor(masked_ssims).mean().item(),
                    "SSIM std": torch.tensor(masked_ssims).std().item(),
                    "MSSSIM": torch.tensor(masked_msssims).mean().item(),
                    "PSNR": torch.tensor(masked_psnrs).mean().item(),
                    "PSNR std": torch.tensor(masked_psnrs).std().item(),
                    "LPIPS (VGG)": torch.tensor(masked_lpipss_vgg).mean().item(),
                    "LPIPS (VGG) std": torch.tensor(masked_lpipss_vgg).std().item(),
                    "LPIPS (ALEX)": torch.tensor(masked_lpipss_alex).mean().item(),
                    "LPIPS (ALEX) std": torch.tensor(masked_lpipss_alex).std().item(),
                }
            )
            frame_idx, camera_idx = 0, 0
            per_frame_metrics = {
                "ssim": defaultdict(list),
                "msssim": defaultdict(list),
                "psnr": defaultdict(list),
                "lpips": defaultdict(list),
            }
            for ssim_i, msssim_i, psnr_i, lpips_vgg_i in zip(
                torch.tensor(ssims).tolist(),
                torch.tensor(msssims).tolist(),
                torch.tensor(psnrs).tolist(),
                torch.tensor(lpipss_vgg).tolist(),
            ):
                if frame_idx == n_frames_per_camera:
                    frame_idx = 0
                    camera_idx += 1
                per_frame_metrics["ssim"][frame_idx].append(ssim_i)
                per_frame_metrics["msssim"][frame_idx].append(msssim_i)
                per_frame_metrics["psnr"][frame_idx].append(psnr_i)
                per_frame_metrics["lpips"][frame_idx].append(lpips_vgg_i)
                frame_idx += 1

            for metric_name in per_frame_metrics.keys():
                for frame_idx, v in per_frame_metrics[metric_name].items():
                    per_frame_metrics[metric_name][frame_idx] = (
                        torch.tensor(v).mean().item()
                    )

            per_frame_dict[scene_dir][method].update(per_frame_metrics)

            print("Computing VMAF scores...")
            # if is_oracle:
            #     video_path = os.path.join(scene_dir, "videos")
            #     if not os.path.exists(video_path):
            #         warnings.warn(
            #             f"Video path {video_path} does not exist. Cannot compute VMAF scores for oracle method.",
            #             UserWarning,
            #         )
            #         continue
            #     for cam in sorted(os.listdir(video_path)):
            #         print(cam)
            #         gt_path = os.path.join(video_path, cam, "gt")
            #         render_path = os.path.join(video_path, cam, "renders")
            #         args = [
            #             "./encode_y4m_video_pairs.sh",
            #             gt_path,
            #             render_path,
            #             str(n_frames_per_camera),
            #             "./temp",
            #         ]
            #         process = subprocess.run(args)
            #         if process.returncode != 0:
            #             raise RuntimeError(
            #                 f"Y4M encoding script failed to run for scene {scene_dir} method {method}."
            #             )
            # else:
            #     gt_path = os.path.join(scene_dir, "test", method, "gt")
            #     render_path = os.path.join(scene_dir, "test", method, "renders")
            #     args = [
            #         "./encode_y4m_video_pairs.sh",
            #         gt_path,
            #         render_path,
            #         str(n_frames_per_camera),
            #         "./temp",
            #     ]
            #     process = subprocess.run(args)
            #     if process.returncode != 0:
            #         raise RuntimeError(
            #             f"Y4M encoding script failed to run for scene {scene_dir} method {method}."
            #         )
            # for cam_i in range(1, 1 + n_images // n_frames_per_camera):
            #     args = [
            #         "vmaf",
            #         f"-r=./temp/y4m_encoded_gt/video_000{cam_i}.y4m",
            #         f"-d=./temp/y4m_encoded_render/video_000{cam_i}.y4m",
            #         "--json",
            #         f"-o=vmaf_{cam_i}.json",
            #     ]
            #     process = subprocess.run(args)
            #     if process.returncode != 0:
            #         raise RuntimeError(
            #             f"VMAF computation failed to run for camera {cam_i}, scene {scene_dir} method {method}."
            #         )

        with open(scene_dir + "/results.json", "w") as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open(scene_dir + "/results_masked.json", "w") as fp:
            json.dump(full_dict_masked[scene_dir], fp, indent=True)
        with open(scene_dir + "/per_frame.json", "w") as fp:
            json.dump(per_frame_dict[scene_dir], fp, indent=True)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument(
        "--model_paths", "-m", required=True, nargs="+", type=str, default=[]
    )
    parser.add_argument("--dynamic_masks_path", required=False, type=str)
    parser.add_argument("--n-frames-per-camera", type=int, default=300)
    parser.add_argument(
        "--oracle", action="store_true", default=False, dest="is_oracle"
    )
    parser.add_argument(
        "--frame_id",
        type=int,
        default=None,
        help="Frame ID to use for masked metrics. Required if --dynamic_masks_path is provided.",
    )
    args = parser.parse_args()
    evaluate(
        args.model_paths,
        args.n_frames_per_camera,
        args.is_oracle,
        args.dynamic_masks_path,
        args.frame_id,
    )
