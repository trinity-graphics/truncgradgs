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
import sys
from argparse import ArgumentParser

import matplotlib.cm as colormap
import numpy as np
import torch
import torchvision
from matplotlib.colors import TwoSlopeNorm
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render
from scene import GaussianModel
from scene.cameras import Camera
from train import prepare_output_and_logger
from utils.general_utils import get_expon_lr_func, safe_state
from utils.loss_utils import l1_loss, ssim
from utils.sh_utils import RGB2SH

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim

    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

SPARSE_ADAM_AVAILABLE = False
# try:
#     SPARSE_ADAM_AVAILABLE = True
# except:
#     SPARSE_ADAM_AVAILABLE = False

IS_BASELINE = os.getenv("IS_BASELINE", "0") == "1"


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(
            "Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel]."
        )

    first_iter = 0
    testing_iterations += [i for i in range(opt.iterations + 1) if i % 100 == 0]
    tb_writer = prepare_output_and_logger(dataset)
    os.makedirs(os.path.join(dataset.model_path, "training_vis"), exist_ok=True)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    img_array = np.zeros((900, 1600, 3), dtype=np.uint8)
    xp = "4"
    if "1" in xp:
        square_size = 30
        # square_x_start, square_y_start = 800, 480 # For the sanity test, XP 0
        # square_x_start, square_y_start = 800, 500 # For XP 1.a
        square_x_start, square_y_start = 1100, 600  # For XP 1.b
        img_array[
            square_y_start : square_y_start + square_size,
            square_x_start : square_x_start + square_size,
        ] = [255, 0, 0]  # RED square
        n_gaussians = 1
        gaussian_pos = torch.tensor(
            [[0.0, 0.0, 5.0]], dtype=torch.float, device="cuda"
        ).repeat(n_gaussians, 1)
        gaussian_color = torch.tensor(
            [5, 0, 0], dtype=torch.float, device="cuda"
        ).repeat(n_gaussians, 1)
    elif "2" in xp:
        square_size = 30
        square_x_start, square_y_start = 1100, 600  # red square
        if xp == "2b":
            img_array = np.ones_like(img_array) * 255
            dataset.white_background = True
        img_array[
            square_y_start : square_y_start + square_size,
            square_x_start : square_x_start + square_size,
        ] = [255, 0, 0]  # RED square
        square_x_start, square_y_start = 550, 350  # blue square
        img_array[
            square_y_start : square_y_start + square_size,
            square_x_start : square_x_start + square_size,
        ] = [0, 0, 255]  # BLUE square
        n_gaussians = 1
        gaussian_pos = torch.tensor(
            [[0.0, 0.0, 5.0]], dtype=torch.float, device="cuda"
        ).repeat(n_gaussians, 1)
        gaussian_color = torch.tensor(
            [5, 0, 0], dtype=torch.float, device="cuda"
        ).repeat(n_gaussians, 1)
        if xp == "2b":
            gaussian_color *= 10
    elif "3" in xp:
        # Neutral colour Gaussian in the middle, to check that it doesn't move
        square_size = 30
        square_x_start, square_y_start = (
            1600 // 4 - square_size // 2,
            900 // 2 - square_size // 2,
        )  # blue square
        img_array[
            square_y_start : square_y_start + square_size,
            square_x_start : square_x_start + square_size,
        ] = [0, 0, 255]  # BLUE square
        # ] = [255, 0, 0]  # RED square
        square_x_start, square_y_start = (
            3 * (1600 // 4) + square_size // 2,
            900 // 2 - square_size // 2,
        )  # red square
        img_array[
            square_y_start : square_y_start + square_size,
            square_x_start : square_x_start + square_size,
        ] = [255, 0, 0]  # RED square
        # ] = [0, 0, 255]  # BLUE square
        n_gaussians = 1
        gaussian_pos = torch.tensor(
            [[0.0, 0.0, 5.0]], dtype=torch.float, device="cuda"
        ).repeat(n_gaussians, 1)
        gaussian_color = torch.tensor(
            [2.5, 0, 2.5], dtype=torch.float, device="cuda"
        ).repeat(n_gaussians, 1)
    elif xp in ["4", "5", "6", "6b"]:
        square_size = 30
        square_x_start, square_y_start = 1100, 600  # red square
        if "5" in xp:
            img_array = np.ones_like(img_array) * 255
            dataset.white_background = True
        img_array[
            square_y_start : square_y_start + square_size,
            square_x_start : square_x_start + square_size,
        ] = [255, 0, 0]  # RED square
        square_x_start, square_y_start = 550, 350  # blue square
        img_array[
            square_y_start : square_y_start + square_size,
            square_x_start : square_x_start + square_size,
        ] = [0, 0, 255]  # BLUE square
        if "6" in xp:
            square_x_start, square_y_start = 1000, 550  # yellow square
            img_array[
                square_y_start : square_y_start + square_size,
                square_x_start : square_x_start + square_size,
            ] = [125, 125, 0]
            square_x_start, square_y_start = 800, 500  # green square
            img_array[
                square_y_start : square_y_start + square_size,
                square_x_start : square_x_start + square_size,
            ] = [0, 255, 0]
            square_x_start, square_y_start = 700, 510  # white square
            img_array[
                square_y_start : square_y_start + square_size,
                square_x_start : square_x_start + square_size,
            ] = [255, 255, 255]

        if "6" in xp:
            n_gaussians = 100
            gaussian_pos = torch.randn(
                (n_gaussians, 3), dtype=torch.float, device="cuda"
            )
        else:
            n_gaussians = 10
            gaussian_pos = torch.tensor(
                # [[2.5, -2.5, 5.0]], dtype=torch.float, device="cuda"
                [
                    [1.0, -1.0, 5.0],
                    [-1.0, -1.0, 5.0],
                    [-1.0, 1.0, 5.0],
                    [1.0, 1.0, 5.0],
                    [0.0, 0.0, 5.0],
                    [2.0, -2.0, 5.0],
                    [-2.0, -2.0, 5.0],
                    [-2.0, 2.0, 5.0],
                    [2.0, 2.0, 5.0],
                    [-1.0, 2.3, 5.0],
                ],
                dtype=torch.float,
                device="cuda",
            )
        if xp == "5":
            gaussian_color = torch.rand(
                (n_gaussians, 3), dtype=torch.float, device="cuda"
            )
        else:
            gaussian_color = (
                torch.rand((n_gaussians, 3), dtype=torch.float, device="cuda") * 5
                + torch.randn((n_gaussians, 3), dtype=torch.float, device="cuda") * 2
            )
        # gaussian_color = torch.tensor(
        #     [0, 0, 5.0], dtype=torch.float, device="cuda"
        # ).repeat(n_gaussians, 1)
        # gaussian_color = torch.tensor(
        #     [[0, 0, 5.0], [5.0, 0.0, 0]], dtype=torch.float, device="cuda"
        # )
    else:
        raise Exception(
            "Please specify which experiment to run by setting the variable xp to a value between 1 and 4."
        )

    pil_image = Image.fromarray(img_array)
    viewpoint_cam = Camera(
        (1600, 900),
        0,
        np.eye(3),
        np.array([0.0, 0.0, 3.0]),
        1.0,
        1.0,
        None,
        pil_image,
        None,
        "debug_view",
        "debug_view",
        train_test_exp=False,
        data_device="cuda",
    )

    gaussians._xyz = torch.nn.Parameter(gaussian_pos)
    # WARN: If I use 255 for the colour, the gaussian blob has a hard edge and is not
    # ever optimized. How strange???
    fused_color = RGB2SH(gaussian_color)
    features = (
        torch.zeros((fused_color.shape[0], 3, (gaussians.max_sh_degree + 1) ** 2))
        .float()
        .cuda()
    )
    features[:, :3, 0] = fused_color
    features[:, 3:, 1:] = 0.0
    gaussians._features_dc = torch.nn.Parameter(
        features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
    )
    gaussians._features_rest = torch.nn.Parameter(
        features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
    )
    scale = torch.tensor([0.2, 0.2, 0.0], dtype=torch.float, device="cuda").repeat(
        n_gaussians, 1
    )
    gaussians._scaling = torch.nn.Parameter(torch.log(scale))
    # gaussians._scaling = torch.log(scale).requires_grad_(False)
    rots = torch.zeros((1, 4), dtype=torch.float, device="cuda").repeat(n_gaussians, 1)
    rots[:, 0] = 1
    # gaussians._rotation = torch.nn.Parameter(rots.requires_grad_(True))
    gaussians._rotation = rots.requires_grad_(False)
    opacities = gaussians.inverse_opacity_activation(
        (0.5 if not dataset.white_background else 0.9)
        * torch.ones((1, 1), dtype=torch.float, device="cuda").repeat(n_gaussians, 1)
    )
    gaussians._opacity = opacities.requires_grad_(False)
    # gaussians._opacity = torch.nn.Parameter(opacities)
    gaussians._exposure = torch.nn.Parameter(
        torch.eye(3, 4, dtype=torch.float, device="cuda")[None].repeat(
            n_gaussians, 1, 1
        )
    )
    gaussians.pretrained_exposures = None
    gaussians.max_radii2D = torch.zeros((n_gaussians), device="cuda")
    gaussians.spatial_lr_scale = 1.0
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(
        opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations
    )

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_kwargs = {
            "use_trained_exp": dataset.train_test_exp,
            "separate_sh": SPARSE_ADAM_AVAILABLE,
        }
        if not IS_BASELINE:
            render_kwargs["mu_grad_pull_strength"] = 0.1
            render_kwargs["exploration_on"] = False

        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
            **render_kwargs,
        )
        # render_pkg = render(
        #     viewpoint_cam,
        #     gaussians,
        #     pipe,
        #     bg,
        #     use_trained_exp=dataset.train_test_exp,
        #     separate_sh=SPARSE_ADAM_AVAILABLE,
        #     mu_grad_pull_strength=0.000003,  # 0.0000001,
        # )
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        if not IS_BASELINE:
            synth_grad_3D = render_pkg["synthetic_grad"]
            if iteration % 10 == 0:
                true_grads_pp, trunc_grads_pp = (
                    render_pkg["true_grads_pp"],
                    render_pkg["trunc_grads_pp"],
                )
                cmap = colormap.get_cmap("inferno")
                true_grads_image = true_grads_pp.to(torch.float32)
                true_grads_image = (true_grads_image - true_grads_image.min()) / (
                    true_grads_image.max() - true_grads_image.min()
                )
                x_np = true_grads_image.cpu().numpy()  # (H, W)
                colored = cmap(x_np)  # (H, W, 4)
                colored = colored[..., :3]  # (H, W, 3)
                true_grads_image = torch.from_numpy(colored).permute(
                    2, 0, 1
                )  # (3, H, W)
                os.makedirs(
                    os.path.join(dataset.model_path, "training_vis"), exist_ok=True
                )
                torchvision.utils.save_image(
                    true_grads_image[None],
                    os.path.join(
                        dataset.model_path,
                        "training_vis",
                        f"true_grads_per_pixel_iter_{iteration:05d}.png",
                    ),
                )
                trunc_grads_image = trunc_grads_pp.to(torch.float32)
                trunc_grads_image = (trunc_grads_image - trunc_grads_image.min()) / (
                    trunc_grads_image.max() - trunc_grads_image.min()
                )
                x_np = trunc_grads_image.cpu().numpy()  # (H, W)
                colored = cmap(x_np)  # (H, W, 4)
                colored = colored[..., :3]  # (H, W, 3)
                trunc_grads_image = torch.from_numpy(colored).permute(
                    2, 0, 1
                )  # (3, H, W)
                torchvision.utils.save_image(
                    trunc_grads_image[None],
                    os.path.join(
                        dataset.model_path,
                        "training_vis",
                        f"trunc_grads_per_pixel_iter_{iteration:05d}.png",
                    ),
                )

                cmap = colormap.get_cmap("bone")
                true_grads_image = true_grads_pp.to(torch.float32)
                true_grads_image = (true_grads_image - true_grads_image.min()) / (
                    true_grads_image.max() - true_grads_image.min()
                )
                x_np = true_grads_image.cpu().numpy()  # (H, W)
                colored = cmap(x_np)  # (H, W, 4)
                colored = colored[..., :3]  # (H, W, 3)
                true_grads_image = torch.from_numpy(colored).permute(
                    2, 0, 1
                )  # (3, H, W)
                cmap = colormap.get_cmap("copper")
                trunc_grads_image = trunc_grads_pp.to(torch.float32)
                trunc_grads_image = (trunc_grads_image - trunc_grads_image.min()) / (
                    trunc_grads_image.max() - trunc_grads_image.min()
                )
                x_np = trunc_grads_image.cpu().numpy()  # (H, W)
                colored = cmap(x_np)  # (H, W, 4)
                colored = colored[..., :3]  # (H, W, 3)
                trunc_grads_image = torch.from_numpy(colored).permute(
                    2, 0, 1
                )  # (3, H, W)
                torchvision.utils.save_image(
                    trunc_grads_image[None] + true_grads_image[None],
                    os.path.join(
                        dataset.model_path,
                        "training_vis",
                        f"true_vs_trunc_grads_per_pixel_iter_{iteration:05d}.png",
                    ),
                )

                if xp not in ["4",  "6", "6b"]:
                    true_grad_mags_pp, trunc_grad_mags_pp = (
                        render_pkg["true_grad_mags_pp"].detach(),
                        render_pkg["trunc_grad_mags_pp"].detach(),
                    )
                    cmap = colormap.get_cmap("inferno")
                    true_mags_image = true_grad_mags_pp.to(torch.float32)
                    if true_mags_image.max() > true_mags_image.min():
                        true_mags_image = (true_mags_image - true_mags_image.min()) / (
                            true_mags_image.max() - true_mags_image.min()
                        )
                    x_np = true_mags_image.cpu().numpy()  # (H, W)
                    colored = cmap(x_np)  # (H, W, 4)
                    colored = colored[..., :3]  # (H, W, 3)
                    true_mags_image = torch.from_numpy(colored).permute(
                        2, 0, 1
                    )  # (3, H, W)
                    torchvision.utils.save_image(
                        true_mags_image[None],
                        os.path.join(
                            dataset.model_path,
                            "training_vis",
                            f"true_mags_per_pixel_iter_{iteration:05d}.png",
                        ),
                    )

                    trunc_mags_image = trunc_grad_mags_pp.to(torch.float32)
                    if trunc_mags_image.max() > trunc_mags_image.min():
                        trunc_mags_image = (
                            trunc_mags_image - trunc_mags_image.min()
                        ) / (trunc_mags_image.max() - trunc_mags_image.min())
                    x_np = trunc_mags_image.cpu().numpy()  # (H, W)
                    colored = cmap(x_np)  # (H, W, 4)
                    colored = colored[..., :3]  # (H, W, 3)
                    trunc_mags_image = torch.from_numpy(colored).permute(
                        2, 0, 1
                    )  # (3, H, W)
                    torchvision.utils.save_image(
                        trunc_mags_image[None],
                        os.path.join(
                            dataset.model_path,
                            "training_vis",
                            f"trunc_mags_per_pixel_iter_{iteration:05d}.png",
                        ),
                    )
                    # cmap = colormap.get_cmap("coolwarm")
                    # norm = TwoSlopeNorm(
                    #     vmin=trunc_mags_image.min(),
                    #     vcenter=0,
                    #     vmax=trunc_mags_image.max()+1e-8,
                    # )
                    # colored = cmap(norm(trunc_mags_image.cpu().numpy()))  # (H, W, 4)
                    # colored = colored[..., :3]  # (H, W, 3)
                    # trunc_mags_image = torch.from_numpy(colored).permute(
                    #     2, 0, 1
                    # )  # (3, H, W)
                    # torchvision.utils.save_image(
                    #     trunc_mags_image[None],
                    #     os.path.join(
                    #         dataset.model_path,
                    #         "training_vis",
                    #         f"trunc_mags_per_pixel_iter_{iteration:05d}.png",
                    #     ),
                    # )

            method = 3
            if method == 1:
                # Method 1: use the norm of the SH coeff gradients:
                if gaussians._features_dc.grad is not None:
                    reconstruct_err = gaussians._features_dc.grad.norm(dim=-1).squeeze()
                    threshold = 0.05
                    pull_weight = (
                        1.0 - torch.exp(-reconstruct_err / threshold)
                    ).unsqueeze(-1)
                else:
                    pull_weight = 1
            elif method == 2:
                # Method 2: use dL_dalpha:
                reconstruct_err = render_pkg["dL_dalpha"].squeeze()
                threshold = 1.5
                pull_weight = (1.0 - torch.exp(-reconstruct_err / threshold)).unsqueeze(
                    -1
                )
            elif method == 3:
                pull_weight = 1
            elif method == 4:
                update_mask = ~render_pkg[
                    "dL_dalpha_neg_mask"
                ]  # Those that didn't get marked can receive the synth grad
                # print(f"Number of Gaussians that will receive the truncated grad: {update_mask.sum().item()}")
                pull_weight = update_mask.float().to(synth_grad_3D.device)[:, None]
                assert pull_weight.max().item() <= 1.0
            else:
                raise ValueError("unknown method")
            gaussians._xyz.grad += synth_grad_3D * pull_weight
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{7}f}",
                        "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}",
                    }
                )
                progress_bar.update(10)

            if iteration == opt.iterations:
                progress_bar.close()

            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                gaussians,
                viewpoint_cam,
                render,
                (
                    pipe,
                    background,
                    1.0,
                    SPARSE_ADAM_AVAILABLE,
                    None,
                    dataset.train_test_exp,
                ),
                dataset.train_test_exp,
                dataset.model_path,
                dataset.white_background,
            )

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )

                if (
                    xp == "6b"
                    and iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        10,
                        size_threshold,
                        radii,
                    )

                if (
                    xp == "6b"
                    and iteration % opt.opacity_reset_interval == 0
                    or (
                        xp == "6b"
                        and dataset.white_background
                        and iteration == opt.densify_from_iter
                    )
                ):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

            # if iteration in checkpoint_iterations:
            #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
            #     torch.save(
            #         (gaussians.capture(), iteration),
            #         scene.model_path + "/chkpnt" + str(iteration) + ".pth",
            #     )


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    gaussians,
    viewpoint_cam,
    renderFunc,
    renderArgs,
    train_test_exp,
    model_path,
    white_background,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)
        vs_grad_norm = gaussians.xyz_gradient_accum / gaussians.denom
        vs_grad_norm[vs_grad_norm.isnan()] = 0.0
        tb_writer.add_scalar(
            "viewpace_grad_norm", vs_grad_norm.mean().cpu().item(), iteration
        )

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {
                "name": "debug",
                "cameras": [viewpoint_cam],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                for _, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        renderFunc(viewpoint, gaussians, *renderArgs)["render"],
                        0.0,
                        1.0,
                    )
                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2 :]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2 :]
                    if tb_writer:
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render_with_target".format(
                                viewpoint.image_name
                            ),
                            image[None] * gt_image[None]
                            if white_background
                            else image[None] + gt_image[None],
                            global_step=iteration,
                        )
                        torchvision.utils.save_image(
                            image[None] + gt_image[None],
                            os.path.join(
                                model_path, "training_vis", f"iter_{iteration:05d}.png"
                            ),
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test += l1_loss(image, gt_image).mean().double()
                l1_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} ".format(
                        iteration, config["name"], l1_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", gaussians.get_opacity, iteration
            )
            tb_writer.add_scalar("total_points", gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_viewer", action="store_true", default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
    )

    # All done
    print("\nTraining complete.")
