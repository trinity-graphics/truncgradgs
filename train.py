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
import traceback
import uuid
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from random import randint

import matplotlib.cm as colormap
import torch
import torchvision
from tqdm import tqdm

import wandb
from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render
from lpipsPyTorch.modules.lpips import LPIPS
from scene import GaussianModel, Scene
from utils.general_utils import get_expon_lr_func, safe_state
from utils.image_utils import draw_text, pil_image_to_tensor, psnr, tensor_to_pil_image
from utils.loss_utils import l1_loss, ssim

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
DEBUG_VIS = int(os.getenv("DEBUG_VIS", "0"))
USING_REVISED_ADC = os.getenv("USING_REVISED_ADC", "0") == "1"
INTERLEAVED_ADC = not IS_BASELINE and os.getenv("INTERLEAVED_ADC", "1") == "1"


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    recovering,
    wandb_run=None,
):

    if recovering:
        print("Fine tuning from checkpoint...")
        opt.iterations = (
            20_000 if not IS_BASELINE else 12_000
        )  # 5K for trunc-grad, 2K for ADC, then 5K again for final trunc-grad round.
        saving_iterations += [20_000 if not IS_BASELINE else 12_000]
        opt.position_lr_init = 0.00008

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(
            "Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel]."
        )

    testing_iterations += [i for i in range(opt.iterations + 1) if i % 5000 == 0]
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        try:
            (model_params, _) = torch.load(checkpoint)
            gaussians.restore(model_params, opt, restore_optimizer=False)
        except Exception:
            print("Could not load checkpoint, trying to load .ply.")
            gaussians.load_ply(checkpoint)
            gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(
        opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations
    )

    # mu_grad_pull_strength_init = 0.0000001
    # mu_grad_pull_strength_max = (
    #     0.00001  # This value should basically stop the effect of the slope
    # )
    # mu_grad_pull_scheduler = get_expon_lr_func(
    #     lr_init=mu_grad_pull_strength_init,
    #     lr_final=mu_grad_pull_strength_max,
    #     max_steps=opt.iterations,
    # )

    viewpoint_stack = scene.getTrainCameras().copy()
    n_cameras = len(viewpoint_stack)
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    primitive_ek_pv = defaultdict(list)
    did_densify = False
    last_adc_switch = 0
    adc_on = True
    opa_is_reset = None

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam != None:
                    net_image = render(
                        custom_cam,
                        gaussians,
                        pipe,
                        background,
                        scaling_modifier=scaling_modifer,
                        use_trained_exp=dataset.train_test_exp,
                        separate_sh=SPARSE_ADAM_AVAILABLE,
                    )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and (
                    (iteration < int(opt.iterations)) or not keep_alive
                ):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()
        if not IS_BASELINE and iteration > opt.densify_from_iter:
            if iteration == 25_000:
                print("FINE TUNING - ONLY OURS FROM NOW ON")
                adc_on = False
                last_adc_switch = iteration
                gaussians.reset_stats()
            elif (
                adc_on
                and (iteration - last_adc_switch) >= opt.adc_duration
                and iteration < 25_000
            ):
                adc_on = False
                last_adc_switch = iteration
                print("ADC IS NOW " + ("ON" if adc_on else "OFF"))
                gaussians.reset_stats()
            elif (
                not adc_on
                and (iteration - last_adc_switch) >= opt.adc_interval
                and iteration < 25_000
            ):
                adc_on = True
                last_adc_switch = iteration
                print("ADC IS NOW " + ("ON" if adc_on else "OFF"))
                gaussians.reset_stats()

        gaussians.update_learning_rate(iteration)
        # Linear scheduler from 0 to max on every 1000 iterations:
        # slope_factor = mu_grad_pull_strength_max / 1000
        # mu_grad_pull_strength = min(
        #     mu_grad_pull_strength_init + (iteration // 1000) * slope_factor,
        #     mu_grad_pull_strength_max,
        # )
        # mu_grad_pull_strength = mu_grad_pull_scheduler(iteration)
        # if iteration % 1000 == 0:
        #     print(f"Mu grad pull strength: {mu_grad_pull_strength:.4f}")
        # mu_grad_pull_strength = 1.0

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_kwargs = {
            "use_trained_exp": dataset.train_test_exp,
            "separate_sh": SPARSE_ADAM_AVAILABLE,
            "use_trunc": not IS_BASELINE and (not INTERLEAVED_ADC or not adc_on),
        }
        if not IS_BASELINE:
            if opt.disable_pull_strength_schedule:
                render_kwargs["mu_grad_pull_strength"] = 1
            else:
                render_kwargs["mu_grad_pull_strength"] = (
                    3
                    if iteration < opt.iterations // 3
                    else 2
                    if iteration < 2 * opt.iterations // 3
                    else 1
                )
            render_kwargs["exploration_on"] = False
            render_kwargs["scene_extent"] = scene.cameras_extent
            assert viewpoint_cam.motion_mask is not None, (
                "Motion mask is required for non-baseline training!"
            )
            render_kwargs["motion_mask"] = (
                viewpoint_cam.motion_mask.cuda()
                if viewpoint_cam.motion_mask is not None
                else None
            )

        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
            **render_kwargs,
        )

        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        if not USING_REVISED_ADC:
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
                1.0 - ssim_value
            )
            loss.backward()
        else:
            # Accessing the per-gaussian alpha-blending-coefficient sum without reimplementing the rasterizer.
            # Here we follow the approach of Revising Densification in Gaussian Splatting by Bulò et al. that reduces runtime performance, but does not require a new rasterizer implementation.

            # Render all-zero dummy image using all-zero fake color.
            fake_color = torch.zeros_like(gaussians._xyz, requires_grad=True)
            fake_render = render(
                viewpoint_cam,
                gaussians,
                pipe,
                torch.tensor(
                    [0, 0, 0], dtype=torch.float32, device="cuda"
                ),  # Use a black background to avoid contributions from the background
                override_color=fake_color,
            )["render"]
            # assert torch.all(fake_render == 0), (
            #     "The fake render should be all zeros because the fake color is all zeros."
            # )
            ssim_loss = (
                ssim(image, gt_image, return_raw=True, size_average=False)
                .mean(dim=0)
                .view(-1)
            )  # Returns an (C,H,W) tensor with the per-pixel ssim loss
            fake_loss = (
                (1.0 - ssim_loss.detach()) * fake_render.mean(dim=0).view(-1)
            ).sum()
            # assert fake_loss.item() == 0, (
            #     "The fake loss should be 0 because the fake color is all zeros."
            # )
            # assert torch.allclose(ssim_loss.mean(), ssim(image, gt_image)), (
            #     "The mean of the per-pixel ssim loss should be equal to the ssim loss computed with size_average=True."
            # )

            # TODO: Return per-pixel accumulated alpha in the rasterizer, so we can
            # remove one forward/backward pass!
            # FIXME: This is wrong because the rasterizer returns C = \sum_i (c_i w_i
            # \prod_j^(i-1) (1 - w_j)) + \prod_i (1 - w_i) * bg, where w_i is the
            # alpha-blending-coefficient of Gaussian i. But what we want is the output
            # without the background contribution.
            # ones_color = torch.ones_like(gaussians._xyz, requires_grad=True)
            # render_sum = render(
            #     viewpoint_cam,
            #     gaussians,
            #     pipe,
            #     torch.tensor([0,0,0], dtype=torch.float32, device="cuda"),
            #     override_color=ones_color,
            # )["render"]
            # # Transmittancce per pixel:
            # sum_weights = render_sum.mean(dim=0)   # (H, W)
            # # Residual transmittance per pixel:
            # T_res = 1.0 - sum_weights
            # residual_transmittance = T_res.mean()

            # Compute loss and add fake auxilary loss which is allways 0.
            Ll1 = l1_loss(image, gt_image)
            loss = (
                (1.0 - opt.lambda_dssim) * Ll1
                + opt.lambda_dssim * (1.0 - ssim_loss.mean())
                + fake_loss
            )
            # if iteration > opt.densify_from_iter:
            #     loss += 0.1 * residual_transmittance
            loss.backward()

            # TODO: Track the max E_k (fake_color.grad) across all views seen between two
            # runs of the ADC module (ie reset the state when we run ADC).
            assert fake_color.grad is not None
            # INFO: Here we accumulate the average per view before taking the max
            # across views, when we run ADC.
            # Gradients of the fake_color tensor yield per-gaussian alpha-blending-coefficient sum.
            primitive_ek_pv[viewpoint_cam.uid].append(
                torch.mean(fake_color.grad, dim=1)
            )

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

        if DEBUG_VIS == 2:
            os.makedirs(os.path.join(scene.model_path, "training_vis"), exist_ok=True)
            if iteration % 1 == 0 and viewpoint_cam.uid == 5 and iteration >= 4500:
                img = torch.clamp(render_pkg["render"].detach(), 0.0, 1.0)
                draw_image = tensor_to_pil_image(img)
                try:
                    draw_text(
                        draw_image,
                        (
                            "ADC on / synth grad off"
                            if adc_on
                            else "ADC off / synth grad on"
                        )
                        + f"\nIter: {iteration}",
                        color="blue",
                        size=32,
                    )
                except Exception as e:
                    print("Could not draw text on image: {}".format(e))
                draw_image = pil_image_to_tensor(draw_image).to(gt_image.device)
                torchvision.utils.save_image(
                    draw_image[None],
                    os.path.join(
                        scene.model_path,
                        "training_vis",
                        f"render_{iteration:05d}.png",
                    ),
                )

        if not IS_BASELINE and not adc_on:
            truncated_grad_3D = render_pkg["synthetic_grad"]
            if DEBUG_VIS == 1:
                if (
                    iteration < 10_000
                    and iteration % 1 == 0
                    and viewpoint_cam.uid == 10
                ):
                    true_grads_pp, trunc_grads_pp = (
                        render_pkg["true_grads_pp"],
                        render_pkg["trunc_grads_pp"],
                    )
                    true_grad_mags_pp, trunc_grad_mags_pp = (
                        render_pkg["true_grad_mags_pp"].detach(),
                        render_pkg["trunc_grad_mags_pp"].detach(),
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
                        os.path.join(scene.model_path, "training_vis"), exist_ok=True
                    )
                    torchvision.utils.save_image(
                        true_grads_image[None],
                        os.path.join(
                            scene.model_path,
                            "training_vis",
                            f"true_grads_per_pixel_iter_{iteration:05d}.png",
                        ),
                    )
                    trunc_grads_image = trunc_grads_pp.to(torch.float32)
                    trunc_grads_image = (
                        trunc_grads_image - trunc_grads_image.min()
                    ) / (trunc_grads_image.max() - trunc_grads_image.min())
                    x_np = trunc_grads_image.cpu().numpy()  # (H, W)
                    colored = cmap(x_np)  # (H, W, 4)
                    colored = colored[..., :3]  # (H, W, 3)
                    trunc_grads_image = torch.from_numpy(colored).permute(
                        2, 0, 1
                    )  # (3, H, W)
                    torchvision.utils.save_image(
                        trunc_grads_image[None],
                        os.path.join(
                            scene.model_path,
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
                    trunc_grads_image = (
                        trunc_grads_image - trunc_grads_image.min()
                    ) / (trunc_grads_image.max() - trunc_grads_image.min())
                    x_np = trunc_grads_image.cpu().numpy()  # (H, W)
                    colored = cmap(x_np)  # (H, W, 4)
                    colored = colored[..., :3]  # (H, W, 3)
                    trunc_grads_image = torch.from_numpy(colored).permute(
                        2, 0, 1
                    )  # (3, H, W)
                    torchvision.utils.save_image(
                        trunc_grads_image[None] + true_grads_image[None],
                        os.path.join(
                            scene.model_path,
                            "training_vis",
                            f"true_vs_trunc_grads_per_pixel_iter_{iteration:05d}.png",
                        ),
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
                            scene.model_path,
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
                            scene.model_path,
                            "training_vis",
                            f"trunc_mags_per_pixel_iter_{iteration:05d}.png",
                        ),
                    )

                    # Visualize Gaussians that were not marked as "having received true grad" (i.e. those that are not contributing to the current error and thus receive the synthetic grad) vs those that were marked:
                    vis_mask = ~render_pkg[
                        "dL_dalpha_neg_mask"
                    ]  # Those that didn't get marked can receive the synth grad
                    render_kwargs["mask"] = vis_mask
                    render_kwargs["opacity_ovrd"] = 0.3
                    out = render(
                        viewpoint_cam,
                        gaussians,
                        pipe,
                        bg,
                        **render_kwargs,
                    )

                    image = out["render"]
                    torchvision.utils.save_image(
                        image[None],
                        os.path.join(
                            scene.model_path,
                            "training_vis",
                            f"trunc_grad_gaussians_iter_{iteration:05d}.png",
                        ),
                    )
                    render_kwargs["mask"] = ~vis_mask
                    out = render(
                        viewpoint_cam,
                        gaussians,
                        pipe,
                        bg,
                        **render_kwargs,
                    )

                    image = out["render"]
                    torchvision.utils.save_image(
                        image[None],
                        os.path.join(
                            scene.model_path,
                            "training_vis",
                            f"true_grad_gaussians_iter_{iteration:05d}.png",
                        ),
                    )

            with torch.no_grad():
                method = 2
                if method == 1:
                    if (INTERLEAVED_ADC and not adc_on) or not INTERLEAVED_ADC:
                        # Step decay: we apply a stronger pull in the beginning of
                        # training, to promote exploration, and then reduce it to avoid
                        # harming convergence. The decay is based on the iteration
                        # number, and is not adaptive to the state of the training.
                        strength = (
                            30
                            if iteration < opt.iterations // 3
                            else 15
                            if iteration < 2 * opt.iterations // 3
                            else 1
                        )
                        scale = strength * (
                            gaussians._xyz.grad.norm()
                            / (truncated_grad_3D.norm() + 1e-8)
                        )
                        # scale = strength
                        if opa_is_reset is None or (iteration - opa_is_reset) > 100:
                            gaussians._xyz.grad += scale * truncated_grad_3D
                            opa_is_reset = None
                elif method == 2:
                    if not opt.disable_truncated_gradient and (
                        (INTERLEAVED_ADC and not adc_on) or not INTERLEAVED_ADC
                    ):
                        if opa_is_reset is None or (iteration - opa_is_reset) > 100:
                            gaussians._xyz.grad += truncated_grad_3D
                            opa_is_reset = None
                else:
                    raise ValueError("unknown method")
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

            # Log and save
            training_report_wandb(
                wandb_run,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
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
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            densify_until = (
                27_000
                if USING_REVISED_ADC
                else (opt.densify_until_iter if (IS_BASELINE or not INTERLEAVED_ADC) else 27_000)
            )
            if iteration < densify_until:
                if not USING_REVISED_ADC:
                    # INFO : Accumulate even during the truncated grad phase, because we
                    # want to split large Gaussians. Otherwise we're using them but
                    # they'll get pruned when ADC kicks in, instead of being split!

                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )
                    gaussians.add_densification_stats(
                        viewspace_point_tensor, visibility_filter
                    )

                if (
                    iteration > opt.densify_from_iter
                    and (not INTERLEAVED_ADC or (INTERLEAVED_ADC and adc_on))
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )

                    if USING_REVISED_ADC:
                        primitive_ek_pv = {
                            k: torch.stack(v).mean(dim=0)
                            for k, v in primitive_ek_pv.items()
                        }
                        max_ek_prim = (
                            torch.stack(list(primitive_ek_pv.values()))
                            .max(dim=0)
                            .values
                        )
                        gaussians.densify_and_prune_revised(
                            max_ek_prim,
                            max_ek=0.1,
                            growth_max_pctg=0.05,
                            max_primitives=700_000,
                            min_opacity=0.005,
                            extent=scene.cameras_extent,
                            max_screen_size=size_threshold,
                            radii=radii,
                        )
                        primitive_ek_pv = defaultdict(list)
                    else:
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold,
                            0.005,
                            # 0.005
                            # if IS_BASELINE or iteration > opt.pruning_from_iter
                            # else 0,
                            scene.cameras_extent,
                            None if not IS_BASELINE else size_threshold,
                            radii,
                            # prune_mask=None if iteration > opt.pruning_from_iter else
                            # gaussians._xyz.grad.norm(dim=-1) < 2e-5,
                            prune=IS_BASELINE or iteration > opt.pruning_from_iter,
                        )
                elif (
                    iteration > opt.densify_from_iter
                    and iteration % (2 * opt.densification_interval) == 0
                    and not IS_BASELINE
                ):
                    # INFO: We're int the truncated grad phase and we want to split very
                    # large Gaussians. Those aren't passing the low pass filter in the
                    # truncated grad rasterizer anyway, so they receive normal gradients,
                    # and as such can safely be split.
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )

                    grads = gaussians.xyz_gradient_accum / gaussians.denom
                    grads[grads.isnan()] = 0.0
                    gaussians.tmp_radii = radii
                    gaussians.densify_and_split(
                        grads,
                        opt.densify_grad_threshold,
                        scene.cameras_extent,
                        # max_screen_size=size_threshold,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()
                    opa_is_reset = iteration

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

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )

            # if did_densify:
            #     gaussians._opacity -= gaussians.inverse_opacity_activation(torch.tensor(0.0001))
            #     did_densify = False


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report_tb(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    train_test_exp,
    mean_grad_ratios,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)
        vs_grad_norm = scene.gaussians.xyz_gradient_accum / scene.gaussians.denom
        vs_grad_norm[vs_grad_norm.isnan()] = 0.0
        tb_writer.add_scalar(
            "viewpace_grad_norm", vs_grad_norm.mean().cpu().item(), iteration
        )
        # if mean_grad_ratios is not None:
        #     grad_ratios = mean_grad_ratios.cpu().view(-1)
        #     values = grad_ratios.numpy()
        #     fig, ax = plt.subplots()
        #     ax.bar(range(len(values)), values)
        #     ax.set_title("XYZ (true_grad - truncated_grad)")
        #     ax.set_xlabel("Index")
        #     ax.set_ylabel("Value")
        #     tb_writer.add_figure(
        #         "xyz_grad_ratios",
        #         fig,
        #         global_step=iteration,
        #     )
        #     plt.close(fig)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"],
                        0.0,
                        1.0,
                    )
                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2 :]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2 :]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                            if viewpoint.motion_mask is not None:
                                mask = viewpoint.motion_mask[
                                    None, :, :, None
                                ].float()  # (1, H, W, 1)
                                grayscale_map = torch.cat(
                                    [mask, mask, mask], dim=-1
                                ).permute(0, 3, 1, 2)  # (1, 3, H, W)
                                tb_writer.add_images(
                                    config["name"]
                                    + f"_view_{viewpoint.image_name}/dynamic_mask",
                                    grayscale_map,
                                )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    if FUSED_SSIM_AVAILABLE:
                        ssim_test += fused_ssim(
                            image.unsqueeze(0), gt_image.unsqueeze(0)
                        )
                    else:
                        ssim_test += ssim(image, gt_image)
                psnr_test /= len(config["cameras"])
                ssim_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - ssim", ssim_test, iteration
                    )

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
            )
            tb_writer.add_scalar(
                "total_points", scene.gaussians.get_xyz.shape[0], iteration
            )
        torch.cuda.empty_cache()


def training_report_wandb(
    wandb_run,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    train_test_exp,
):
    if wandb_run:
        vs_grad_norm = scene.gaussians.xyz_gradient_accum / scene.gaussians.denom
        vs_grad_norm[vs_grad_norm.isnan()] = 0.0
        wandb_run.log(
            {
                "train_loss_patches/l1_loss": Ll1.item(),
                "train_loss_patches/total_loss": loss.item(),
                "iter_time": elapsed,
                "viewpace_grad_norm": vs_grad_norm.mean().cpu().item(),
            },
            step=iteration,
        )

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )
        lpips_model_alex = LPIPS(net_type="alex").to("cuda")
        lpips_model_vgg = LPIPS(net_type="vgg").to("cuda")

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test_alex = 0.0
                lpips_test_vgg = 0.0
                l1_test_motion = 0.0
                psnr_test_motion = 0.0
                ssim_test_motion = 0.0
                lpips_test_alex_motion = 0.0
                lpips_test_vgg_motion = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"],
                        0.0,
                        1.0,
                    )
                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )
                    motion_mask = (
                        viewpoint.motion_mask.cuda()
                        if viewpoint.motion_mask is not None
                        else torch.ones_like(gt_image[..., 0])
                    )
                    masked_gt = motion_mask * gt_image
                    masked_render = motion_mask * image
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2 :]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2 :]
                        masked_gt = masked_gt[..., masked_gt.shape[-1] // 2 :]
                        masked_render = masked_render[
                            ..., masked_render.shape[-1] // 2 :
                        ]
                    if wandb_run and (idx < 5):
                        wandb_run.log(
                            {
                                f"{config['name']}_view_{viewpoint.image_name}/render": wandb.Image(
                                    image
                                ),
                                f"{config['name']}_view_{viewpoint.image_name}/masked_render": wandb.Image(
                                    masked_render
                                ),
                            },
                            step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            wandb_run.log(
                                {
                                    f"{config['name']}_view_{viewpoint.image_name}/ground_truth": wandb.Image(
                                        gt_image
                                    ),
                                    f"{config['name']}_view_{viewpoint.image_name}/gt_masked_motion": wandb.Image(
                                        masked_gt
                                    ),
                                },
                                step=iteration,
                            )
                            # if viewpoint.motion_mask is not None:
                            #     mask = viewpoint.motion_mask[None, :, :, None].float()
                            #     grayscale_map = torch.cat([mask, mask, mask], dim=-1).permute(
                            #         0, 3, 1, 2
                            #     )
                            #     wandb_run.log(
                            #         {
                            #             f"{config['name']}_view_{viewpoint.image_name}/dynamic_mask": wandb.Image(
                            #                 grayscale_map
                            #             ),
                            #         },
                            #         step=iteration,
                            #     )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    l1_test_motion += l1_loss(masked_render, masked_gt).mean().double()
                    psnr_test_motion += psnr(masked_render, masked_gt).mean().double()
                    if FUSED_SSIM_AVAILABLE:
                        ssim_test += fused_ssim(
                            image.unsqueeze(0), gt_image.unsqueeze(0)
                        )
                        ssim_test_motion += fused_ssim(
                            masked_render.unsqueeze(0), masked_gt.unsqueeze(0)
                        )
                    else:
                        ssim_test += ssim(image, gt_image)
                        ssim_test_motion += ssim(masked_render, masked_gt)
                    lpips_test_alex += lpips_model_alex(
                        image.unsqueeze(0), gt_image.unsqueeze(0)
                    ).item()
                    lpips_test_vgg += lpips_model_vgg(
                        image.unsqueeze(0), gt_image.unsqueeze(0)
                    ).item()
                    lpips_test_alex_motion += lpips_model_alex(
                        masked_render.unsqueeze(0), masked_gt.unsqueeze(0)
                    ).item()
                    lpips_test_vgg_motion += lpips_model_vgg(
                        masked_render.unsqueeze(0), masked_gt.unsqueeze(0)
                    ).item()
                psnr_test /= len(config["cameras"])
                ssim_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                psnr_test_motion /= len(config["cameras"])
                ssim_test_motion /= len(config["cameras"])
                l1_test_motion /= len(config["cameras"])
                lpips_test_alex /= len(config["cameras"])
                lpips_test_vgg /= len(config["cameras"])
                lpips_test_alex_motion /= len(config["cameras"])
                lpips_test_vgg_motion /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                if wandb_run:
                    wandb_run.log(
                        {
                            f"{config['name']}/loss_viewpoint - l1_loss": l1_test,
                            f"{config['name']}/loss_viewpoint - psnr": psnr_test,
                            f"{config['name']}/loss_viewpoint - ssim": ssim_test,
                            f"{config['name']}/loss_viewpoint_motion - l1_loss": l1_test_motion,
                            f"{config['name']}/loss_viewpoint_motion - psnr": psnr_test_motion,
                            f"{config['name']}/loss_viewpoint_motion - ssim": ssim_test_motion,
                            f"{config['name']}/loss_viewpoint - lpips_alex": lpips_test_alex,
                            f"{config['name']}/loss_viewpoint - lpips_vgg": lpips_test_vgg,
                            f"{config['name']}/loss_viewpoint_motion - lpips_alex": lpips_test_alex_motion,
                            f"{config['name']}/loss_viewpoint_motion - lpips_vgg": lpips_test_vgg_motion,
                        },
                        step=iteration,
                    )

        if wandb_run:
            wandb_run.log(
                {
                    "scene/opacity_histogram": wandb.Histogram(
                        scene.gaussians.get_opacity.cpu().detach().numpy()
                    ),
                    "total_points": scene.gaussians.get_xyz.shape[0],
                },
                step=iteration,
            )
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
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment for logging purposes (e.g. in WandB)",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="WandB group name to group multiple runs together",
    )
    parser.add_argument(
        "--recovering",
        action="store_true",
        help="Whether we are fitting from a frame to another, ie fine tune.",
    )
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    wandb_run = wandb.init(
        project="experiment-1-final",
        name=args.exp_name,
        group=args.wandb_group,
        config=args,
        save_code=True,
        resume=False,
    )
    wandb_run.log_code("./scene")
    try:
        training(
            lp.extract(args),
            op.extract(args),
            pp.extract(args),
            args.test_iterations,
            args.save_iterations,
            args.checkpoint_iterations,
            args.start_checkpoint,
            args.debug_from,
            args.recovering,
            wandb_run=wandb_run,
        )
    except Exception as e:
        print("Error during training: ", e)
        traceback.print_exc()
        wandb.finish()
        raise e
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        wandb.finish()
    # All done
    finally:
        # print("\nTraining complete.")
        if wandb_run:
            wandb.finish()

    # All done
    print("\nTraining complete.")
