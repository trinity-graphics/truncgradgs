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

import torch
import math
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import os
IS_BASELINE = os.getenv("IS_BASELINE", "0") == "1"
# if IS_BASELINE:
#     from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# else:
#     from diff_gaussian_rasterization_trunc import GaussianRasterizationSettings, GaussianRasterizer

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization_trunc import  GaussianRasterizer as TruncGaussianRasterizer
from diff_gaussian_rasterization_trunc import  GaussianRasterizationSettings as TruncGaussianRasteriationSettings

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor,
           scaling_modifier = 1.0, separate_sh = False, override_color = None,
           use_trained_exp=False, mu_grad_pull_strength=0.0, exploration_on=False,
           mask=None, opacity_ovrd=None, use_trunc=False, motion_mask=None,
           scene_extent=0.0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings_kwargs = {
        "image_height": int(viewpoint_camera.image_height),
        "image_width": int(viewpoint_camera.image_width),
        "tanfovx": tanfovx,
        "tanfovy": tanfovy,
        "bg": bg_color,
        "scale_modifier": scaling_modifier,
        "viewmatrix": viewpoint_camera.world_view_transform,
        "projmatrix": viewpoint_camera.full_proj_transform,
        "sh_degree": pc.active_sh_degree,
        "campos": viewpoint_camera.camera_center,
        "prefiltered": False,
        "debug": pipe.debug,
    }
    IS_BASELINE = not use_trunc
    if not IS_BASELINE:
        raster_settings_kwargs["pull_strength"] = mu_grad_pull_strength
        raster_settings_kwargs["explore"] = exploration_on
        raster_settings_kwargs["scene_extent"] = scene_extent

    raster_settings = GaussianRasterizationSettings(**raster_settings_kwargs) if not use_trunc else TruncGaussianRasteriationSettings(**raster_settings_kwargs)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings) if not use_trunc else TruncGaussianRasterizer(raster_settings=raster_settings)
    # raster_settings = GaussianRasterizationSettings(**raster_settings_kwargs)
    # rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if mask is None:
        mask = torch.ones(pc.get_xyz.shape[0], dtype=torch.bool, device=pc.get_xyz.device)
    means3D = pc.get_xyz[mask]
    means2D = screenspace_points[mask]
    opacity = pc.get_opacity[mask]
    if opacity_ovrd is not None:
        opacity = opacity_ovrd * torch.ones_like(opacity)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)[mask]
    else:
        scales = pc.get_scaling[mask]
        rotations = pc.get_rotation[mask]

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features[mask].transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz[mask] - viewpoint_camera.camera_center.repeat(pc.get_features[mask].shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc[mask], pc.get_features_rest[mask]
            else:
                shs = pc.get_features[mask]
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    kwargs = {}
    if not IS_BASELINE:
        kwargs = {"motion_mask":motion_mask}
    if separate_sh:
        out = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp, **kwargs)
        if IS_BASELINE:
            rendered_image, radii, depth_image = out
        else:
            rendered_image, radii, depth_image, synthetic_grad, dL_dalpha, true_grads_pp, trunc_grads_pp, true_grad_mags_pp, trunc_grad_mags_pp, dL_dalpha_neg_mask = out
    else:
        out = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp, **kwargs)
        if IS_BASELINE:
            rendered_image, radii = out
        else:
            rendered_image, radii, synthetic_grad, dL_dalpha, true_grads_pp, trunc_grads_pp, true_grad_mags_pp, trunc_grad_mags_pp, dL_dalpha_neg_mask = out
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        # "depth" : depth_image
    }
    if not IS_BASELINE:
        out["synthetic_grad"] = synthetic_grad
        out["dL_dalpha"] = dL_dalpha
        out["true_grads_pp"] = true_grads_pp
        out["trunc_grads_pp"] = trunc_grads_pp
        out["true_grad_mags_pp"] = true_grad_mags_pp
        out["trunc_grad_mags_pp"] = trunc_grad_mags_pp
        out["dL_dalpha_neg_mask"] = dL_dalpha_neg_mask

    
    return out
