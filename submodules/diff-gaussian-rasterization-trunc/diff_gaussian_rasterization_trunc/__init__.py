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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
    motion_mask,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        motion_mask,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        motion_mask,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            raster_settings.scene_extent
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        synth_grad3D = torch.zeros((means3D.shape[0], 3), dtype=torch.float32, device=means3D.device)
        dL_dalpha = torch.zeros((means3D.shape[0], 1), dtype=torch.float32, device=means3D.device)
        true_grads_pp = torch.zeros((raster_settings.image_height, raster_settings.image_width), dtype=torch.int32, device=means3D.device)
        trunc_grads_pp = torch.zeros((raster_settings.image_height, raster_settings.image_width), dtype=torch.int32, device=means3D.device)
        true_grad_mags_pp = torch.zeros((raster_settings.image_height, raster_settings.image_width), dtype=torch.float32, device=means3D.device)
        trunc_grad_mags_pp = torch.zeros((raster_settings.image_height, raster_settings.image_width), dtype=torch.float32, device=means3D.device)
        dL_dalpha_neg_mask = torch.zeros((means3D.shape[0],), dtype=torch.bool, device=means3D.device)
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, synth_grad3D, dL_dalpha, true_grads_pp, trunc_grads_pp, true_grad_mags_pp, trunc_grad_mags_pp, dL_dalpha_neg_mask, motion_mask)
        return color, radii, synth_grad3D, dL_dalpha, true_grads_pp, trunc_grads_pp, true_grad_mags_pp, trunc_grad_mags_pp, dL_dalpha_neg_mask

    @staticmethod
    def backward(ctx, grad_out_color, _, _grad_synth, _grad_dL_dalpha, _grad_true_grads_pp, _grad_trunc_grads_pp, _grad_true_grad_mags_pp, _grad_trunc_grad_mags_pp, _grad_dL_dalpha_neg_mask):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, synth_grad3D, dL_dalpha, true_grads_pp, trunc_grads_pp, true_grad_mags_pp, trunc_grad_mags_pp, dL_dalpha_neg_mask, motion_mask = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug, raster_settings.pull_strength,
                raster_settings.explore, synth_grad3D, motion_mask,
                raster_settings.scene_extent)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, dL_dalpha_out, true_grads_pp_out, trunc_grads_pp_out, true_grad_mags_pp_out, trunc_grad_mags_pp_out, dL_dalpha_neg_mask_out = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, dL_dalpha_out, true_grads_pp_out, trunc_grads_pp_out, true_grad_mags_pp_out, trunc_grad_mags_pp_out, dL_dalpha_neg_mask_out = _C.rasterize_gaussians_backward(*args)

        dL_dalpha.copy_(dL_dalpha_out)
        true_grads_pp.copy_(true_grads_pp_out)
        trunc_grads_pp.copy_(trunc_grads_pp_out)
        true_grad_mags_pp.copy_(true_grad_mags_pp_out)
        trunc_grad_mags_pp.copy_(trunc_grad_mags_pp_out)
        dL_dalpha_neg_mask.copy_(dL_dalpha_neg_mask_out.bool())
        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    pull_strength : float
    explore: bool
    scene_extent: float

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None, motion_mask = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
            motion_mask,
        )

