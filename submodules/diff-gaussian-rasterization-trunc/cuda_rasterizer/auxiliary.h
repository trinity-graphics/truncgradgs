/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE / 32)
#define RADIUS_PADDING 96
#define TRUNCATED_OPACITY_THRESHOLD 0.01f

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {1.0925484305920792f, -1.0925484305920792f,
                                  0.31539156525252005f, -1.0925484305920792f,
                                  0.5462742152960396f};
__device__ const float SH_C3[] = {-0.5900435899266435f, 2.890611442640554f,
                                  -0.4570457994644658f, 0.3731763325901154f,
                                  -0.4570457994644658f, 1.445305721320277f,
                                  -0.5900435899266435f};

__forceinline__ __device__ float ndc2Pix(float v, int S) {
  return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius,
                                        uint2 &rect_min, uint2 &rect_max,
                                        dim3 grid) {
  rect_min = {min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
              min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))};
  rect_max = {
      min(grid.x,
          max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
      min(grid.y,
          max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))};
}

__forceinline__ __device__ float3 transformPoint4x3(const float3 &p,
                                                    const float *matrix) {
  float3 transformed = {
      matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
      matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
      matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
  };
  return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3 &p,
                                                    const float *matrix) {
  float4 transformed = {
      matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
      matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
      matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
      matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]};
  return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3 &p,
                                                  const float *matrix) {
  float3 transformed = {
      matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
      matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
      matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
  };
  return transformed;
}

__forceinline__ __device__ float3
transformVec4x3Transpose(const float3 &p, const float *matrix) {
  float3 transformed = {
      matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
      matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
      matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
  };
  return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv) {
  float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
  float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
  float dnormvdz =
      (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) *
      invsum32;
  return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv) {
  float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
  float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

  float3 dnormvdv;
  dnormvdv.x =
      ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) *
      invsum32;
  dnormvdv.y =
      (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) *
      invsum32;
  dnormvdv.z =
      (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) *
      invsum32;
  return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv) {
  float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

  float4 vdv = {v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w};
  float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
  float4 dnormvdv;
  dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
  dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
  dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
  dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
  return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_frustum(int idx, const float *orig_points,
                                           const float *viewmatrix,
                                           const float *projmatrix,
                                           bool prefiltered, float3 &p_view) {
  float3 p_orig = {orig_points[3 * idx], orig_points[3 * idx + 1],
                   orig_points[3 * idx + 2]};

  // Bring points to screen space
  float4 p_hom = transformPoint4x4(p_orig, projmatrix);
  float p_w = 1.0f / (p_hom.w + 0.0000001f);
  float3 p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};
  p_view = transformPoint4x3(p_orig, viewmatrix);

  if (p_view.z <= 0.2f) // || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y <
                        // -1.3 || p_proj.y > 1.3)))
  {
    if (prefiltered) {
      printf("Point is filtered although prefiltered is set. This shouldn't "
             "happen!");
      __trap();
    }
    return false;
  }
  return true;
}

#define CHECK_CUDA(A, debug)                                                   \
  A;                                                                           \
  if (debug) {                                                                 \
    auto ret = cudaDeviceSynchronize();                                        \
    if (ret != cudaSuccess) {                                                  \
      std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__   \
                << ": " << cudaGetErrorString(ret);                            \
      throw std::runtime_error(cudaGetErrorString(ret));                       \
    }                                                                          \
  }

__forceinline__ __device__ void compute_gaussian_derivative_at_boundary(
    float2 d, float4 con_o, float ellipse_level_thresh,
    float mahalanobis_dist_squared, float *gdx_hit, float *gdy_hit,
    float *mahalanobis_dist_squared_hit) {
  const float q = sqrtf(d.x * d.x + d.y * d.y);
  const float inv_q = 1.0f / (q + 0.0000001f);

  // Guard against any negative values due to floating point inaccuracies
  const float safe_ratio = fmaxf(
      0.0f, ellipse_level_thresh / fmaxf(mahalanobis_dist_squared, 1e-6f));
  const float sqrt_term = sqrtf(safe_ratio);

  const float s0 = q * (1.0f - sqrt_term);
  const float s1 = q * (1.0f + sqrt_term);
  const float s = min(s0, s1);
  const float dx_hit = d.x - s * d.x * inv_q;
  const float dy_hit = d.y - s * d.y * inv_q;
  *mahalanobis_dist_squared_hit = con_o.x * dx_hit * dx_hit +
                                  con_o.z * dy_hit * dy_hit +
                                  2.0f * con_o.y * dx_hit * dy_hit;
  const float G_hit = expf(-0.5f * (*mahalanobis_dist_squared_hit));
  *gdx_hit = G_hit * dx_hit;
  *gdy_hit = G_hit * dy_hit;
}

__forceinline__ __device__ float
truncated_derivative_x(float dG_ddelx, float2 d, float4 con_o,
                       float cutoff_slope, float ellipse_level_thresh,
                       float mahalanobis_dist_squared) {
  float gdx_hit, gdy_hit, mahalanobis_dist_squared_hit;
  compute_gaussian_derivative_at_boundary(
      d, con_o, ellipse_level_thresh, mahalanobis_dist_squared, &gdx_hit,
      &gdy_hit, &mahalanobis_dist_squared_hit);
  const float grad_at_boundary = -gdx_hit * con_o.x - gdy_hit * con_o.y;

  if (grad_at_boundary < 0.0f) {
    const float bias = grad_at_boundary - cutoff_slope * ellipse_level_thresh;
    float ret = min(cutoff_slope * mahalanobis_dist_squared + bias, dG_ddelx);
    return ret;
  } else {
    const float bias = grad_at_boundary + cutoff_slope * ellipse_level_thresh;
    float ret = max(-cutoff_slope * mahalanobis_dist_squared + bias, dG_ddelx);
    return ret;
  }
}

__forceinline__ __device__ float
truncated_derivative_y(float dG_ddely, float2 d, float4 con_o,
                       float cutoff_slope, float ellipse_level_thresh,
                       float mahalanobis_dist_squared) {
  float gdx_hit, gdy_hit, mahalanobis_dist_squared_hit;
  compute_gaussian_derivative_at_boundary(
      d, con_o, ellipse_level_thresh, mahalanobis_dist_squared, &gdx_hit,
      &gdy_hit, &mahalanobis_dist_squared_hit);
  const float grad_at_boundary = -gdy_hit * con_o.z - gdx_hit * con_o.y;

  if (grad_at_boundary < 0.0f) {
    const float bias = grad_at_boundary - cutoff_slope * ellipse_level_thresh;
    float ret = min(cutoff_slope * mahalanobis_dist_squared + bias, dG_ddely);
    return ret;
  } else {
    const float bias = grad_at_boundary + cutoff_slope * ellipse_level_thresh;
    float ret = max(-cutoff_slope * mahalanobis_dist_squared + bias, dG_ddely);
    return ret;
  }
}

__forceinline__ __device__ float truncated_derivative_x_interpolated(
    float dG_ddelx, float2 d, float4 con_o, float ellipse_level_thresh,
    float mahalanobis_dist_squared, float radius) {
  const float q = sqrtf(d.x * d.x + d.y * d.y);
  if (q > radius) {
    return 0.0f;
  }
  float gdx_hit, gdy_hit, mahalanobis_dist_squared_hit;
  compute_gaussian_derivative_at_boundary(
      d, con_o, ellipse_level_thresh, mahalanobis_dist_squared, &gdx_hit,
      &gdy_hit, &mahalanobis_dist_squared_hit);
  const float grad_at_boundary = -gdx_hit * con_o.x - gdy_hit * con_o.y;
  const float inv_q = 1.0f / (q + 0.0000001f);
  const float dx_radius = d.x + abs(q - radius) * d.x * inv_q;
  const float dy_radius = d.y + abs(q - radius) * d.y * inv_q;

  const float mahalanobis_dist_squared_radius =
      con_o.x * dx_radius * dx_radius + con_o.z * dy_radius * dy_radius +
      2.0f * con_o.y * dx_radius * dy_radius;
  const float G_radius = expf(-0.5f * mahalanobis_dist_squared_hit);
  const float gdx = G_radius * dx_radius;
  const float gdy = G_radius * dy_radius;
  const float grad_at_radius = -gdx * con_o.x - gdy * con_o.y;
  const float x0 = mahalanobis_dist_squared_radius;
  const float y0 = grad_at_radius;
  const float x1 = mahalanobis_dist_squared_hit;
  const float y1 = grad_at_boundary;
  return (y0 * (x1 - mahalanobis_dist_squared) +
          y1 * (mahalanobis_dist_squared - x0)) /
         (x1 - x0 + 0.0000001f);
}

__forceinline__ __device__ float truncated_derivative_y_interpolated(
    float dG_ddely, float2 d, float4 con_o, float ellipse_level_thresh,
    float mahalanobis_dist_squared, float radius) {
  const float q = sqrtf(d.x * d.x + d.y * d.y);
  if (q > radius) {
    return 0.0f;
  }
  float gdx_hit, gdy_hit, mahalanobis_dist_squared_hit;
  compute_gaussian_derivative_at_boundary(
      d, con_o, ellipse_level_thresh, mahalanobis_dist_squared, &gdx_hit,
      &gdy_hit, &mahalanobis_dist_squared_hit);
  const float grad_at_boundary = -gdy_hit * con_o.z - gdx_hit * con_o.y;
  const float inv_q = 1.0f / (q + 0.0000001f);
  const float dx_radius = d.x + abs(q - radius) * d.x * inv_q;
  const float dy_radius = d.y + abs(q - radius) * d.y * inv_q;

  const float mahalanobis_dist_squared_radius =
      con_o.x * dx_radius * dx_radius + con_o.z * dy_radius * dy_radius +
      2.0f * con_o.y * dx_radius * dy_radius;
  const float G_radius = expf(-0.5f * mahalanobis_dist_squared_hit);
  const float gdx = G_radius * dx_radius;
  const float gdy = G_radius * dy_radius;
  const float grad_at_radius = -gdy * con_o.z - gdx * con_o.y;
  const float x0 = mahalanobis_dist_squared_radius;
  const float y0 = grad_at_radius;
  const float x1 = mahalanobis_dist_squared_hit;
  const float y1 = grad_at_boundary;
  return (y0 * (x1 - mahalanobis_dist_squared) +
          y1 * (mahalanobis_dist_squared - x0)) /
         (x1 - x0 + 0.0000001f);
}

#endif
