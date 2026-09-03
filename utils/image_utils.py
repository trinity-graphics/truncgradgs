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
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def easy_cmap(x: torch.Tensor):
    x_rgb = torch.zeros(
        (3, x.shape[0], x.shape[1]), dtype=torch.float32, device=x.device
    )
    x_max, x_min = x.max(), x.min()
    x_normalize = (x - x_min) / (x_max - x_min)
    x_rgb[0] = torch.clamp(x_normalize, 0, 1)
    x_rgb[1] = torch.clamp(x_normalize, 0, 1)
    x_rgb[2] = torch.clamp(x_normalize, 0, 1)
    return x_rgb


def draw_text(img: Image.Image, text: str, color: str, size: int):
    font_path = "./HackNerdFont-Bold.ttf"
    font = ImageFont.truetype(font_path, size)
    ImageDraw.Draw(img, "RGB").text((10, 10), text, fill=color, font=font)


def tensor_to_pil_image(x: torch.Tensor, mode="RGB") -> Image.Image:
    return Image.fromarray(
        (x.transpose(0, 2).transpose(0, 1).cpu().numpy() * 255).astype(np.uint8),
        mode=mode,
    )


def pil_image_to_tensor(img: Image.Image) -> torch.Tensor:
    return (
        torch.from_numpy(np.asarray(img.convert("RGB")))
        .transpose(0, 2)
        .transpose(1, 2)
        .float()
        / 255.0
    )
