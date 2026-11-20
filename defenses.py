"""
Defense Mechanisms for Adversarial Robustness.
Implements input sanitization / preprocessing techniques:
1. Spatial Smoothing / Gaussian Blur
2. Median Filtering
3. JPEG Compression Simulation (Quantization / Quality Reduction)
4. Bit-depth Reduction
"""

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
import io
import numpy as np


def apply_gaussian_blur(image_tensor: torch.Tensor, kernel_size: int = 5, sigma: float = 1.5) -> torch.Tensor:
    """
    Apply Gaussian Blur defense to remove high-frequency adversarial noise.
    image_tensor: shape (1, 3, H, W) with values in [0, 1]
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    # PyTorch torchvision transform functional gaussian blur
    return TF.gaussian_blur(image_tensor, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])


def apply_median_filter(image_tensor: torch.Tensor, size: int = 3) -> torch.Tensor:
    """
    Apply Median Filter defense using PIL.
    """
    img_np = (image_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    filtered = pil_img.filter(ImageFilter.MedianFilter(size=size))
    
    filtered_np = np.array(filtered).astype(np.float32) / 255.0
    return torch.from_numpy(filtered_np).permute(2, 0, 1).unsqueeze(0).to(image_tensor.device)


def apply_jpeg_compression(image_tensor: torch.Tensor, quality: int = 40) -> torch.Tensor:
    """
    Simulate JPEG lossy compression to discard high-frequency adversarial perturbations.
    """
    img_np = (image_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)

    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed_img = Image.open(buffer)

    compressed_np = np.array(compressed_img).astype(np.float32) / 255.0
    return torch.from_numpy(compressed_np).permute(2, 0, 1).unsqueeze(0).to(image_tensor.device)


def apply_bit_depth_reduction(image_tensor: torch.Tensor, bits: int = 3) -> torch.Tensor:
    """
    Quantize pixel values to fewer bits (e.g. 3 or 4 bits instead of 8 bits).
    """
    levels = 2 ** bits
    quantized = torch.round(image_tensor * (levels - 1)) / (levels - 1)
    return torch.clamp(quantized, 0.0, 1.0)
