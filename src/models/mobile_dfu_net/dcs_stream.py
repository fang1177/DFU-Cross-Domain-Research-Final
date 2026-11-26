"""
Decoupled Chromatic-Structural Invariant Stem (DCS-Stem):
Separates colour/chromatic information from structural edge/luminance features.

Contract:
- Input is RAW RGB, float32, in [0, 1]. The dataset must NOT apply ImageNet
  normalization; it is performed here, internally and only for colour.
- Luminance and Sobel gradients are computed from the raw RGB input.
- ImageNet mean/std normalization is applied only to the chromatic branch.

Streams:
- Chromatic Stream: ImageNet-normalized RGB -> depthwise separable convolution
  extracting color representations (16 channels).
- Structural Stream: Differentiable Sobel gradient magnitude over raw-RGB
  luminance (8 channels).
- Structure-conditioned sigmoid feature gate (not query-key-value cross-attention):
  F_0 = Concat[F_struct, F_color * Sigmoid(F_struct)] (24 channels).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mobile_dfu_net.blocks import ConvBNAct
from src.models.mobile_dfu_net.config import DCSStemConfig


class DCSStem(nn.Module):
    """
    Decoupled Chromatic-Structural Invariant Stem.
    Inputs: [B, 3, H, W] raw RGB, float32, in [0, 1] (not dataset-normalized).
    Outputs: [B, 24, H, W] fused invariant stem representation F_0.
    """
    def __init__(self, config: DCSStemConfig = DCSStemConfig()):
        super().__init__()
        self.config = config

        # ImageNet normalization statistics for the chromatic branch.
        # Non-persistent: they are derived from config, not learned, and must not
        # appear in the persistent state dictionary.
        image_mean = torch.tensor(config.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        image_std = torch.tensor(config.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean, persistent=False)
        self.register_buffer("image_std", image_std, persistent=False)

        # Fixed luminance weights for RGB -> Greyscale: Y = 0.2989 R + 0.5870 G + 0.1140 B
        # Standard ITU-R BT.601 luminance coefficients
        luminance_kernel = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("luminance_kernel", luminance_kernel)

        # Fixed Sobel gradient filters for structural gradient calculation
        # G_x and G_y
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        # 1. Chromatic Stream: 3x3 Depthwise Separable Conv
        # In: 3 channels -> Depthwise 3x3 -> Pointwise 1x1 -> 16 channels
        self.color_stem = nn.Sequential(
            ConvBNAct(3, 3, kernel_size=3, stride=1, groups=3, act_layer="hardswish"),
            ConvBNAct(3, config.color_channels, kernel_size=1, stride=1, act_layer="hardswish")
        )

        # 2. Structural Stream: 3x3 Conv over Sobel Gradient
        # In: 1 channel (gradient magnitude) -> 8 channels
        self.struct_stem = nn.Sequential(
            ConvBNAct(1, config.struct_channels, kernel_size=3, stride=1, act_layer="hardswish"),
            ConvBNAct(config.struct_channels, config.struct_channels, kernel_size=3, stride=1, act_layer="hardswish")
        )

        # 3. Structure-conditioned sigmoid feature-gate projection
        # (channel-wise gating, not query-key-value cross-attention)
        # Struct stream (8 channels) projects to color channels (16 channels) to gate color features
        self.gate_proj = nn.Sequential(
            nn.Conv2d(config.struct_channels, config.color_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def compute_luminance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Raw-RGB luminance using fixed BT.601 coefficients.
        Input: [B, 3, H, W] raw RGB in [0, 1] -> Output: [B, 1, H, W]
        """
        return F.conv2d(x, self.luminance_kernel)

    def normalize_chromatic(self, x: torch.Tensor) -> torch.Tensor:
        """
        ImageNet mean/std normalization, applied to the chromatic branch only.
        Input: [B, 3, H, W] raw RGB in [0, 1] -> Output: [B, 3, H, W]
        """
        return (x - self.image_mean) / self.image_std

    def extract_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Differentiable Sobel gradient computed from RAW RGB (no normalization).
        Input: [B, 3, H, W] raw RGB in [0, 1] -> Output: [B, 1, H, W]
        """
        # Luminance conversion on raw RGB
        lum = self.compute_luminance(x)
        
        # Sobel filtering (padding=1 to preserve dimensions)
        gx = F.conv2d(lum, self.sobel_x, padding=1)
        gy = F.conv2d(lum, self.sobel_y, padding=1)
        
        # Gradient magnitude with eps for gradient stability at 0
        grad = torch.sqrt(gx * gx + gy * gy + 1e-8)
        return grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Structural gradient from RAW RGB: [B, 1, H, W]
        grad = self.extract_gradient(x)

        # ImageNet normalization for the chromatic branch only: [B, 3, H, W]
        x_norm = self.normalize_chromatic(x)

        # Extract chromatic features from normalized RGB: [B, 16, H, W]
        f_color = self.color_stem(x_norm)

        # Extract structural features from raw-RGB gradients: [B, 8, H, W]
        f_struct = self.struct_stem(grad)

        # Cross-attention gating: color features modulated by structural gate
        # Structure modulates color to suppress non-structural color shifts
        struct_gate = self.gate_proj(f_struct)
        f_color_gated = f_color * struct_gate

        # Concatenation: [B, 8 + 16 = 24, H, W]
        f_0 = torch.cat([f_struct, f_color_gated], dim=1)
        return f_0

