"""
Fundamental building blocks for MobileDFU-Net:
- ConvBNAct (Standard, Depthwise, Pointwise)
- InvertedResidualBlock (MobileNetV3/MobileNeXt style with SE attention)
- SqueezeAndExcitation (SE channel attention)
- ChannelAttentionGate (CAG for skip connections)
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """
    Convolution + Batch Normalization + Activation (Hard-Swish or ReLU or Identity).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        act_layer: Optional[str] = "hardswish"
    ):
        super().__init__()
        if padding is None:
            padding = ((kernel_size - 1) * dilation) // 2
            
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        
        if act_layer == "hardswish":
            self.act = nn.Hardswish(inplace=False)
        elif act_layer == "relu":
            self.act = nn.ReLU(inplace=False)
        elif act_layer == "sigmoid":
            self.act = nn.Sigmoid()
        elif act_layer is None or act_layer == "identity":
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation layer: {act_layer}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SqueezeAndExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) channel attention module.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 4):
        super().__init__()
        reduced_channels = max(1, in_channels // reduction_ratio)
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=False)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=True)
        self.hardsigmoid = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global Average Pooling: [B, C, H, W] -> [B, C, 1, 1]
        w = F.adaptive_avg_pool2d(x, (1, 1))
        w = self.fc1(w)
        w = self.relu(w)
        w = self.fc2(w)
        w = self.hardsigmoid(w)
        return x * w


class InvertedResidualBlock(nn.Module):
    """
    Inverted Residual Bottleneck Block (MobileNetV3 style):
    1. 1x1 Pointwise expansion conv (expansion_factor t)
    2. 3x3 Depthwise conv with stride s and Hard-Swish
    3. Squeeze-and-Excitation (SE) channel attention
    4. 1x1 Pointwise linear projection
    5. Residual skip connection if stride == 1 and in_channels == out_channels
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion_factor: int = 4,
        se_reduction_ratio: int = 4
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expansion_factor

        layers = []
        # Expansion
        if expansion_factor != 1:
            layers.append(ConvBNAct(in_channels, hidden_dim, kernel_size=1, act_layer="hardswish"))
        
        # Depthwise
        layers.append(
            ConvBNAct(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                groups=hidden_dim,
                act_layer="hardswish"
            )
        )
        
        # Squeeze-and-Excitation
        if se_reduction_ratio is not None and se_reduction_ratio > 0:
            layers.append(SqueezeAndExcitation(hidden_dim, reduction_ratio=se_reduction_ratio))
            
        # Linear projection (No activation)
        layers.append(ConvBNAct(hidden_dim, out_channels, kernel_size=1, act_layer="identity"))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class ChannelAttentionGate(nn.Module):
    """
    Channel Attention Gate (CAG) for skip connections:
    Skip_gated = Skip * Sigmoid(MLP(GAP(Decoder)))
    Filters out background noise from encoder skip connections guided by decoder features.
    """
    def __init__(self, skip_channels: int, decoder_channels: int, reduction_ratio: int = 4):
        super().__init__()
        reduced_channels = max(1, decoder_channels // reduction_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(decoder_channels, reduced_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, skip_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, skip: torch.Tensor, decoder: torch.Tensor) -> torch.Tensor:
        # GAP on decoder: [B, C_dec, H, W] -> [B, C_dec, 1, 1]
        gap_dec = F.adaptive_avg_pool2d(decoder, (1, 1))
        # Compute channel gate: [B, C_skip, 1, 1]
        gate = self.mlp(gap_dec)
        return skip * gate

