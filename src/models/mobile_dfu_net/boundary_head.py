"""
Boundary Prediction Head for MobileDFU-Net:
Predicts fine boundary probability map from decoder features D_1 [B, 16, H, W].
Architecture:
- 3x3 ConvBNAct (16 -> 8 channels)
- 1x1 Conv (8 -> 1 channel)
Returns raw boundary logits [B, 1, H, W] and boundary probability map [B, 1, H, W].
"""

from typing import Tuple
import torch
import torch.nn as nn

from src.models.mobile_dfu_net.blocks import ConvBNAct
from src.models.mobile_dfu_net.config import HeadConfig


class BoundaryHead(nn.Module):
    """
    Auxiliary Boundary Prediction Head.
    Input: D_1 [B, 16, H, W]
    Output:
      boundary_logits: [B, 1, H, W]
      boundary_probs:  [B, 1, H, W] (Sigmoid activated)
    """
    def __init__(self, config: HeadConfig = HeadConfig()):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNAct(config.in_channels, config.in_channels, kernel_size=3, stride=1, groups=config.in_channels, act_layer="hardswish"),
            ConvBNAct(config.in_channels, 8, kernel_size=1, stride=1, act_layer="hardswish"),
            nn.Conv2d(8, config.boundary_channels, kernel_size=1, bias=True)
        )

    def forward(self, d1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        boundary_logits = self.conv(d1)
        boundary_probs = torch.sigmoid(boundary_logits)
        return boundary_logits, boundary_probs
