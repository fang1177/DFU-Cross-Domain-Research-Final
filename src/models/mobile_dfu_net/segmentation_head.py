"""
Primary Boundary-Gated Segmentation Head for MobileDFU-Net:
Inputs:
- Decoder native features D_1 [B, 16, H, W]
- Predicted boundary probability map [B, 1, H, W] (from BoundaryHead)
Processing:
- Boundary modulation: F_seg = D_1 + Conv1x1(boundary_probs) * D_1
- 3x3 ConvBNAct refinement: F_seg -> [B, 16, H, W]
- 1x1 Conv output projection -> Raw segmentation logits [B, 1, H, W]
Output:
- Primary segmentation logits [B, 1, H, W] (strictly logits, not probabilities)
"""

import torch
import torch.nn as nn

from src.models.mobile_dfu_net.blocks import ConvBNAct
from src.models.mobile_dfu_net.config import HeadConfig


class SegmentationHead(nn.Module):
    """
    Boundary-Gated Primary Segmentation Head.
    Inputs:
      d1: Decoder native feature map [B, 16, H, W]
      boundary_probs: Predicted boundary probability map [B, 1, H, W]
    Output:
      seg_logits: Raw binary segmentation logits [B, 1, H, W]
    """
    def __init__(self, config: HeadConfig = HeadConfig()):
        super().__init__()
        # Boundary projection conv mapping 1 boundary channel to 16 feature channels
        self.boundary_proj = nn.Conv2d(config.boundary_channels, config.in_channels, kernel_size=1, bias=True)
        
        # Lightweight depthwise separable refinement and final output projection
        self.refine = nn.Sequential(
            ConvBNAct(config.in_channels, config.in_channels, kernel_size=3, stride=1, groups=config.in_channels, act_layer="hardswish"),
            ConvBNAct(config.in_channels, config.in_channels, kernel_size=1, stride=1, act_layer="hardswish")
        )
        self.classifier = nn.Conv2d(config.in_channels, config.segmentation_channels, kernel_size=1, bias=True)

    def forward(self, d1: torch.Tensor, boundary_probs: torch.Tensor) -> torch.Tensor:
        # Boundary-gated residual modulation
        # F_seg = D_1 + Conv1x1(Y_hat_boundary) * D_1
        boundary_gate = self.boundary_proj(boundary_probs)
        f_seg = d1 + (boundary_gate * d1)
        
        # Refinement & classification
        f_refine = self.refine(f_seg)
        seg_logits = self.classifier(f_refine)
        return seg_logits
