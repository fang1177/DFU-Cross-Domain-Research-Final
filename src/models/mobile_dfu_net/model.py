"""
MobileDFU-Net: Lightweight Cross-Domain Diabetic Foot Ulcer Segmentation Architecture.
Combines:
1. Decoupled Chromatic-Structural Invariant Stem (DCS-Stem)
2. Inverted Residual Encoder Stages (Stages 1-4)
3. Multi-Scale Receptive Field Block (RFB) Bottleneck
4. Lightweight Decoder with Channel Attention Gated Skips (Stages 4-1)
5. Auxiliary Boundary Prediction Head
6. Boundary-Gated Primary Segmentation Head

Input:
  x: [B, 3, 512, 512] raw RGB, float32, in [0, 1].
     The dataset must NOT apply ImageNet normalization; DCS-Stem performs it
     internally on the chromatic branch only. Luminance/Sobel structure is
     derived from the raw RGB input.
Outputs:
  Dict with keys:
    "seg_logits": [B, 1, 512, 512] (Raw primary segmentation logits)
    "boundary_logits": [B, 1, 512, 512] (Raw boundary logits)
    "boundary_probs": [B, 1, 512, 512] (Boundary probability map)
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from src.models.mobile_dfu_net.config import MobileDFUNetConfig
from src.models.mobile_dfu_net.blocks import ConvBNAct
from src.models.mobile_dfu_net.dcs_stream import DCSStem
from src.models.mobile_dfu_net.backbone import MobileDFUEncoder
from src.models.mobile_dfu_net.feature_fusion import MobileDFUDecoder
from src.models.mobile_dfu_net.boundary_head import BoundaryHead
from src.models.mobile_dfu_net.segmentation_head import SegmentationHead

from torch.utils.checkpoint import checkpoint as _activation_checkpoint


class MobileDFUNet(nn.Module):
    """
    MobileDFU-Net Master Model Architecture.
    """
    def __init__(self, config: Optional[MobileDFUNetConfig] = None, activation_checkpointing: bool = False, colour_ablation: bool = False, boundary_ablation: bool = False):
        super().__init__()
        self.config = config or MobileDFUNetConfig()
        self.activation_checkpointing = bool(activation_checkpointing)
        self.colour_ablation = bool(colour_ablation)
        self.boundary_ablation = bool(boundary_ablation)

        # 1. Stem: Decoupled Chromatic-Structural Invariant Stem (or simple stem for colour ablation)
        if self.colour_ablation:
            self.stem = nn.Sequential(
                ConvBNAct(3, 24, kernel_size=3, stride=1, act_layer="hardswish"),
            )
        else:
            self.stem = DCSStem(self.config.stem)

        # 2. Encoder: Inverted Residual Stages + RFB Bottleneck
        self.encoder = MobileDFUEncoder(
            self.config.encoder, self.config.bottleneck, activation_checkpointing=self.activation_checkpointing
        )

        # 3. Decoder: Multi-Scale Feature Fusion with Channel Gated Skips
        self.decoder = MobileDFUDecoder(
            self.config.decoder, activation_checkpointing=self.activation_checkpointing
        )

        # 4. Auxiliary Boundary Head
        self.boundary_head = BoundaryHead(self.config.head)

        # 5. Primary Boundary-Gated Segmentation Head
        self.seg_head = SegmentationHead(self.config.head)

        self._init_weights()

    def _init_weights(self):
        """Kaiming normal initialization for conv layers, constant for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_auxiliary: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.
        Args:
          x: [B, 3, 512, 512] raw RGB float32 input in [0, 1]. Must not be
             ImageNet-normalized by the dataset; DCS-Stem normalizes the
             chromatic branch internally.
          return_auxiliary: If True, returns dict with "seg_logits", "boundary_logits", "boundary_probs".
                            If False (inference mode), returns strictly "seg_logits" tensor [B, 1, 512, 512].
        """
        # 1. DCS-Stem: [B, 3, H, W] -> F_0 [B, 24, H, W]
        f0 = self.stem(x)

        # 2. Encoder stages
        s1, s2, s3, s4, bottleneck = self.encoder(f0)

        # 3. Decoder fusion -> D_1: [B, 16, H, W]
        d1 = self.decoder(f0, s1, s2, s3, s4, bottleneck)

        # 4. Boundary Head
        if self.boundary_ablation:
            boundary_logits = torch.zeros_like(d1[:, :1, :, :])
            boundary_probs = torch.zeros_like(d1[:, :1, :, :])
        else:
            boundary_logits, boundary_probs = self.boundary_head(d1)

        # 5. Segmentation Head (Boundary-Gated)
        seg_logits = self.seg_head(d1, boundary_probs)

        if not return_auxiliary:
            return seg_logits

        return {
            "seg_logits": seg_logits,
            "boundary_logits": boundary_logits,
            "boundary_probs": boundary_probs
        }
