"""
Lightweight Decoder with Channel-Gated Skip Connections:
Combines:
- Bilinear 2x upsampling.
- Channel Attention Gated Skips: Skip_gated = Skip * Sigmoid(MLP(GAP(Decoder))).
- Depthwise separable convolutions to restore fine spatial details.
Progressions:
- Up4: [Bottleneck: 160] up 2x + [Skip4: 128] -> 96 channels (64x64)
- Up3: [Up4: 96] up 2x + [Skip3: 64] -> 48 channels (128x128)
- Up2: [Up3: 48] up 2x + [Skip2: 32] -> 24 channels (256x256)
- Up1: [Up2: 24] up 2x + [Skip1: 24] -> 16 channels (512x512)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint as _activation_checkpoint

from src.models.mobile_dfu_net.blocks import ConvBNAct, ChannelAttentionGate
from src.models.mobile_dfu_net.config import DecoderConfig


class DecoderBlock(nn.Module):
    """
    Single Decoder Stage:
    1. Bilinear upsample 2x decoder features.
    2. Modulate encoder skip features via ChannelAttentionGate.
    3. Concatenate upsampled decoder + gated skip features.
    4. Lightweight depthwise separable convolution to refine fused representation.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.cag = ChannelAttentionGate(skip_channels=skip_channels, decoder_channels=in_channels)
        fused_channels = in_channels + skip_channels
        
        # Lightweight depthwise separable refinement block
        self.conv = nn.Sequential(
            ConvBNAct(fused_channels, fused_channels, kernel_size=3, stride=1, groups=fused_channels, act_layer="hardswish"),
            ConvBNAct(fused_channels, out_channels, kernel_size=1, stride=1, act_layer="hardswish"),
            ConvBNAct(out_channels, out_channels, kernel_size=3, stride=1, groups=out_channels, act_layer="hardswish"),
            ConvBNAct(out_channels, out_channels, kernel_size=1, stride=1, act_layer="hardswish")
        )

    def forward(self, dec: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Bilinear 2x upsample decoder features
        dec_up = F.interpolate(dec, size=skip.shape[2:], mode="bilinear", align_corners=False)
        # Apply Channel Attention Gate to skip features
        skip_gated = self.cag(skip, dec_up)
        # Concatenate along channel dimension
        fused = torch.cat([dec_up, skip_gated], dim=1)
        return self.conv(fused)


class MobileDFUDecoder(nn.Module):
    """
    Multi-Scale Feature Fusion Decoder:
    Inputs:
      f0:    [B, 24, 512, 512] (Native resolution stem feature map)
      skip1: [B, 24, 256, 256]
      skip2: [B, 32, 128, 128]
      skip3: [B, 64, 64, 64]
      skip4: [B, 128, 32, 32]
      bottleneck: [B, 160, 32, 32]
    Output:
      D_1: [B, 16, 512, 512]
    """
    def __init__(self, config: DecoderConfig = DecoderConfig(), activation_checkpointing: bool = False):
        super().__init__()
        self.config = config
        self.activation_checkpointing = bool(activation_checkpointing)

        # Up4: Bottleneck (160) at 32x32 + Skip4 (128) at 32x32 -> Up4 (96) at 64x64
        self.up4 = DecoderBlock(in_channels=160, skip_channels=128, out_channels=config.stage_channels[0])

        # Up3: Up4 (96) at 64x64 + Skip3 (64) at 64x64 -> Up3 (48) at 128x128
        self.up3 = DecoderBlock(in_channels=config.stage_channels[0], skip_channels=64, out_channels=config.stage_channels[1])

        # Up2: Up3 (48) at 128x128 + Skip2 (32) at 128x128 -> Up2 (24) at 256x256
        self.up2 = DecoderBlock(in_channels=config.stage_channels[1], skip_channels=32, out_channels=config.stage_channels[2])

        # Up1: Up2 (24) at 256x256 + Skip1 (24) at 256x256 -> Up1 (24) at 512x512
        self.up1 = DecoderBlock(in_channels=config.stage_channels[2], skip_channels=24, out_channels=24)

        # Native resolution stem fusion: Up1 (24) at 512x512 + Stem f0 (24) at 512x512 -> D_1 (16) at 512x512
        self.stem_fuse = DecoderBlock(in_channels=24, skip_channels=24, out_channels=config.stage_channels[3])

    def forward(
        self,
        f0: torch.Tensor,
        s1: torch.Tensor,
        s2: torch.Tensor,
        s3: torch.Tensor,
        s4: torch.Tensor,
        bottleneck: torch.Tensor
    ) -> torch.Tensor:
        # bottleneck: [B, 160, 32, 32]
        d4 = self.up4(bottleneck, s4)    # up to 64x64, fuses with s4 [B, 128, 32, 32] -> [B, 96, 64, 64]
        d3 = self.up3(d4, s3)            # up to 128x128, fuses with s3 [B, 64, 64, 64] -> [B, 48, 128, 128]
        if self.activation_checkpointing and self.training:
            d2 = _activation_checkpoint(self.up2, d3, s2, use_reentrant=False)
        else:
            d2 = self.up2(d3, s2)            # up to 256x256, fuses with s2 [B, 32, 128, 128] -> [B, 24, 256, 256]
        if self.activation_checkpointing and self.training:
            d1 = _activation_checkpoint(self.up1, d2, s1, use_reentrant=False)
        else:
            d1 = self.up1(d2, s1)            # up to 512x512, fuses with s1 [B, 24, 256, 256] -> [B, 24, 512, 512]
        if self.activation_checkpointing and self.training:
            d_out = _activation_checkpoint(self.stem_fuse, d1, f0, use_reentrant=False)
        else:
            d_out = self.stem_fuse(d1, f0)   # at 512x512, fuses with f0 [B, 24, 512, 512] -> [B, 16, 512, 512]
        return d_out
