"""Lightweight Inverted Residual Encoder and Multi-Scale Receptive Field Block (RFB) Bottleneck:
- Encoder Stages 1 to 4 using InvertedResidualBlocks with Hard-Swish and SE attention.
- Receptive Field Block (RFB) Bottleneck mimicking human receptive field profiles via dilated depthwise convolutions.
"""

from typing import List, Tuple
import torch
import torch.nn as nn

from src.models.mobile_dfu_net.blocks import ConvBNAct, InvertedResidualBlock
from src.models.mobile_dfu_net.config import EncoderConfig, RFBConfig


class ReceptiveFieldBlock(nn.Module):
    """
    Multi-Scale Receptive Field Block (RFB) Bottleneck:
    Operates at deepest resolution (H/16 x W/16 x 128) -> output (H/16 x W/16 x 160).
    Captures multi-scale receptive fields without excessive spatial downsampling.
    Four parallel branches:
      Branch 1: 1x1 conv
      Branch 2: 1x1 conv -> 3x3 dilated conv (dilation rate d=1)
      Branch 3: 1x1 conv -> 3x3 dilated conv (dilation rate d=3)
      Branch 4: 1x1 conv -> 3x3 dilated conv (dilation rate d=5)
    Concatenation + 1x1 projection + residual connection.
    """
    def __init__(self, in_channels: int = 128, out_channels: int = 160, dilation_rates: Tuple[int, int, int] = (1, 3, 5)):
        super().__init__()
        branch_channels = out_channels // 4  # 160 // 4 = 40 channels per branch

        # Branch 1: 1x1 Conv
        self.branch1 = ConvBNAct(in_channels, branch_channels, kernel_size=1, stride=1, act_layer="hardswish")

        # Branch 2: 1x1 Conv -> 3x3 Conv (rate=d1)
        self.branch2 = nn.Sequential(
            ConvBNAct(in_channels, branch_channels, kernel_size=1, stride=1, act_layer="hardswish"),
            ConvBNAct(branch_channels, branch_channels, kernel_size=3, stride=1, dilation=dilation_rates[0], act_layer="hardswish")
        )

        # Branch 3: 1x1 Conv -> 3x3 Conv (rate=d2)
        self.branch3 = nn.Sequential(
            ConvBNAct(in_channels, branch_channels, kernel_size=1, stride=1, act_layer="hardswish"),
            ConvBNAct(branch_channels, branch_channels, kernel_size=3, stride=1, dilation=dilation_rates[1], act_layer="hardswish")
        )

        # Branch 4: 1x1 Conv -> 3x3 Conv (rate=d3)
        self.branch4 = nn.Sequential(
            ConvBNAct(in_channels, branch_channels, kernel_size=1, stride=1, act_layer="hardswish"),
            ConvBNAct(branch_channels, branch_channels, kernel_size=3, stride=1, dilation=dilation_rates[2], act_layer="hardswish")
        )

        # 1x1 Fusion projection
        self.proj = ConvBNAct(branch_channels * 4, out_channels, kernel_size=1, stride=1, act_layer="identity")
        
        # Residual projection if in_channels != out_channels
        self.res_conv = ConvBNAct(in_channels, out_channels, kernel_size=1, stride=1, act_layer="identity") if in_channels != out_channels else nn.Identity()
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        concat = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.proj(concat)
        res = self.res_conv(x)
        return self.act(out + res)


class MobileDFUEncoder(nn.Module):
    """
    Inverted Residual Encoder Stages (Stages 1 to 4):
    Input: F_0 from DCS-Stem [B, 24, H, W]
    Returns:
      skip1: [B, 24, H/2, W/2]
      skip2: [B, 32, H/4, W/4]
      skip3: [B, 64, H/8, W/8]
      skip4: [B, 128, H/16, W/16]
      bottleneck: [B, 160, H/16, W/16]
    """
    def __init__(self, encoder_config: EncoderConfig = EncoderConfig(), rfb_config: RFBConfig = RFBConfig(), activation_checkpointing: bool = False):
        super().__init__()
        self.config = encoder_config
        self.activation_checkpointing = bool(activation_checkpointing)
        
        # Stage 1: Down /2 -> 24 channels (2 blocks)
        self.stage1 = self._make_stage(
            in_channels=24,
            out_channels=encoder_config.stage_channels[0],
            stride=encoder_config.stage_strides[0],
            num_blocks=encoder_config.stage_num_blocks[0],
            expansion_factor=encoder_config.expansion_factor,
            se_ratio=encoder_config.se_reduction_ratio
        )
        
        # Dropout after stage1 (spatial dropout p=0.1 from encoder config)
        self.dropout1 = nn.Dropout2d(p=encoder_config.dropout_p)

        # Stage 2: Down /2 -> 32 channels (3 blocks)
        self.stage2 = self._make_stage(
            in_channels=encoder_config.stage_channels[0],
            out_channels=encoder_config.stage_channels[1],
            stride=encoder_config.stage_strides[1],
            num_blocks=encoder_config.stage_num_blocks[1],
            expansion_factor=encoder_config.expansion_factor,
            se_ratio=encoder_config.se_reduction_ratio
        )
        
        # Dropout after stage2
        self.dropout2 = nn.Dropout2d(p=encoder_config.dropout_p)

        # Stage 3: Down /2 -> 64 channels (4 blocks)
        self.stage3 = self._make_stage(
            in_channels=encoder_config.stage_channels[1],
            out_channels=encoder_config.stage_channels[2],
            stride=encoder_config.stage_strides[2],
            num_blocks=encoder_config.stage_num_blocks[2],
            expansion_factor=encoder_config.expansion_factor,
            se_ratio=encoder_config.se_reduction_ratio
        )
        
        # Dropout after stage3
        self.dropout3 = nn.Dropout2d(p=encoder_config.dropout_p)

        # Stage 4: Down /2 -> 128 channels (3 blocks)
        self.stage4 = self._make_stage(
            in_channels=encoder_config.stage_channels[2],
            out_channels=encoder_config.stage_channels[3],
            stride=encoder_config.stage_strides[3],
            num_blocks=encoder_config.stage_num_blocks[3],
            expansion_factor=encoder_config.expansion_factor,
            se_ratio=encoder_config.se_reduction_ratio
        )
        
        # Dropout after stage4
        self.dropout4 = nn.Dropout2d(p=encoder_config.dropout_p)

        # Receptive Field Block Bottleneck (128 -> 160 channels)
        self.rfb = ReceptiveFieldBlock(
            in_channels=encoder_config.stage_channels[3],
            out_channels=rfb_config.out_channels,
            dilation_rates=rfb_config.dilation_rates
        )

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        num_blocks: int,
        expansion_factor: int,
        se_ratio: int
    ) -> nn.Sequential:
        layers = []
        # First block handles stride downsampling
        layers.append(
            InvertedResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expansion_factor=expansion_factor,
                se_reduction_ratio=se_ratio
            )
        )
        # Subsequent blocks maintain stride 1
        for _ in range(1, num_blocks):
            layers.append(
                InvertedResidualBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1,
                    expansion_factor=expansion_factor,
                    se_reduction_ratio=se_ratio
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, 24, 512, 512]
        
        from torch.utils.checkpoint import checkpoint as _activation_checkpoint

        # Stage 1 + dropout
        if self.activation_checkpointing and self.training:
            s1 = _activation_checkpoint(self.stage1, x, use_reentrant=False)
        else:
            s1 = self.stage1(x)
        s1 = self.dropout1(s1)

        # Stage 2 + dropout
        if self.activation_checkpointing and self.training:
            s2 = _activation_checkpoint(self.stage2, s1, use_reentrant=False)
        else:
            s2 = self.stage2(s1)
        s2 = self.dropout2(s2)

        # Stage 3 + dropout
        s3 = self.stage3(s2)
        s3 = self.dropout3(s3)

        # Stage 4 + dropout
        s4 = self.stage4(s3)
        s4 = self.dropout4(s4)
        
        # Receptive Field Block Bottleneck (128 -> 160 channels)
        bottleneck = self.rfb(s4) # [B, 160, 32, 32]
        
        return s1, s2, s3, s4, bottleneck