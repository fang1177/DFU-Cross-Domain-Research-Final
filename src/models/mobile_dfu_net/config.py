"""
Configuration schemas and hyperparameters for MobileDFU-Net architecture and losses.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DCSStemConfig:
    in_channels: int = 3
    color_channels: int = 16
    struct_channels: int = 8
    out_channels: int = 24  # Concat[struct (8), color (16) * gate] = 24
    # ImageNet RGB statistics, applied internally to the chromatic branch only.
    # The model itself receives raw RGB in [0, 1]; the dataset must not normalize.
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class EncoderConfig:
    # Stages 1 to 4 channel definitions
    stage_channels: Tuple[int, int, int, int] = (24, 32, 64, 128)
    stage_num_blocks: Tuple[int, int, int, int] = (2, 3, 4, 3)
    stage_strides: Tuple[int, int, int, int] = (2, 2, 2, 2)
    expansion_factor: int = 4
    se_reduction_ratio: int = 4
    dropout_p: float = 0.1


@dataclass
class RFBConfig:
    in_channels: int = 128
    out_channels: int = 160
    dilation_rates: Tuple[int, int, int] = (1, 3, 5)


@dataclass
class DecoderConfig:
    # Decoder stages 4 down to 1
    # Inputs: (Bottleneck: 160 -> Up4: 96, Up3: 48, Up2: 24, Up1: 16)
    stage_channels: Tuple[int, int, int, int] = (96, 48, 24, 16)


@dataclass
class HeadConfig:
    in_channels: int = 16
    boundary_channels: int = 1
    segmentation_channels: int = 1


@dataclass
class MobileDFUNetConfig:
    stem: DCSStemConfig = field(default_factory=DCSStemConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    bottleneck: RFBConfig = field(default_factory=RFBConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    input_resolution: Tuple[int, int] = (512, 512)


@dataclass
class MobileDFULossConfig:
    lambda_dice: float = 1.0
    lambda_bce: float = 0.5
    lambda_boundary: float = 0.5
    lambda_aux: float = 0.3
    
    # BCE parameters
    bce_pos_weight: float = 2.0
    aux_bce_pos_weight: float = 2.0
    
    # Boundary-field generation parameters (consumed by
    # compute_tolerance_boundary_fields, not by the boundary-loss forward pass)
    tau: float = 2.0
    max_cap: float = 20.0
    smooth_dice_eps: float = 1e-6

