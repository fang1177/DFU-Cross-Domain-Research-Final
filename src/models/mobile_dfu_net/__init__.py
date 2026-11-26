"""
MobileDFU-Net package initialization.
Exposes master model architecture, building blocks, and compound loss functions.
"""

from src.models.mobile_dfu_net.config import (
    MobileDFUNetConfig,
    MobileDFULossConfig,
    DCSStemConfig,
    EncoderConfig,
    RFBConfig,
    DecoderConfig,
    HeadConfig
)
from src.models.mobile_dfu_net.blocks import (
    ConvBNAct,
    InvertedResidualBlock,
    SqueezeAndExcitation,
    ChannelAttentionGate
)
from src.models.mobile_dfu_net.dcs_stream import DCSStem
from src.models.mobile_dfu_net.backbone import MobileDFUEncoder, ReceptiveFieldBlock
from src.models.mobile_dfu_net.feature_fusion import MobileDFUDecoder, DecoderBlock
from src.models.mobile_dfu_net.boundary_head import BoundaryHead
from src.models.mobile_dfu_net.segmentation_head import SegmentationHead
from src.models.mobile_dfu_net.model import MobileDFUNet
from src.models.mobile_dfu_net.losses import (
    SmoothDiceLoss,
    WeightedBCELoss,
    ToleranceBoundedBoundaryLoss,
    AuxiliaryBoundaryLoss,
    MobileDFUCompoundLoss
)
from src.models.mobile_dfu_net.profile import (
    profile_mobile_dfu_net,
    count_parameters,
    compute_model_size_mb
)

__all__ = [
    "MobileDFUNet",
    "MobileDFUNetConfig",
    "MobileDFULossConfig",
    "DCSStemConfig",
    "EncoderConfig",
    "RFBConfig",
    "DecoderConfig",
    "HeadConfig",
    "ConvBNAct",
    "InvertedResidualBlock",
    "SqueezeAndExcitation",
    "ChannelAttentionGate",
    "DCSStem",
    "MobileDFUEncoder",
    "ReceptiveFieldBlock",
    "MobileDFUDecoder",
    "DecoderBlock",
    "BoundaryHead",
    "SegmentationHead",
    "SmoothDiceLoss",
    "WeightedBCELoss",
    "ToleranceBoundedBoundaryLoss",
    "AuxiliaryBoundaryLoss",
    "MobileDFUCompoundLoss",
    "profile_mobile_dfu_net",
    "count_parameters",
    "compute_model_size_mb"
]

