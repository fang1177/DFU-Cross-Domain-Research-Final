"""
Data models and manifest schemas for multi-mask representation and provenance tracking.
Ensures one image with multiple masks is represented with full provenance without inflating image counts.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

@dataclass
class ImageRecord:
    image_id: str
    image_sha256: str
    dataset_source: str  # AZH, WSEG, MENDELEY
    original_image_path: str
    extracted_image_path: str
    original_width: int
    original_height: int
    channels: int
    format: str

@dataclass
class MaskRecord:
    mask_id: str
    mask_sha256: str
    original_mask_path: str
    extracted_mask_path: str
    mask_width: int
    mask_height: int
    mask_type: str  # STRICT_BINARY, JPEG_ARTEFACT, CONTINUOUS_GRAYSCALE, ALL_ZERO
    is_empty: bool
    foreground_pct_at_128: Optional[float]

@dataclass
class ImageMaskAssociation:
    association_id: str
    image_id: str
    image_sha256: str
    mask_id: str
    mask_sha256: str
    dataset_source: str
    cohort_assignment: str  # TRAIN, VAL, TARGET_NATIVE, TARGET_MISMATCH_DIAGNOSTIC, CONTROLS_HEALTHY
    cluster_id: str         # Groups multiple masks sharing the same image_sha256
    cluster_size: int       # Number of masks in this cluster
    is_primary_reference: bool  # Lowest alphanumeric pair ID convention
    resolution_status: str  # NATIVE_MATCH (1:1) or MISMATCHED_331x224
    dimensions_match: bool
    inclusion_status: str   # RETAINED, QUARANTINED, EXCLUDED_LEAKAGE
    exclusion_reason: Optional[str] = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

