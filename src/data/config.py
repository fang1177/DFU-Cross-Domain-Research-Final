"""
Configuration management classes and schemas for DFU Cross-Domain preprocessing pipeline.
Establishes configuration contracts without executing transformations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple

@dataclass
class PathConfig:
    project_root: Path = Path("E:/DFU_Cross_Domain_Research")
    raw_azh_archive: Path = Path("data/raw/AZH/archives/wound-segmentation-master.zip")
    raw_wseg_archive: Path = Path("data/raw/WSEG/archives/wseg_dataset.zip")
    raw_mendeley_archive: Path = Path("data/raw/MENDELEY/archives/Wound Image Dataset.zip")
    
    extracted_root: Path = Path("data/interim/stage_3_1/extracted")
    extracted_azh: Path = Path("data/interim/stage_3_1/extracted/AZH")
    extracted_wseg: Path = Path("data/interim/stage_3_1/extracted/WSEG")
    extracted_mendeley: Path = Path("data/interim/stage_3_1/extracted/MENDELEY")
    
    interim_manifests: Path = Path("data/interim/stage_3_1/extraction_manifests")
    interim_reports: Path = Path("data/interim/stage_3_1/integrity_reports")
    quarantine_dir: Path = Path("data/interim/stage_3_1/quarantine")
    
    processed_dir: Path = Path("data/processed")

@dataclass
class PreprocessingSpec:
    target_canvas_size: Tuple[int, int] = (512, 512)
    spatial_transform: str = "aspect_preserving_letterbox"
    
    # Interpolation strategies (strictly separated)
    image_interpolation: str = "bilinear"  # Options: bilinear, bicubic
    mask_interpolation: str = "nearest"    # Categorical masks must use nearest-neighbor
    
    # ImageNet statistics are applied internally by the DCS-Stem chromatic branch
    # only. The dataset must emit raw RGB in [0, 1] and must not normalize.
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Mask binarization & threshold parameters
    canonical_mask_threshold: int = 128
    sensitivity_threshold_sweep: List[int] = field(default_factory=lambda: [64, 96, 128, 160, 192])
    
    # Distance transform parameters for Boundary Loss
    boundary_tolerance_tau: float = 2.0  # pixels on 512x512 canvas
    boundary_max_distance_cap: float = 20.0
    
    # Post-processing restriction
    enable_component_size_filtering: bool = False  # Strictly False to preserve micro-lesions
    negative_control_noise_floor_pixels: int = 10  # Decision boundary for negative foot classification

@dataclass
class CohortPartitionSpec:
    # Training
    train_source: str = "AZH_Challenge_Train + AZH_Medetec_Train"
    train_expected_pairs: int = 961
    train_expected_unique_images: int = 945
    train_multi_mask_clusters: int = 15
    train_sampling_strategy: str = "two_stage_cluster_balanced"
    
    # Validation
    val_source: str = "AZH_Challenge_Val (Clean Holdout)"
    val_expected_pairs: int = 193
    val_expected_unique_images: int = 192
    val_leaked_clusters_excluded: int = 7
    
    # External Target
    target_source: str = "Mendeley_WSEG"
    target_total_records: int = 2686
    target_total_unique_images: int = 2508
    target_native_disjoint_images: int = 2186
    target_mismatched_diagnostic_images: int = 322
    target_repeated_clusters: int = 139
    
    # Healthy Controls
    controls_source: str = "Mendeley_Normal"
    controls_expected_images: int = 2757
    controls_mask_derivation: str = "derived_all_background_zero"

@dataclass
class Stage4DataLoaderConfig:
    """
    Production Stage 4 data-loading defaults.

    The batch size follows the approved baseline optimisation protocol
    (G4-D9: 8). ``num_workers=0`` is the approved initial setting and avoids
    Windows multiprocessing pitfalls; the loader factory can override it.
    """
    batch_size: int = 8
    num_workers: int = 0
    train_seed: int = 42
    pin_memory: bool = False
    persistent_workers: bool = False


@dataclass
class MasterPipelineConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    preprocessing: PreprocessingSpec = field(default_factory=PreprocessingSpec)
    cohorts: CohortPartitionSpec = field(default_factory=CohortPartitionSpec)
    random_seeds: List[int] = field(default_factory=lambda: [42, 1337, 2024])

