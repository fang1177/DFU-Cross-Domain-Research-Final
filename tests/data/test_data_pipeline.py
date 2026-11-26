"""
Automated unit and regression tests for safe extraction, manifest integrity,
spatial transforms, distance transforms, and cluster-balanced sampling.
Operates strictly on synthetic, temporary in-memory fixtures without modifying real dataset files.
"""

import pytest
import os
import io
import shutil
import zipfile
import hashlib
import numpy as np
from pathlib import Path
from PIL import Image

from scripts.data.safe_extract_datasets import validate_archive_member
from src.data.transforms import (
    calculate_letterbox_geometry,
    apply_letterbox_image,
    apply_letterbox_mask,
    compute_tolerance_boundary_fields
)
from src.data.samplers import TwoStageClusterSampler

def test_safe_extraction_rejects_path_traversal():
    """Verifies that zip members attempting directory escape are rejected."""
    target_root = Path("temp_test_root")
    seen_paths = set()
    
    # Normal member
    zinfo_valid = zipfile.ZipInfo("subfolder/image.png")
    is_valid, err = validate_archive_member(zinfo_valid, target_root, seen_paths, None)
    assert is_valid is True
    assert err is None
    
    # Path traversal members
    traversal_paths = [
        "../outside.txt",
        "subfolder/../../escaped.txt",
        "/absolute/path.txt",
        "\\windows\\absolute.txt"
    ]
    for bad_path in traversal_paths:
        zinfo_bad = zipfile.ZipInfo(bad_path)
        is_valid, err = validate_archive_member(zinfo_bad, target_root, seen_paths, None)
        assert is_valid is False
        assert "Path traversal attempt" in err

def test_letterbox_geometry_preserves_aspect_ratio():
    """Verifies isotropic scaling and symmetric padding calculations."""
    # Test 1: Square image (331x331) to (512x512)
    geom_sq = calculate_letterbox_geometry(331, 331, 512, 512)
    assert geom_sq["scaled_width"] == 512
    assert geom_sq["scaled_height"] == 512
    assert geom_sq["pad_left"] == 0
    assert geom_sq["pad_top"] == 0
    
    # Test 2: Landscape rectangular image (640x480) to (512x512)
    geom_rect = calculate_letterbox_geometry(640, 480, 512, 512)
    assert geom_rect["scaled_width"] == 512
    assert geom_rect["scaled_height"] == 384
    assert geom_rect["pad_top"] == 64
    assert geom_rect["pad_bottom"] == 64
    assert geom_rect["pad_left"] == 0
    assert geom_rect["pad_right"] == 0

def test_mask_letterbox_uses_nearest_neighbor():
    """Verifies that mask letterboxing preserves strict categorical values {0, 1}."""
    synthetic_mask = Image.fromarray(np.array([[0, 1], [1, 0]], dtype=np.uint8) * 255, mode="L")
    letterboxed = apply_letterbox_mask(synthetic_mask, target_size=(512, 512))
    
    arr = np.array(letterboxed)
    unique_vals = set(np.unique(arr))
    # Must only contain original values (0 and 255), zero interpolation blur
    assert unique_vals.issubset({0, 255})
    assert len(unique_vals) <= 2

def test_tolerance_boundary_fields():
    """Verifies that boundary distance fields have exact zero deadband within tau."""
    # Create synthetic 512x512 mask with a 100x100 square foreground
    synthetic_mask = np.zeros((512, 512), dtype=np.uint8)
    synthetic_mask[200:300, 200:300] = 1
    
    phi_fp, phi_fn = compute_tolerance_boundary_fields(synthetic_mask, tau=2.0, max_cap=20.0)
    
    # 1. Non-negativity check
    assert (phi_fp >= 0.0).all()
    assert (phi_fn >= 0.0).all()
    
    # 2. Maximum distance cap check
    assert phi_fp.max() <= 20.0
    assert phi_fn.max() <= 20.0
    
    # 3. Disjoint activity check: FP field only active on background, FN field only on foreground
    assert (phi_fp[synthetic_mask == 1] == 0.0).all()
    assert (phi_fn[synthetic_mask == 0] == 0.0).all()
    
    # 4. Deadband check: pixels 1 pixel from boundary must have zero penalty
    # Pixel (199, 250) is 1 pixel outside boundary -> phi_fp should be 0.0
    assert phi_fp[199, 250] == 0.0
    # Pixel (201, 250) is 1 pixel inside boundary -> phi_fn should be 0.0
    assert phi_fn[201, 250] == 0.0
    
    # Pixel 10 pixels outside boundary -> phi_fp should be 10 - 2 = 8.0
    assert abs(phi_fp[190, 250] - 8.0) < 0.5

def test_empty_mask_boundary_fields_are_zero():
    """Verifies that authentic negative control masks generate zero boundary fields."""
    empty_mask = np.zeros((512, 512), dtype=np.uint8)
    phi_fp, phi_fn = compute_tolerance_boundary_fields(empty_mask, tau=2.0, max_cap=20.0)
    assert (phi_fp == 0.0).all()
    assert (phi_fn == 0.0).all()

def test_two_stage_cluster_sampler_balance():
    """Verifies that Two-Stage Cluster Sampling guarantees equal epoch exposure."""
    # Synthetic cluster map: 930 single-mask clusters, 15 dual-mask clusters
    cluster_map = {f"cluster_{i}": [i] for i in range(930)}
    idx_counter = 930
    for j in range(15):
        cluster_map[f"multi_cluster_{j}"] = [idx_counter, idx_counter + 1]
        idx_counter += 2
        
    sampler = TwoStageClusterSampler(cluster_map, seed=42)
    assert len(sampler) == 945  # Exactly 945 unique images per epoch
    
    # Run over 10 epochs and count exposure frequency
    cluster_hits = {c: 0 for c in cluster_map}
    for ep in range(10):
        sampler.set_epoch(ep)
        sampled_indices = list(sampler)
        assert len(sampled_indices) == 945
        
        # Invert index to cluster
        for idx in sampled_indices:
            # find which cluster it belongs to
            for c, id_list in cluster_map.items():
                if idx in id_list:
                    cluster_hits[c] += 1
                    break
                    
    # Every single cluster must have been sampled exactly 10 times in 10 epochs
    for c, count in cluster_hits.items():
        assert count == 10, f"Cluster {c} was sampled {count} times instead of 10!"

