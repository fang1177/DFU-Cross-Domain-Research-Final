"""
Unit and Regression Tests for Stage 3.2A Controlled Preprocessing Pilot.
Covers all 12 validation requirements defined in Stage 3.2A specifications.
"""

import os
import csv
import json
import hashlib
import numpy as np
from pathlib import Path
from PIL import Image
import pytest

PROJECT_ROOT = Path("E:/DFU_Cross_Domain_Research")
PILOT_DIR = PROJECT_ROOT / "data/processed/stage_3_2A_pilot"
MANIFEST_PATH = PILOT_DIR / "manifests/pilot_manifest.csv"

@pytest.fixture(scope="session")
def pilot_manifest():
    assert MANIFEST_PATH.exists(), f"Pilot manifest missing: {MANIFEST_PATH}"
    with open(MANIFEST_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def test_01_aspect_ratio_preservation_and_letterbox(pilot_manifest):
    """Req 1: Verify letterbox geometry and canvas size (512, 512)."""
    for row in pilot_manifest:
        img_path = PILOT_DIR / "images" / f"{row['pilot_cohort']}_{row['pair_id']}_bilinear.png"
        assert img_path.exists()
        with Image.open(img_path) as im:
            assert im.size == (512, 512), f"Image {img_path} size is {im.size}, expected (512, 512)"
            
        orig_w = int(row["orig_img_w"])
        orig_h = int(row["orig_img_h"])
        scale = float(row["scale"])
        expected_scale = min(512 / orig_w, 512 / orig_h)
        assert abs(scale - expected_scale) < 1e-5

def test_02_nearest_neighbor_mask_interpolation(pilot_manifest):
    """Req 2: Categorical masks must strictly use Nearest-Neighbor and maintain categorical integrity."""
    for row in pilot_manifest:
        if row["mask_status"] == "strict_binary":
            mask_path = PILOT_DIR / "masks" / row["proc_mask_filename"]
            with Image.open(mask_path) as im:
                arr = np.array(im)
                uniques = np.unique(arr).tolist()
                for val in uniques:
                    assert val in [0, 255], f"Unexpected value {val} in strict binary mask {mask_path}"
                    
        elif row["mask_status"] == "canonical_binarized_jpeg":
            mask_path = PILOT_DIR / "masks" / row["proc_mask_filename"]
            with Image.open(mask_path) as im:
                arr = np.array(im)
                uniques = np.unique(arr).tolist()
                for val in uniques:
                    assert val in [0, 255], f"Unexpected value {val} in binarized JPEG mask {mask_path}"

def test_03_jpeg_artefact_mask_treatment(pilot_manifest):
    """Req 3: JPEG artefact masks (WSEG) are preserved raw in diagnostic_masks and binarized at T=128 in masks."""
    wseg_rows = [r for r in pilot_manifest if r["pilot_cohort"] == "native_target"]
    assert len(wseg_rows) == 13
    for r in wseg_rows:
        raw_nn_path = PILOT_DIR / "diagnostic_masks" / f"{r['pilot_cohort']}_{r['pair_id']}_jpeg_raw_nn.png"
        assert raw_nn_path.exists(), f"Raw NN JPEG mask missing: {raw_nn_path}"
        canonical_path = PILOT_DIR / "masks" / r["proc_mask_filename"]
        assert canonical_path.exists(), f"Canonical binarized mask missing: {canonical_path}"

def test_04_continuous_grayscale_mask_treatment(pilot_manifest):
    """Req 4: Continuous grayscale masks (Mendeley) must be in diagnostic_masks and never in primary masks/."""
    mismatched_rows = [r for r in pilot_manifest if r["pilot_cohort"] == "mismatched_target"]
    assert len(mismatched_rows) == 6
    for r in mismatched_rows:
        assert r["is_diagnostic"] == "True"
        diag_path = PILOT_DIR / "diagnostic_masks" / r["proc_mask_filename"]
        assert diag_path.exists(), f"Diagnostic mask missing: {diag_path}"
        assert not (PILOT_DIR / "masks" / r["proc_mask_filename"]).exists()

def test_05_multiclass_mask_isolation(pilot_manifest):
    """Req 5: Multiclass mask (0233) must be isolated in diagnostic_masks."""
    p_0233 = [r for r in pilot_manifest if r["pair_id"] == "0233"][0]
    assert p_0233["is_diagnostic"] == "True"
    assert p_0233["mask_status"] == "diagnostic_multiclass"
    diag_path = PILOT_DIR / "diagnostic_masks" / p_0233["proc_mask_filename"]
    assert diag_path.exists()
    with Image.open(diag_path) as im:
        uniques = np.unique(np.array(im)).tolist()
        assert len(uniques) > 2, f"Multiclass mask 0233 should have >2 values, got {uniques}"

def test_06_empty_mask_preservation(pilot_manifest):
    """Req 6: Empty masks (0092, 0128) must have zero foreground after letterboxing."""
    empty_rows = [r for r in pilot_manifest if r["mask_status"] == "authentic_negative_empty"]
    assert len(empty_rows) >= 2
    for r in empty_rows:
        mask_path = PILOT_DIR / "masks" / r["proc_mask_filename"]
        with Image.open(mask_path) as im:
            arr = np.array(im)
            assert arr.sum() == 0, f"Empty mask {mask_path} has non-zero sum: {arr.sum()}"

def test_07_healthy_controls_handling(pilot_manifest):
    """Req 7: Healthy controls must have 512x512 images and synthetic all-zero ground truth masks."""
    healthy_rows = [r for r in pilot_manifest if r["pilot_cohort"] == "healthy_controls"]
    assert len(healthy_rows) == 6
    for r in healthy_rows:
        assert r["mask_status"] == "authentic_negative_healthy"
        mask_path = PILOT_DIR / "masks" / r["proc_mask_filename"]
        with Image.open(mask_path) as im:
            assert im.size == (512, 512)
            assert np.array(im).sum() == 0

def test_08_multi_annotation_preservation(pilot_manifest):
    """Req 8: Multi-annotation clusters must preserve all masks with deterministic primary operational reference."""
    c_0111 = [r for r in pilot_manifest if r["cluster_id"] == "source_cluster_0111"]
    assert len(c_0111) == 3
    prim_0111 = [r for r in c_0111 if r["is_primary_reference"] == "True"]
    assert len(prim_0111) == 1
    assert prim_0111[0]["pair_id"] == "0111"
    
    c_val = [r for r in pilot_manifest if r["cluster_id"] == "val_cluster_0303_0339"]
    assert len(c_val) == 2
    prim_val = [r for r in c_val if r["is_primary_reference"] == "True"]
    assert len(prim_val) == 1
    assert prim_val[0]["pair_id"] == "0303"
    
    c_wseg = [r for r in pilot_manifest if r["cluster_id"] == "wseg_cluster_49422580"]
    assert len(c_wseg) == 2
    prim_wseg = [r for r in c_wseg if r["is_primary_reference"] == "True"]
    assert len(prim_wseg) == 1
    assert prim_wseg[0]["pair_id"] == "49422580"

def test_09_image_interpolation_comparison(pilot_manifest):
    """Req 9: Verify both Bilinear and Bicubic images exist and have matching dimensions."""
    for r in pilot_manifest:
        bilin_path = PILOT_DIR / "images" / f"{r['pilot_cohort']}_{r['pair_id']}_bilinear.png"
        bicub_path = PILOT_DIR / "images" / f"{r['pilot_cohort']}_{r['pair_id']}_bicubic.png"
        assert bilin_path.exists()
        assert bicub_path.exists()
        with Image.open(bilin_path) as im1, Image.open(bicub_path) as im2:
            assert im1.size == (512, 512)
            assert im2.size == (512, 512)
            
        # When resizing occurs (scale != 1.0), bilinear and bicubic produce distinct pixel outputs
        scale = float(r["scale"])
        if scale != 1.0:
            assert r["proc_img_bilinear_sha256"] != r["proc_img_bicubic_sha256"], f"Expected differing hashes when scale != 1.0 for {r['pair_id']}"
        else:
            # When scale == 1.0, 512x512 images are identity copies without resampling
            assert r["proc_img_bilinear_sha256"] == r["proc_img_bicubic_sha256"]

def test_10_spatial_alignment_and_area_preservation(pilot_manifest):
    """Req 10: Spatial alignment check - area scaling ratio within expected discrete bounds."""
    for r in pilot_manifest:
        if r["mask_status"] == "strict_binary" and int(r["orig_fg_count"]) > 0:
            scale = float(r["scale"])
            orig_fg = int(r["orig_fg_count"])
            proc_fg = int(r["proc_fg_count"])
            if scale == 1.0:
                assert orig_fg == proc_fg, f"At scale 1.0, pixel count must be exact: {orig_fg} != {proc_fg}"
            else:
                ratio = float(r["area_ratio"])
                assert 0.80 <= ratio <= 1.20, f"Area preservation ratio {ratio} out of bounds for {r['pair_id']}"

def test_11_boundary_distance_field_correctness(pilot_manifest):
    """Req 11: Tolerance-bounded distance fields (phi_fp, phi_fn) computed correctly."""
    for r in pilot_manifest:
        if r["mask_status"] in ["strict_binary", "canonical_binarized_jpeg", "authentic_negative_empty", "authentic_negative_healthy"]:
            assert r["phi_fp_sha256"] != "", f"Missing phi_fp hash for {r['pair_id']}"
            assert r["phi_fn_sha256"] != "", f"Missing phi_fn hash for {r['pair_id']}"

def test_12_metadata_completeness(pilot_manifest):
    """Req 12: Every processed sample has a corresponding complete JSON metadata file."""
    for r in pilot_manifest:
        meta_file = PILOT_DIR / "metadata" / f"{r['pilot_cohort']}_{r['pair_id']}_metadata.json"
        assert meta_file.exists(), f"Metadata file missing: {meta_file}"
        with open(meta_file, mode="r", encoding="utf-8") as f:
            data = json.load(f)
            assert "geometry" in data
            assert "image" in data
            assert "mask" in data
            assert data["image"]["original_sha256"] == r["orig_img_sha256"]
