"""
Stage 3.2B-R2 Regression Test Suite.
Validates the corrected manifest and metadata layer under stage_3_2B_manifest_repair_v1.
"""

import csv
import json
import pytest
import numpy as np
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path("E:/DFU_Cross_Domain_Research")
CORRECTED_DIR = PROJECT_ROOT / "data/processed/stage_3_2B_manifest_repair_v1"
CORRECTED_MANIFEST = CORRECTED_DIR / "canonical_manifest_corrected.csv"
CORRECTED_META_DIR = CORRECTED_DIR / "metadata_corrected"

CANONICAL_DIR = PROJECT_ROOT / "data/processed/stage_3_2B_canonical"
MASK_DIR = CANONICAL_DIR / "masks"
DIAG_MASK_DIR = CANONICAL_DIR / "diagnostic_masks"

@pytest.fixture(scope="session")
def corrected_manifest():
    assert CORRECTED_MANIFEST.exists(), f"Corrected manifest not found: {CORRECTED_MANIFEST}"
    return list(csv.DictReader(open(CORRECTED_MANIFEST, encoding="utf-8")))

def test_01_nonzero_mask_never_classified_as_empty(corrected_manifest):
    """Regression Test 1: A mask with nonzero foreground pixels is never classified as empty."""
    for r in corrected_manifest:
        if r["mask_status"] == "authentic_negative_empty" and r["proc_mask_filename"]:
            mask_path = MASK_DIR / r["proc_mask_filename"]
            arr = np.array(Image.open(mask_path))
            assert (arr > 0).sum() == 0, f"Record {r['cohort']}_{r['pair_id']} classified empty but has {(arr > 0).sum()} pixels!"

def test_02_truly_empty_mask_classified_as_empty(corrected_manifest):
    """Regression Test 2: Truly empty masks are correctly classified as empty."""
    empty_recs = [r for r in corrected_manifest if r["mask_status"] == "authentic_negative_empty"]
    assert len(empty_recs) == 24, f"Expected exactly 24 authentic empty masks, got {len(empty_recs)}"
    for r in empty_recs:
        mask_path = MASK_DIR / r["proc_mask_filename"]
        arr = np.array(Image.open(mask_path))
        assert (arr > 0).sum() == 0

def test_03_small_ulcer_below_one_percent_remains_nonempty(corrected_manifest):
    """Regression Test 3: Masks with foreground percentage below 1% remain nonempty and strict_binary."""
    # Find sample of small ulcers that were previously affected
    small_ulcers = [
        r for r in corrected_manifest
        if r["cohort"] in ("source_training", "internal_validation")
        and r["mask_status"] == "strict_binary"
        and float(r["proc_fg_count"]) < 2621.44  # < 1.0% of 512x512
        and float(r["proc_fg_count"]) > 0
    ]
    assert len(small_ulcers) >= 591, f"Expected at least 591 small ulcers, got {len(small_ulcers)}"
    for r in small_ulcers[:20]:
        assert r["mask_status"] == "strict_binary"
        assert int(r["proc_fg_count"]) > 0

def test_04_manifest_fg_count_equals_actual_mask_pixel_count(corrected_manifest):
    """Regression Test 4: Foreground count in the manifest equals actual processed mask pixel count."""
    # Sample 50 records across cohorts
    sample = corrected_manifest[:50]
    for r in sample:
        fn = r["proc_mask_filename"]
        if not fn: continue
        p = (DIAG_MASK_DIR if r["is_diagnostic"] == "True" else MASK_DIR) / fn
        arr = np.array(Image.open(p))
        actual_fg = int((arr == 255).sum())
        assert int(r["proc_fg_count"]) == actual_fg, (
            f"Record {r['cohort']}_{r['pair_id']} proc_fg_count {r['proc_fg_count']} != actual {actual_fg}"
        )

def test_05_metadata_foreground_pct_matches_mask_array(corrected_manifest):
    """Regression Test 5: Foreground percentage in metadata matches the actual processed mask array."""
    sample = [r for r in corrected_manifest if r["mask_status"] == "strict_binary"][:30]
    for r in sample:
        meta_p = CORRECTED_META_DIR / f"{r['cohort']}_{r['pair_id']}_metadata.json"
        assert meta_p.exists()
        meta = json.load(open(meta_p, encoding="utf-8"))
        fn = Path(meta["mask"]["processed_path"]).name
        arr = np.array(Image.open(MASK_DIR / fn))
        actual_pct = (int((arr == 255).sum()) / (512 * 512)) * 100.0
        reported_pct = meta["mask"]["mask_foreground_pct"]
        assert abs(actual_pct - reported_pct) < 1e-4

def test_06_authentic_negative_empty_masks_contain_zero_foreground(corrected_manifest):
    """Regression Test 6: authentic_negative_empty masks contain strictly zero foreground pixels."""
    empty_recs = [r for r in corrected_manifest if r["mask_status"] == "authentic_negative_empty"]
    for r in empty_recs:
        arr = np.array(Image.open(MASK_DIR / r["proc_mask_filename"]))
        assert np.all(arr == 0)

def test_07_invalid_or_uncertain_masks_flagged():
    """Regression Test 7: Multiclass mask 0233 remains flagged as diagnostic and isolated."""
    corr = list(csv.DictReader(open(CORRECTED_MANIFEST, encoding="utf-8")))
    r0233 = next(r for r in corr if r["pair_id"] == "0233")
    assert r0233["mask_status"] == "diagnostic_multiclass"
    assert r0233["is_diagnostic"] == "True"

def test_08_alternate_annotation_roles_preserved(corrected_manifest):
    """Regression Test 8: Alternate annotation roles remain preserved."""
    clusters = {}
    for r in corrected_manifest:
        cid = r["cluster_id"]
        if cid not in clusters: clusters[cid] = []
        clusters[cid].append(r)
    
    multi = {k: v for k, v in clusters.items() if len(v) > 1}
    for cid, recs in multi.items():
        prim = [r for r in recs if r["is_primary_reference"] == "True"]
        assert len(prim) == 1, f"Cluster {cid} does not have exactly 1 primary reference!"

def test_09_diagnostic_records_excluded_from_canonical_training(corrected_manifest):
    """Regression Test 9: Diagnostic-only records remain excluded from canonical training."""
    mismatched = [r for r in corrected_manifest if r["cohort"] == "mismatched_target"]
    assert len(mismatched) == 322
    for r in mismatched:
        assert r["is_diagnostic"] == "True"

def test_10_corrected_manifest_preserves_original_record_identifiers(corrected_manifest):
    """Regression Test 10: The corrected manifest preserves original record identifiers and split assignments."""
    orig = list(csv.DictReader(open(CANONICAL_DIR / "manifests/canonical_manifest.csv", encoding="utf-8")))
    assert len(corrected_manifest) == len(orig)
    for r_corr, r_orig in zip(corrected_manifest, orig):
        assert r_corr["cohort"] == r_orig["cohort"]
        assert r_corr["pair_id"] == r_orig["pair_id"]
        assert r_corr["cluster_id"] == r_orig["cluster_id"]
        assert r_corr["orig_img_sha256"] == r_orig["orig_img_sha256"]
        assert r_corr["orig_mask_sha256"] == r_orig["orig_mask_sha256"]

