"""
Stage 3.2B Test Suite — Canonical Preprocessing Validation.

Tests verify correctness, completeness, and methodology compliance
of the full-dataset canonical preprocessing output.

Runs AFTER stage 3.2B execution. All tests are READ-ONLY on the output.
"""

import csv
import json
import hashlib
import numpy as np
import pytest
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image

PROJECT_ROOT = Path("E:/DFU_Cross_Domain_Research")
CANONICAL_DIR = PROJECT_ROOT / "data/processed/stage_3_2B_canonical"
IMG_DIR       = CANONICAL_DIR / "images"
MASK_DIR      = CANONICAL_DIR / "masks"
DIAG_MASK_DIR = CANONICAL_DIR / "diagnostic_masks"
DIAG_IMG_DIR  = CANONICAL_DIR / "diagnostic_images"
MANIFEST_DIR  = CANONICAL_DIR / "manifests"
METADATA_DIR  = CANONICAL_DIR / "metadata"

CANONICAL_MANIFEST = MANIFEST_DIR / "canonical_manifest.csv"

# Expected cohort record counts (locked from stage 1D-R2)
EXPECTED_COHORT_COUNTS = {
    "source_training":   962,
    "internal_validation": 193,
    "native_target":     2364,
    "mismatched_target": 322,
    "healthy_controls":  2757,
}
EXPECTED_TOTAL = sum(EXPECTED_COHORT_COUNTS.values())  # 6598


@pytest.fixture(scope="session")
def manifest():
    assert CANONICAL_MANIFEST.exists(), f"Canonical manifest not found: {CANONICAL_MANIFEST}"
    rows = list(csv.DictReader(open(CANONICAL_MANIFEST, encoding="utf-8")))
    return rows


@pytest.fixture(scope="session")
def manifest_by_cohort(manifest):
    result = defaultdict(list)
    for r in manifest:
        result[r["cohort"]].append(r)
    return result


# ---------------------------------------------------------------------------
# test_01 — Manifest row count
# ---------------------------------------------------------------------------
def test_01_manifest_total_row_count(manifest):
    """Canonical manifest must have exactly EXPECTED_TOTAL rows."""
    assert len(manifest) == EXPECTED_TOTAL, (
        f"Expected {EXPECTED_TOTAL} manifest rows, got {len(manifest)}"
    )


# ---------------------------------------------------------------------------
# test_02 — Per-cohort counts
# ---------------------------------------------------------------------------
def test_02_per_cohort_counts(manifest):
    """Each cohort must have the exact locked count."""
    counts = Counter(r["cohort"] for r in manifest)
    for cohort, expected in EXPECTED_COHORT_COUNTS.items():
        actual = counts.get(cohort, 0)
        assert actual == expected, (
            f"Cohort '{cohort}': expected {expected}, got {actual}"
        )


# ---------------------------------------------------------------------------
# test_03 — No leaked validation pairs
# ---------------------------------------------------------------------------
def test_03_no_leaked_validation_pairs(manifest_by_cohort):
    """
    The 7 AZH_Challenge_Val leaked pair_ids must not appear in
    internal_validation cohort.
    """
    val_recon = list(csv.DictReader(
        open(PROJECT_ROOT / "results/temporary/stage_1d_r2/validation_record_reconciliation.csv", encoding="utf-8")
    ))
    leaked_ids = {r["pair_id"] for r in val_recon if r["train_overlap_status"] == "TRAIN_OVERLAP"}
    assert len(leaked_ids) == 7, f"Expected 7 leaked pairs, found {len(leaked_ids)}"

    val_rows = manifest_by_cohort["internal_validation"]
    val_pair_ids = {r["pair_id"] for r in val_rows}
    intersection = leaked_ids & val_pair_ids
    assert len(intersection) == 0, (
        f"Leaked pair_ids found in internal_validation: {intersection}"
    )


# ---------------------------------------------------------------------------
# test_04 — Image size is exactly 512×512
# ---------------------------------------------------------------------------
def test_04_canonical_images_are_512x512(manifest_by_cohort):
    """
    Sample 20 images per non-diagnostic cohort; all must be 512×512 PNG.
    """
    target_cohorts = ["source_training", "internal_validation", "native_target", "healthy_controls"]
    for cohort in target_cohorts:
        rows = manifest_by_cohort[cohort]
        sample = rows[:20]
        for r in sample:
            pair_id = r["pair_id"]
            img_path = IMG_DIR / f"{cohort}_{pair_id}_bilinear.png"
            assert img_path.exists(), f"Missing image: {img_path}"
            with Image.open(img_path) as img:
                assert img.size == (512, 512), (
                    f"Image {img_path.name} has size {img.size}, expected (512, 512)"
                )
                assert img.mode == "RGB", f"Image {img_path.name} mode is {img.mode}, expected RGB"


# ---------------------------------------------------------------------------
# test_05 — Mask size is exactly 512×512 and strictly binary {0, 255}
# ---------------------------------------------------------------------------
def test_05_canonical_masks_are_binary_512x512(manifest_by_cohort):
    """
    Sample 20 non-diagnostic, non-healthy masks; all must be 512×512 L-mode
    with values strictly in {0, 255}.
    """
    target_cohorts = ["source_training", "internal_validation", "native_target"]
    for cohort in target_cohorts:
        rows = [r for r in manifest_by_cohort[cohort] if r["is_diagnostic"] == "False"]
        sample = rows[:20]
        for r in sample:
            pair_id = r["pair_id"]
            fname = r["proc_mask_filename"]
            if not fname:
                continue
            mask_path = MASK_DIR / fname
            assert mask_path.exists(), f"Missing mask: {mask_path}"
            with Image.open(mask_path) as m:
                arr = np.array(m)
                assert arr.shape == (512, 512), (
                    f"Mask {fname} shape {arr.shape}, expected (512, 512)"
                )
                uniq = set(np.unique(arr).tolist())
                assert uniq.issubset({0, 255}), (
                    f"Mask {fname} has non-binary values: {uniq}"
                )


# ---------------------------------------------------------------------------
# test_06 — Healthy controls have all-zero masks
# ---------------------------------------------------------------------------
def test_06_healthy_controls_have_zero_masks(manifest_by_cohort):
    """All healthy control masks must be all-zero (synthetic negative)."""
    rows = manifest_by_cohort["healthy_controls"]
    sample = rows[:30]
    for r in sample:
        fname = r["proc_mask_filename"]
        mask_path = MASK_DIR / fname
        assert mask_path.exists(), f"Missing healthy mask: {mask_path}"
        arr = np.array(Image.open(mask_path))
        assert arr.sum() == 0, f"Healthy mask {fname} has non-zero pixels: {arr.sum()}"
        assert arr.shape == (512, 512), f"Healthy mask {fname} has wrong shape: {arr.shape}"


# ---------------------------------------------------------------------------
# test_07 — Mismatched target records are diagnostic-only
# ---------------------------------------------------------------------------
def test_07_mismatched_target_is_diagnostic_only(manifest_by_cohort):
    """All mismatched_target records must have is_diagnostic=True."""
    rows = manifest_by_cohort["mismatched_target"]
    assert len(rows) == 322, f"Expected 322 mismatched_target rows, got {len(rows)}"
    non_diagnostic = [r for r in rows if r["is_diagnostic"] != "True"]
    assert len(non_diagnostic) == 0, (
        f"{len(non_diagnostic)} mismatched_target records have is_diagnostic != True"
    )


# ---------------------------------------------------------------------------
# test_08 — Mismatched target images in diagnostic_images/, not images/
# ---------------------------------------------------------------------------
def test_08_mismatched_images_in_diagnostic_dir(manifest_by_cohort):
    """Mismatched target images must exist in diagnostic_images/, not images/."""
    rows = manifest_by_cohort["mismatched_target"]
    sample = rows[:20]
    for r in sample:
        pair_id = r["pair_id"]
        cohort = r["cohort"]
        diag_path = DIAG_IMG_DIR / f"{cohort}_{pair_id}_bilinear.png"
        wrong_path = IMG_DIR / f"{cohort}_{pair_id}_bilinear.png"
        assert diag_path.exists(), f"Diagnostic image missing: {diag_path}"
        assert not wrong_path.exists(), (
            f"Diagnostic image erroneously in images/: {wrong_path}"
        )


# ---------------------------------------------------------------------------
# test_09 — Mask nearest-neighbor interpolation (no continuous interpolation)
# ---------------------------------------------------------------------------
def test_09_mask_nearest_neighbor_interpolation(manifest_by_cohort):
    """
    For AZH strict binary masks: values after letterboxing must still be
    strictly in {0, 255} — no intermediate values from continuous interpolation.
    """
    rows = [
        r for r in manifest_by_cohort["source_training"]
        if r["mask_status"] == "strict_binary"
    ]
    sample = rows[:15]
    for r in sample:
        fname = r["proc_mask_filename"]
        if not fname:
            continue
        mask_path = MASK_DIR / fname
        arr = np.array(Image.open(mask_path))
        uniq = set(np.unique(arr).tolist())
        assert uniq.issubset({0, 255}), (
            f"strict_binary mask {fname} has non-binary values after NN interp: {uniq}"
        )


# ---------------------------------------------------------------------------
# test_10 — WSEG canonical masks binarized at T=128
# ---------------------------------------------------------------------------
def test_10_wseg_canonical_binarization(manifest_by_cohort):
    """
    For native_target (WSEG) canonical masks: verify they are binarized
    (all-zero or all-255 pixel values) and that the raw NN diagnostic sweep
    files exist in diagnostic_masks/.
    """
    rows = [r for r in manifest_by_cohort["native_target"] if r["mask_status"] == "canonical_binarized_jpeg"]
    sample = rows[:15]
    for r in sample:
        pair_id = r["pair_id"]
        cohort = r["cohort"]
        # Canonical mask must be binary
        fname = r["proc_mask_filename"]
        mask_path = MASK_DIR / fname
        assert mask_path.exists(), f"Missing WSEG canonical mask: {mask_path}"
        arr = np.array(Image.open(mask_path))
        uniq = set(np.unique(arr).tolist())
        assert uniq.issubset({0, 255}), f"WSEG canonical mask {fname} not binary: {uniq}"
        # Sensitivity sweep must exist
        for t in [64, 96, 128, 160, 192]:
            sweep_path = DIAG_MASK_DIR / f"{cohort}_{pair_id}_sweep_{t}.png"
            assert sweep_path.exists(), f"Missing sweep mask: {sweep_path}"


# ---------------------------------------------------------------------------
# test_11 — Primary reference selection (lowest alphanumeric pair_id per cluster)
# ---------------------------------------------------------------------------
def test_11_primary_reference_selection(manifest):
    """
    For every cluster with multiple records, the is_primary_reference=True record
    must have the lowest alphanumeric pair_id in that cluster.
    """
    clusters = defaultdict(list)
    for r in manifest:
        clusters[r["cluster_id"]].append(r)

    multi_clusters = {cid: rows for cid, rows in clusters.items() if len(rows) > 1}
    for cid, rows in multi_clusters.items():
        primary_rows = [r for r in rows if r["is_primary_reference"] == "True"]
        assert len(primary_rows) == 1, (
            f"Cluster {cid}: expected 1 primary, got {len(primary_rows)}"
        )
        primary_pid = primary_rows[0]["pair_id"]
        all_pids = sorted(r["pair_id"] for r in rows)
        expected_primary = all_pids[0]
        assert primary_pid == expected_primary, (
            f"Cluster {cid}: primary is {primary_pid!r}, expected lowest {expected_primary!r}"
        )


# ---------------------------------------------------------------------------
# test_12 — Metadata files exist for every manifest record
# ---------------------------------------------------------------------------
def test_12_metadata_completeness(manifest):
    """Every manifest record must have a corresponding metadata JSON file."""
    missing = []
    for r in manifest:
        cohort = r["cohort"]
        pair_id = r["pair_id"]
        meta_path = METADATA_DIR / f"{cohort}_{pair_id}_metadata.json"
        if not meta_path.exists():
            missing.append(str(meta_path))
    assert len(missing) == 0, (
        f"{len(missing)} metadata files missing. First 5: {missing[:5]}"
    )


# ---------------------------------------------------------------------------
# test_13 — Metadata JSON is valid and contains required fields
# ---------------------------------------------------------------------------
def test_13_metadata_json_validity(manifest):
    """Sample metadata JSONs must be parseable and contain required fields."""
    required_top = {"pair_id", "cohort", "cluster_id", "unique_content_id",
                    "is_primary_reference", "image", "geometry", "mask"}
    required_img = {"original_sha256", "processed_bilinear_sha256", "processed_bicubic_sha256"}
    required_mask = {"mask_status", "is_diagnostic", "processed_sha256"}

    # Sample 1 per cohort
    cohort_seen = set()
    for r in manifest:
        cohort = r["cohort"]
        if cohort in cohort_seen:
            continue
        cohort_seen.add(cohort)
        meta_path = METADATA_DIR / f"{cohort}_{r['pair_id']}_metadata.json"
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        missing_top = required_top - set(meta.keys())
        assert len(missing_top) == 0, f"Metadata {meta_path.name} missing top-level fields: {missing_top}"
        missing_img = required_img - set(meta["image"].keys())
        assert len(missing_img) == 0, f"Metadata {meta_path.name} missing image fields: {missing_img}"
        missing_mask = required_mask - set(meta["mask"].keys())
        assert len(missing_mask) == 0, f"Metadata {meta_path.name} missing mask fields: {missing_mask}"
    assert len(cohort_seen) == 5, f"Only saw {len(cohort_seen)} cohorts in metadata sample"


# ---------------------------------------------------------------------------
# test_14 — SHA256 provenance (proc_img_bilinear_sha256 matches actual file)
# ---------------------------------------------------------------------------
def test_14_image_sha256_provenance(manifest_by_cohort):
    """
    For 10 source_training records, verify proc_img_bilinear_sha256 in manifest
    matches the actual file on disk.
    """
    rows = manifest_by_cohort["source_training"][:10]
    for r in rows:
        pair_id = r["pair_id"]
        cohort = r["cohort"]
        img_path = IMG_DIR / f"{cohort}_{pair_id}_bilinear.png"
        assert img_path.exists(), f"Missing image: {img_path}"
        h = hashlib.sha256()
        with open(img_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        actual_sha = h.hexdigest()
        assert actual_sha == r["proc_img_bilinear_sha256"], (
            f"SHA256 mismatch for {img_path.name}: "
            f"manifest={r['proc_img_bilinear_sha256']!r}, actual={actual_sha!r}"
        )


# ---------------------------------------------------------------------------
# test_15 — Mask SHA256 provenance
# ---------------------------------------------------------------------------
def test_15_mask_sha256_provenance(manifest_by_cohort):
    """
    For 10 native_target records, verify proc_mask_sha256 in manifest
    matches the actual canonical mask file on disk.
    """
    rows = [r for r in manifest_by_cohort["native_target"] if r["proc_mask_filename"]][:10]
    for r in rows:
        fname = r["proc_mask_filename"]
        mask_path = MASK_DIR / fname
        assert mask_path.exists(), f"Missing mask: {mask_path}"
        h = hashlib.sha256()
        with open(mask_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        actual_sha = h.hexdigest()
        assert actual_sha == r["proc_mask_sha256"], (
            f"Mask SHA256 mismatch for {fname}: "
            f"manifest={r['proc_mask_sha256']!r}, actual={actual_sha!r}"
        )


# ---------------------------------------------------------------------------
# test_16 — Boundary fields exist for canonical non-diagnostic masks
# ---------------------------------------------------------------------------
def test_16_boundary_fields_recorded(manifest_by_cohort):
    """
    Non-diagnostic, non-empty canonical masks must have non-empty
    phi_fp_sha256 and phi_fn_sha256 in manifest.
    """
    expected_statuses = {"strict_binary", "canonical_binarized_jpeg",
                         "authentic_negative_empty", "authentic_negative_healthy"}
    failing = []
    for cohort in ["source_training", "internal_validation", "native_target", "healthy_controls"]:
        rows = manifest_by_cohort[cohort]
        for r in rows[:5]:  # sample 5 per cohort
            if r["mask_status"] in expected_statuses:
                if not r["phi_fp_sha256"] or not r["phi_fn_sha256"]:
                    failing.append(f"{cohort}/{r['pair_id']} mask_status={r['mask_status']}")
    assert len(failing) == 0, f"Missing boundary field SHAs for: {failing}"


# ---------------------------------------------------------------------------
# test_17 — Aspect ratio preservation (letterbox geometry check)
# ---------------------------------------------------------------------------
def test_17_aspect_ratio_preservation(manifest):
    """
    For records where scale < 1.0 (image was downscaled), verify the
    letterboxed image has black padding in the correct region.
    """
    scaled_records = [r for r in manifest if float(r["scale"]) < 1.0][:10]
    for r in scaled_records:
        cohort = r["cohort"]
        pair_id = r["pair_id"]
        if cohort == "mismatched_target":
            img_path = DIAG_IMG_DIR / f"{cohort}_{pair_id}_bilinear.png"
        else:
            img_path = IMG_DIR / f"{cohort}_{pair_id}_bilinear.png"
        if not img_path.exists():
            continue
        arr = np.array(Image.open(img_path))
        pad_top = int(r["pad_top"])
        pad_left = int(r["pad_left"])
        # If there is top padding, those rows should be all-zero (black)
        if pad_top > 0:
            top_region = arr[:pad_top, :, :]
            assert top_region.max() == 0, (
                f"Non-zero padding in top region of {img_path.name}: max={top_region.max()}"
            )
        if pad_left > 0:
            left_region = arr[:, :pad_left, :]
            assert left_region.max() == 0, (
                f"Non-zero padding in left region of {img_path.name}: max={left_region.max()}"
            )


# ---------------------------------------------------------------------------
# test_18 — No duplicate pair_id within any cohort
# ---------------------------------------------------------------------------
def test_18_no_duplicate_pair_ids_within_cohort(manifest_by_cohort):
    """Within each cohort, pair_id must be unique."""
    for cohort, rows in manifest_by_cohort.items():
        pair_ids = [r["pair_id"] for r in rows]
        duplicates = [pid for pid, cnt in Counter(pair_ids).items() if cnt > 1]
        assert len(duplicates) == 0, (
            f"Cohort '{cohort}' has duplicate pair_ids: {duplicates[:5]}"
        )

