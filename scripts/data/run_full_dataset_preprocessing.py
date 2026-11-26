"""
Stage 3.2B: Full-Dataset Canonical Preprocessing Execution Script.

Applies deterministic letterboxing, interpolation separation, mask binarization,
diagnostic mask preservation, metadata generation, and provenance tracking
across all 6,598 cohort records.

Cohort Input Sources:
  - Source Training (962 pairs):  AZH_Challenge_Train + AZH_Medetec_Train
  - Internal Validation (193 pairs): AZH_Challenge_Val — 7 leaked pairs excluded
  - Native Target (2364 pairs): WSEG dimensions_match=True, BINARY_LIKE_JPEG_ARTEFACTS
  - Mismatched Target / Diagnostic (322 pairs): Mendeley dimensions_match=False
  - Healthy Controls (2757 images): Mendeley Nomal/

Output: data/processed/stage_3_2B_canonical/
Manifest: data/processed/stage_3_2B_canonical/manifests/canonical_manifest.csv

RULES (from approved methodology):
  - Image resampling: Bilinear (canonical), Bicubic (ablation variant)
  - Mask resampling: Nearest-Neighbor ONLY
  - Target size: 512x512 with letterboxing (pad with black)
  - WSEG threshold: T=128 (Provisional), sensitivity sweep {64,96,128,160,192} to diagnostic_masks
  - Medetec {0,1} -> {0,255} standardization
  - Healthy controls: synthetic all-zero mask
  - Mismatched target: diagnostic_masks only (QUARANTINED)
  - 7 AZH_Challenge_Val leaked pairs: EXCLUDED
  - Cluster assignment: lowest alphanumeric pair_id = is_primary_reference
  - Boundary fields (phi_fp, phi_fn) for non-diagnostic canonical masks only
"""

import os
import sys
import csv
import json
import hashlib
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("E:/DFU_Cross_Domain_Research")
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.transforms import (
    calculate_letterbox_geometry,
    apply_letterbox_image,
    apply_letterbox_mask,
    compute_tolerance_boundary_fields,
)

# Source manifests (LOCKED — never modify)
PAIR_AUDIT_CSV        = PROJECT_ROOT / "data/manifests/image_mask_pair_audit.csv"
MENDELEY_MANIFEST_CSV = PROJECT_ROOT / "data/interim/stage_3_1/extraction_manifests/mendeley_extraction_manifest.csv"
SRC_DUP_MAP_CSV       = PROJECT_ROOT / "results/temporary/stage_1d_r2/source_training_duplicate_mapping.csv"
VAL_RECON_CSV         = PROJECT_ROOT / "results/temporary/stage_1d_r2/validation_record_reconciliation.csv"

# Output directory (must be pre-created and empty)
OUT_DIR           = PROJECT_ROOT / "data/processed/stage_3_2B_canonical"
IMG_DIR           = OUT_DIR / "images"
MASK_DIR          = OUT_DIR / "masks"
DIAG_MASK_DIR     = OUT_DIR / "diagnostic_masks"
DIAG_IMG_DIR      = OUT_DIR / "diagnostic_images"
MANIFEST_DIR      = OUT_DIR / "manifests"
METADATA_DIR      = OUT_DIR / "metadata"

TARGET_SIZE = (512, 512)
WSEG_THRESHOLD = 128
SWEEP_THRESHOLDS = [64, 96, 128, 160, 192]

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def compute_sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def compute_sha256_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(arr.tobytes())
    return h.hexdigest()


def get_bounding_box(mask_arr: np.ndarray):
    pos = np.argwhere(mask_arr > 0)
    if len(pos) == 0:
        return None
    ymin, xmin = pos.min(axis=0)
    ymax, xmax = pos.max(axis=0)
    return int(ymin), int(xmin), int(ymax), int(xmax)


def get_disk_path(dataset_key: str, rel_path: str) -> Path:
    if dataset_key == "AZH":
        return PROJECT_ROOT / "data/interim/stage_3_1/extracted/AZH" / rel_path
    elif dataset_key == "WSEG":
        return PROJECT_ROOT / "data/interim/stage_3_1/extracted/WSEG" / rel_path
    elif dataset_key == "MENDELEY":
        # Mendeley files can live at either extracted/ or extracted/MENDELEY/
        p1 = PROJECT_ROOT / "data/interim/stage_3_1/extracted" / rel_path
        p2 = PROJECT_ROOT / "data/interim/stage_3_1/extracted/MENDELEY" / rel_path
        return p1 if p1.exists() else p2
    else:
        raise ValueError(f"Unknown dataset_key: {dataset_key}")


# ---------------------------------------------------------------------------
# Fail-fast pre-checks
# ---------------------------------------------------------------------------
def preflight_checks():
    log.info("=== STAGE 3.2B PRE-FLIGHT CHECKS ===")

    # 1. Source manifests exist
    for p in [PAIR_AUDIT_CSV, MENDELEY_MANIFEST_CSV, SRC_DUP_MAP_CSV, VAL_RECON_CSV]:
        assert p.exists(), f"MISSING source manifest: {p}"
        log.info(f"  [OK] Source manifest exists: {p.name}")

    # 2. Output subdirectories exist and are EMPTY
    for d in [IMG_DIR, MASK_DIR, DIAG_MASK_DIR, DIAG_IMG_DIR, MANIFEST_DIR, METADATA_DIR]:
        assert d.exists(), f"MISSING output directory: {d}"
        file_count = sum(1 for _ in d.rglob("*") if _.is_file())
        assert file_count == 0, (
            f"OUTPUT DIRECTORY NOT EMPTY: {d} has {file_count} files. "
            "Stage 3.2B must start from an empty canonical directory."
        )
        log.info(f"  [OK] Output dir empty: {d.relative_to(PROJECT_ROOT)}")

    # 3. Extracted source files exist (spot-check using pilot-known paths)
    spot_checks = [
        PROJECT_ROOT / "data/interim/stage_3_1/extracted/AZH/wound-segmentation-master/data/Foot Ulcer Segmentation Challenge/train/images/0111.png",
        PROJECT_ROOT / "data/interim/stage_3_1/extracted/WSEG/wseg_dataset/sample/49422580.jpg",
        PROJECT_ROOT / "data/interim/stage_3_1/extracted/MENDELEY/Nomal/Female_normal-001.jpg",
    ]
    for p in spot_checks:
        assert p.exists(), f"MISSING extracted file: {p}"
        log.info(f"  [OK] Extracted source file: {p.name}")

    log.info("=== PRE-FLIGHT CHECKS PASSED ===\n")


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------
def build_source_training_records(all_pairs, src_dups):
    """
    Build 962 source training records.
    Multi-annotation clusters: lowest alphanumeric pair_id = is_primary_reference.
    """
    records = []
    clusters = defaultdict(list)
    for r in src_dups:
        clusters[r["image_sha256"]].append(r)

    dup_pair_ids = {r["pair_id"] for r in src_dups}
    dup_image_paths = {r["image_path"] for r in src_dups}

    # Pairs that belong to duplicate clusters
    processed_shas = set()
    for sha, dup_rows in clusters.items():
        if sha in processed_shas:
            continue
        processed_shas.add(sha)
        sorted_pairs = sorted(dup_rows, key=lambda x: x["pair_id"])
        primary_id = sorted_pairs[0]["pair_id"]
        cluster_id = f"source_cluster_{primary_id}"
        unique_content_id = f"source_img_{primary_id}"
        for dup_row in dup_rows:
            pid = dup_row["pair_id"]
            pair_data = next(
                (p for p in all_pairs
                 if p["subset"] in ("AZH_Challenge_Train", "AZH_Medetec_Train")
                 and p["pair_id"] == pid),
                None,
            )
            if pair_data is None:
                log.warning(f"  [WARN] Duplicate pair {pid} not found in all_pairs")
                continue
            img_disk = get_disk_path("AZH", pair_data["image_path"])
            mask_disk = get_disk_path("AZH", pair_data["mask_path"])
            rec = dict(pair_data)
            rec["cohort"] = "source_training"
            rec["cluster_id"] = cluster_id
            rec["unique_content_id"] = unique_content_id
            rec["is_primary_reference"] = str(pid == primary_id)
            rec["image_disk_path"] = str(img_disk.relative_to(PROJECT_ROOT)).replace("\\", "/")
            rec["mask_disk_path"] = str(mask_disk.relative_to(PROJECT_ROOT)).replace("\\", "/")
            records.append(rec)

    # Non-duplicate single pairs
    single_source = [
        p for p in all_pairs
        if p["subset"] in ("AZH_Challenge_Train", "AZH_Medetec_Train")
        and p["image_path"] not in dup_image_paths
    ]
    for p in single_source:
        pid = p["pair_id"]
        img_disk = get_disk_path("AZH", p["image_path"])
        mask_disk = get_disk_path("AZH", p["mask_path"])
        rec = dict(p)
        rec["cohort"] = "source_training"
        rec["cluster_id"] = f"source_single_{pid}"
        rec["unique_content_id"] = f"source_single_{pid}"
        rec["is_primary_reference"] = "True"
        rec["image_disk_path"] = str(img_disk.relative_to(PROJECT_ROOT)).replace("\\", "/")
        rec["mask_disk_path"] = str(mask_disk.relative_to(PROJECT_ROOT)).replace("\\", "/")
        records.append(rec)

    log.info(f"Source Training: {len(records)} records ({len(clusters)} duplicate clusters + {len(single_source)} singles)")
    return records


def build_internal_validation_records(all_pairs, val_recon):
    """
    Build 193 internal validation records.
    7 TRAIN_OVERLAP records are excluded.
    Multi-annotation cluster 0303/0339 preserved; lowest pair_id = primary.
    """
    independent_ids = {r["pair_id"] for r in val_recon if r["train_overlap_status"] == "INDEPENDENT"}

    # Group by image_sha256 to identify duplicates within val set
    val_pairs = [
        p for p in all_pairs
        if p["subset"] == "AZH_Challenge_Val" and p["pair_id"] in independent_ids
    ]

    # Build SHA -> list mapping to find multi-annotation val clusters
    sha_to_pairs = defaultdict(list)
    for p in val_pairs:
        # We use the image_path to get sha from val_recon
        recon_row = next((r for r in val_recon if r["pair_id"] == p["pair_id"]), None)
        if recon_row:
            sha_to_pairs[recon_row["image_sha256"]].append(p)
        else:
            sha_to_pairs[p["pair_id"]].append(p)  # unique key fallback

    records = []
    for sha, group in sha_to_pairs.items():
        sorted_group = sorted(group, key=lambda x: x["pair_id"])
        primary_id = sorted_group[0]["pair_id"]
        if len(sorted_group) > 1:
            cluster_id = f"val_cluster_{primary_id}"
            unique_content_id = f"val_img_{primary_id}"
        else:
            cluster_id = f"val_single_{primary_id}"
            unique_content_id = f"val_img_{primary_id}"

        for p in sorted_group:
            pid = p["pair_id"]
            img_disk = get_disk_path("AZH", p["image_path"])
            mask_disk = get_disk_path("AZH", p["mask_path"])
            rec = dict(p)
            rec["cohort"] = "internal_validation"
            rec["cluster_id"] = cluster_id
            rec["unique_content_id"] = unique_content_id
            rec["is_primary_reference"] = str(pid == primary_id)
            rec["image_disk_path"] = str(img_disk.relative_to(PROJECT_ROOT)).replace("\\", "/")
            rec["mask_disk_path"] = str(mask_disk.relative_to(PROJECT_ROOT)).replace("\\", "/")
            records.append(rec)

    log.info(f"Internal Validation: {len(records)} records (excluded 7 leaked pairs)")
    return records


def build_native_target_records(all_pairs):
    """
    Build 2364 native target records (WSEG clean 1:1, BINARY_LIKE_JPEG_ARTEFACTS).
    Cluster 49422580/49498192 is multi-annotation; lowest alphanumeric = primary.
    All remaining WSEG pairs are singletons.
    """
    wseg_clean = [
        p for p in all_pairs
        if p["dataset_key"] == "WSEG"
        and p["dimensions_match"] == "True"
        and p["mask_type"] == "BINARY_LIKE_JPEG_ARTEFACTS"
    ]

    # Identify multi-annotation clusters in WSEG via image SHA from cross-dataset duplicate audit
    # We know from Stage 3.2A-R5 that 49422580 and 49498192 share the same image SHA
    # Build SHA mapping from pair_id by loading images to compare hashes (too slow for 2364)
    # Instead, use known multi-annotation pair from manifests: check for identical image filenames
    # The WSEG duplicate cluster is documented: pair 49498192 has the same JPEG content as 49422580
    # Strategy: group by pair_id; known cross-dataset duplicate audit JSON contains WSEG duplicate info
    # For 3.2B: treat each WSEG pair_id as its own cluster (singleton) EXCEPT the documented cluster
    # from Stage 1D-R2 / cross_dataset_duplicate_audit
    
    # Use the cross-dataset duplicate audit to find WSEG internal duplicates
    wseg_clusters = defaultdict(list)
    for p in wseg_clean:
        # Default: each pair is its own cluster (we'll merge known duplicates below)
        wseg_clusters[p["pair_id"]].append(p)

    # Known multi-annotation WSEG cluster (documented in Stage 3.2A-R5)
    # 49422580 and 49498192 share the same underlying image content
    wseg_known_cluster_pairs = ["49422580", "49498192"]
    if all(pid in wseg_clusters for pid in wseg_known_cluster_pairs):
        # Merge into single cluster keyed by lowest pair_id
        primary_id = min(wseg_known_cluster_pairs)
        combined = []
        for pid in wseg_known_cluster_pairs:
            combined.extend(wseg_clusters.pop(pid))
        wseg_clusters[primary_id] = combined

    records = []
    for primary_id, group in wseg_clusters.items():
        sorted_group = sorted(group, key=lambda x: x["pair_id"])
        actual_primary = sorted_group[0]["pair_id"]
        if len(sorted_group) > 1:
            cluster_id = f"wseg_cluster_{actual_primary}"
            unique_content_id = f"wseg_img_{actual_primary}"
        else:
            cluster_id = f"wseg_single_{actual_primary}"
            unique_content_id = f"wseg_single_{actual_primary}"

        for p in sorted_group:
            pid = p["pair_id"]
            img_disk = get_disk_path("WSEG", p["image_path"])
            mask_disk = get_disk_path("WSEG", p["mask_path"])
            rec = dict(p)
            rec["cohort"] = "native_target"
            rec["cluster_id"] = cluster_id
            rec["unique_content_id"] = unique_content_id
            rec["is_primary_reference"] = str(pid == actual_primary)
            rec["image_disk_path"] = str(img_disk.relative_to(PROJECT_ROOT)).replace("\\", "/")
            rec["mask_disk_path"] = str(mask_disk.relative_to(PROJECT_ROOT)).replace("\\", "/")
            records.append(rec)

    log.info(f"Native Target: {len(records)} records ({len(wseg_clusters)} unique content images)")
    return records


def build_mismatched_target_records(all_pairs):
    """
    Build 322 mismatched target (diagnostic) records.
    QUARANTINED — preprocessed to diagnostic_masks only. Never enters primary training.
    """
    mend_mismatch = [
        p for p in all_pairs
        if p["dataset_key"] == "MENDELEY"
        and p["dimensions_match"] == "False"
    ]
    records = []
    for p in mend_mismatch:
        pid = p["pair_id"]
        img_disk = get_disk_path("MENDELEY", p["image_path"])
        mask_disk = get_disk_path("MENDELEY", p["mask_path"])
        rec = dict(p)
        rec["cohort"] = "mismatched_target"
        rec["cluster_id"] = f"mendeley_mismatch_{pid}"
        rec["unique_content_id"] = f"mendeley_mismatch_{pid}"
        rec["is_primary_reference"] = "True"
        rec["image_disk_path"] = str(img_disk.relative_to(PROJECT_ROOT)).replace("\\", "/")
        rec["mask_disk_path"] = str(mask_disk.relative_to(PROJECT_ROOT)).replace("\\", "/")
        records.append(rec)

    log.info(f"Mismatched Target (QUARANTINED): {len(records)} records")
    return records


def build_healthy_control_records(mendeley_manifest):
    """
    Build 2757 healthy control records from Mendeley Nomal/.
    No mask files — synthetic all-zero mask will be generated.
    """
    nomal_entries = [
        m for m in mendeley_manifest
        if "Nomal" in m["extracted_relative_path"]
        and m["extension"].lower() in (".jpg", ".jpeg", ".png")
        and m["extraction_status"] == "SUCCESS"
    ]
    records = []
    for m in nomal_entries:
        img_disk = PROJECT_ROOT / "data/interim/stage_3_1/extracted" / m["extracted_relative_path"]
        if not img_disk.exists():
            img_disk = PROJECT_ROOT / "data/interim/stage_3_1/extracted/MENDELEY" / m["extracted_relative_path"]
        fname = m["filename"]
        rec = {
            "dataset_key": "MENDELEY",
            "subset": "Nomal",
            "pair_id": fname,
            "normalized_pair_id": fname,
            "pair_status": "healthy_control_no_mask",
            "dimensions_match": "N/A",
            "image_width": "N/A",
            "image_height": "N/A",
            "mask_width": "N/A",
            "mask_height": "N/A",
            "mask_type": "none",
            "foreground_estimation_status": "authentic_negative",
            "mask_foreground_pct": "0.0",
            "requires_visual_review": "False",
            "review_reason": "none",
            "image_path": m["extracted_relative_path"],
            "mask_path": "",
            "cohort": "healthy_controls",
            "cluster_id": f"healthy_{fname}",
            "unique_content_id": f"healthy_{fname}",
            "is_primary_reference": "True",
            "image_disk_path": str(img_disk.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "mask_disk_path": "",
        }
        records.append(rec)

    log.info(f"Healthy Controls: {len(records)} records")
    return records


# ---------------------------------------------------------------------------
# Processing core
# ---------------------------------------------------------------------------
def process_record(rec, idx, total):
    """
    Process one cohort record. Returns manifest_row dict.
    """
    cohort = rec["cohort"]
    pair_id = rec["pair_id"]
    img_disk = PROJECT_ROOT / rec["image_disk_path"]
    mask_disk = PROJECT_ROOT / rec["mask_disk_path"] if rec["mask_disk_path"] else None

    if idx % 500 == 0:
        log.info(f"  Processing {idx}/{total}: cohort={cohort} pair_id={pair_id}")

    # ---- Image ----
    orig_img_sha = compute_sha256_file(img_disk)
    with Image.open(img_disk) as raw_img:
        img_rgb = raw_img.convert("RGB")
        orig_img_w, orig_img_h = img_rgb.size
        geom = calculate_letterbox_geometry(orig_img_w, orig_img_h, TARGET_SIZE[0], TARGET_SIZE[1])
        img_bilinear = apply_letterbox_image(img_rgb, TARGET_SIZE, interpolation_mode=Image.BILINEAR)
        img_bicubic  = apply_letterbox_image(img_rgb, TARGET_SIZE, interpolation_mode=Image.BICUBIC)

    # Save images — both canonical (bilinear) and ablation (bicubic)
    if cohort == "mismatched_target":
        # Diagnostic image → diagnostic_images/
        img_bilinear_name = f"{cohort}_{pair_id}_bilinear.png"
        img_bilinear_path = DIAG_IMG_DIR / img_bilinear_name
        img_bilinear.save(img_bilinear_path, format="PNG")
        img_bicubic_name = f"{cohort}_{pair_id}_bicubic.png"
        img_bicubic_path = DIAG_IMG_DIR / img_bicubic_name
        img_bicubic.save(img_bicubic_path, format="PNG")
    else:
        img_bilinear_name = f"{cohort}_{pair_id}_bilinear.png"
        img_bilinear_path = IMG_DIR / img_bilinear_name
        img_bilinear.save(img_bilinear_path, format="PNG")
        img_bicubic_name = f"{cohort}_{pair_id}_bicubic.png"
        img_bicubic_path = IMG_DIR / img_bicubic_name
        img_bicubic.save(img_bicubic_path, format="PNG")

    proc_img_bilinear_sha = compute_sha256_file(img_bilinear_path)
    proc_img_bicubic_sha  = compute_sha256_file(img_bicubic_path)

    # Active ROI stats (bilinear)
    arr_img = np.array(img_bilinear, dtype=np.float32) / 255.0
    h_sl = slice(geom["pad_top"], geom["pad_top"] + geom["scaled_height"])
    w_sl = slice(geom["pad_left"], geom["pad_left"] + geom["scaled_width"])
    active_pixels = arr_img[h_sl, w_sl, :]
    img_mean = active_pixels.mean(axis=(0, 1)).tolist()
    img_std  = active_pixels.std(axis=(0, 1)).tolist()

    # ---- Mask ----
    mask_status = "none"
    orig_mask_sha = ""
    orig_mask_w = orig_mask_h = 0
    proc_mask_filename = ""
    proc_mask_sha = ""
    orig_fg_count = proc_fg_count = 0
    area_ratio = 1.0
    bbox_orig = bbox_proc = None
    unique_mask_values = []
    is_diagnostic = False
    phi_fp_sha = phi_fn_sha = ""
    proc_mask_path = None

    if cohort == "healthy_controls":
        # Synthetic all-zero mask
        mask_status = "authentic_negative_healthy"
        is_diagnostic = False
        zero_mask = Image.new("L", TARGET_SIZE, 0)
        proc_mask_filename = f"{cohort}_{pair_id}_zero_mask.png"
        proc_mask_path = MASK_DIR / proc_mask_filename
        zero_mask.save(proc_mask_path, format="PNG")
        proc_mask_sha = compute_sha256_file(proc_mask_path)
        unique_mask_values = [0]

    elif mask_disk and mask_disk.exists():
        orig_mask_sha = compute_sha256_file(mask_disk)
        with Image.open(mask_disk) as raw_mask:
            mask_l = raw_mask.convert("L")
            orig_mask_w, orig_mask_h = mask_l.size
            arr_orig = np.array(mask_l)

        # Medetec {0,1} -> {0,255} standardization
        orig_unique = np.unique(arr_orig).tolist()
        unique_mask_values = orig_unique
        if orig_unique == [0, 1]:
            arr_orig = (arr_orig * 255).astype(np.uint8)
            mask_l = Image.fromarray(arr_orig)

        bbox_orig = get_bounding_box(arr_orig)

        # Apply NN letterboxing to mask
        mask_nn = apply_letterbox_mask(mask_l, TARGET_SIZE, fill_value=0)
        arr_nn = np.array(mask_nn)

        # Cohort-specific mask handling
        if cohort == "mismatched_target":
            # QUARANTINED — raw NN to diagnostic_masks, plus threshold sweep
            is_diagnostic = True
            mask_status = "diagnostic_continuous_grayscale"
            proc_mask_filename = f"{cohort}_{pair_id}_continuous_raw.png"
            proc_mask_path = DIAG_MASK_DIR / proc_mask_filename
            mask_nn.save(proc_mask_path, format="PNG")
            proc_mask_sha = compute_sha256_file(proc_mask_path)
            for t in SWEEP_THRESHOLDS:
                t_arr = (arr_nn >= t).astype(np.uint8) * 255
                t_path = DIAG_MASK_DIR / f"{cohort}_{pair_id}_thresh_{t}.png"
                Image.fromarray(t_arr).save(t_path, format="PNG")

        elif pair_id == "0233" and cohort == "internal_validation":
            # Multiclass mask — diagnostic only
            is_diagnostic = True
            mask_status = "diagnostic_multiclass"
            proc_mask_filename = f"{cohort}_{pair_id}_multiclass_raw.png"
            proc_mask_path = DIAG_MASK_DIR / proc_mask_filename
            mask_nn.save(proc_mask_path, format="PNG")
            proc_mask_sha = compute_sha256_file(proc_mask_path)

        elif rec.get("mask_type", "") == "BINARY_LIKE_JPEG_ARTEFACTS":
            # WSEG: canonical binarized at T=128 + sensitivity sweep
            mask_status = "canonical_binarized_jpeg"
            is_diagnostic = False
            # Raw NN to diagnostic
            diag_raw = DIAG_MASK_DIR / f"{cohort}_{pair_id}_jpeg_raw_nn.png"
            mask_nn.save(diag_raw, format="PNG")
            # Canonical
            canon_arr = (arr_nn >= WSEG_THRESHOLD).astype(np.uint8) * 255
            proc_mask_filename = f"{cohort}_{pair_id}_canonical_mask.png"
            proc_mask_path = MASK_DIR / proc_mask_filename
            Image.fromarray(canon_arr).save(proc_mask_path, format="PNG")
            proc_mask_sha = compute_sha256_file(proc_mask_path)
            # Sweep
            for t in SWEEP_THRESHOLDS:
                t_arr = (arr_nn >= t).astype(np.uint8) * 255
                t_path = DIAG_MASK_DIR / f"{cohort}_{pair_id}_sweep_{t}.png"
                Image.fromarray(t_arr).save(t_path, format="PNG")
            orig_fg_count = int((arr_orig >= WSEG_THRESHOLD).sum())
            proc_fg_count = int((canon_arr == 255).sum())
            bbox_proc = get_bounding_box(canon_arr)

        # Stage 3.2B-R2 Correction: Never use integer-truncated percentage (int(float(pct)) == 0)
        # to determine mask emptiness, as it incorrectly truncates legitimate small ulcers (<1.0%).
        # Determine emptiness authoritatively from actual foreground pixel counts on the mask array.
        elif orig_unique == [0] or (len(orig_unique) == 1 and orig_unique[0] == 0) or int((arr_orig == 255).sum()) == 0:
            # Authentic empty negative-control mask (0 foreground pixels)
            mask_status = "authentic_negative_empty"
            is_diagnostic = False
            proc_mask_filename = f"{cohort}_{pair_id}_empty_mask.png"
            proc_mask_path = MASK_DIR / proc_mask_filename
            mask_nn.save(proc_mask_path, format="PNG")
            proc_mask_sha = compute_sha256_file(proc_mask_path)
            orig_fg_count = 0
            proc_fg_count = 0
            bbox_proc = None

        else:
            # Strict binary mask (AZH Challenge + Medetec after standardization)
            mask_status = "strict_binary"
            is_diagnostic = False
            proc_mask_filename = f"{cohort}_{pair_id}_canonical_mask.png"
            proc_mask_path = MASK_DIR / proc_mask_filename
            mask_nn.save(proc_mask_path, format="PNG")
            proc_mask_sha = compute_sha256_file(proc_mask_path)
            orig_fg_count = int((arr_orig == 255).sum())
            proc_fg_count = int((arr_nn == 255).sum())
            bbox_proc = get_bounding_box(arr_nn)

        # Area ratio for non-diagnostic, non-empty masks
        if orig_fg_count > 0:
            expected_scaled_fg = orig_fg_count * (geom["scale"] ** 2)
            area_ratio = proc_fg_count / expected_scaled_fg if expected_scaled_fg > 0 else 1.0

    # ---- Boundary distance fields ----
    if mask_status in ("strict_binary", "canonical_binarized_jpeg", "authentic_negative_empty", "authentic_negative_healthy"):
        target_arr = np.array(Image.open(proc_mask_path))
        bin_bool = target_arr == 255
        phi_fp, phi_fn = compute_tolerance_boundary_fields(bin_bool, tau=2.0, max_cap=20.0)
        phi_fp_sha = compute_sha256_array(phi_fp)
        phi_fn_sha = compute_sha256_array(phi_fn)

    # ---- Metadata JSON ----
    metadata = {
        "pair_id": pair_id,
        "cohort": cohort,
        "cluster_id": rec.get("cluster_id", ""),
        "unique_content_id": rec.get("unique_content_id", ""),
        "is_primary_reference": rec.get("is_primary_reference", "True"),
        "processing_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "image": {
            "original_path": rec.get("image_path", ""),
            "original_disk_path": rec["image_disk_path"],
            "original_dimensions": [orig_img_w, orig_img_h],
            "original_sha256": orig_img_sha,
            "processed_bilinear_path": str(img_bilinear_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "processed_bilinear_sha256": proc_img_bilinear_sha,
            "processed_bicubic_path": str(img_bicubic_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "processed_bicubic_sha256": proc_img_bicubic_sha,
            "active_roi_mean_rgb": img_mean,
            "active_roi_std_rgb": img_std,
        },
        "geometry": geom,
        "mask": {
            "original_path": rec.get("mask_path", ""),
            "original_disk_path": rec.get("mask_disk_path", ""),
            "original_dimensions": [orig_mask_w, orig_mask_h] if mask_disk else None,
            "original_sha256": orig_mask_sha,
            "processed_path": str(proc_mask_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if proc_mask_path else "",
            "processed_sha256": proc_mask_sha,
            "mask_status": mask_status,
            "is_diagnostic": is_diagnostic,
            "original_unique_values": unique_mask_values,
            "original_foreground_pixels": orig_fg_count,
            "processed_foreground_pixels": proc_fg_count,
            "area_preservation_ratio": float(area_ratio),
            "original_bbox": bbox_orig,
            "processed_bbox": bbox_proc,
            "phi_fp_sha256": phi_fp_sha,
            "phi_fn_sha256": phi_fn_sha,
        },
    }
    meta_file = METADATA_DIR / f"{cohort}_{pair_id}_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # ---- Manifest row ----
    return {
        "cohort": cohort,
        "pair_id": pair_id,
        "cluster_id": rec.get("cluster_id", ""),
        "unique_content_id": rec.get("unique_content_id", ""),
        "is_primary_reference": rec.get("is_primary_reference", "True"),
        "orig_img_path": rec.get("image_path", ""),
        "orig_img_w": orig_img_w,
        "orig_img_h": orig_img_h,
        "orig_img_sha256": orig_img_sha,
        "proc_img_bilinear_sha256": proc_img_bilinear_sha,
        "proc_img_bicubic_sha256": proc_img_bicubic_sha,
        "scale": geom["scale"],
        "pad_left": geom["pad_left"],
        "pad_top": geom["pad_top"],
        "pad_right": geom["pad_right"],
        "pad_bottom": geom["pad_bottom"],
        "orig_mask_path": rec.get("mask_path", ""),
        "orig_mask_w": orig_mask_w,
        "orig_mask_h": orig_mask_h,
        "orig_mask_sha256": orig_mask_sha,
        "proc_mask_filename": proc_mask_filename,
        "proc_mask_sha256": proc_mask_sha,
        "mask_status": mask_status,
        "is_diagnostic": str(is_diagnostic),
        "orig_fg_count": orig_fg_count,
        "proc_fg_count": proc_fg_count,
        "area_ratio": round(float(area_ratio), 6),
        "phi_fp_sha256": phi_fp_sha,
        "phi_fn_sha256": phi_fn_sha,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_full_preprocessing():
    start_time = datetime.now(timezone.utc)
    log.info("=" * 70)
    log.info("STAGE 3.2B — FULL DATASET CANONICAL PREPROCESSING")
    log.info(f"Started: {start_time.isoformat()}")
    log.info("=" * 70)

    preflight_checks()

    # Load source manifests
    all_pairs = list(csv.DictReader(open(PAIR_AUDIT_CSV, encoding="utf-8")))
    mendeley_manifest = list(csv.DictReader(open(MENDELEY_MANIFEST_CSV, encoding="utf-8")))
    src_dups = list(csv.DictReader(open(SRC_DUP_MAP_CSV, encoding="utf-8")))
    val_recon = list(csv.DictReader(open(VAL_RECON_CSV, encoding="utf-8")))

    # Build all cohort records
    log.info("\n--- Building cohort record sets ---")
    source_training_recs   = build_source_training_records(all_pairs, src_dups)
    internal_val_recs      = build_internal_validation_records(all_pairs, val_recon)
    native_target_recs     = build_native_target_records(all_pairs)
    mismatched_target_recs = build_mismatched_target_records(all_pairs)
    healthy_control_recs   = build_healthy_control_records(mendeley_manifest)

    all_records = (
        source_training_recs
        + internal_val_recs
        + native_target_recs
        + mismatched_target_recs
        + healthy_control_recs
    )

    log.info(f"\nTotal records to process: {len(all_records)}")
    log.info(f"  source_training:   {len(source_training_recs)}")
    log.info(f"  internal_val:      {len(internal_val_recs)}")
    log.info(f"  native_target:     {len(native_target_recs)}")
    log.info(f"  mismatched_target: {len(mismatched_target_recs)}")
    log.info(f"  healthy_controls:  {len(healthy_control_recs)}")

    # Verify disk paths before processing
    log.info("\n--- Verifying source disk paths ---")
    missing = []
    for rec in all_records:
        img_p = PROJECT_ROOT / rec["image_disk_path"]
        if not img_p.exists():
            missing.append(str(img_p))
        if rec["mask_disk_path"]:
            mask_p = PROJECT_ROOT / rec["mask_disk_path"]
            if not mask_p.exists():
                missing.append(str(mask_p))
    if missing:
        log.error(f"MISSING {len(missing)} source files! First 5: {missing[:5]}")
        raise FileNotFoundError(f"{len(missing)} source files missing. Aborting.")
    log.info(f"  All {len(all_records)} source image paths verified.\n")

    # Process all records
    log.info("--- Processing records ---")
    manifest_rows = []
    errors = []
    for idx, rec in enumerate(all_records):
        try:
            row = process_record(rec, idx, len(all_records))
            manifest_rows.append(row)
        except Exception as e:
            errors.append({"idx": idx, "cohort": rec.get("cohort"), "pair_id": rec.get("pair_id"), "error": str(e)})
            log.error(f"  [ERROR] idx={idx} cohort={rec.get('cohort')} pair_id={rec.get('pair_id')}: {e}")

    log.info(f"\nProcessed: {len(manifest_rows)} records | Errors: {len(errors)}")

    # Write canonical manifest
    manifest_csv = MANIFEST_DIR / "canonical_manifest.csv"
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    log.info(f"Canonical manifest written: {manifest_csv} ({len(manifest_rows)} rows)")

    # Write error log if any
    if errors:
        err_csv = MANIFEST_DIR / "processing_errors.csv"
        with open(err_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["idx", "cohort", "pair_id", "error"])
            writer.writeheader()
            writer.writerows(errors)
        log.warning(f"Error log written: {err_csv} ({len(errors)} errors)")

    # Post-generation file count verification
    log.info("\n--- Post-generation file count verification ---")
    img_files  = list(IMG_DIR.glob("*.png"))
    mask_files = list(MASK_DIR.glob("*.png"))
    diag_mask_files = list(DIAG_MASK_DIR.glob("*.png"))
    diag_img_files  = list(DIAG_IMG_DIR.glob("*.png"))
    meta_files = list(METADATA_DIR.glob("*.json"))

    # Expected image count: 2 per non-diagnostic record × (source_training + internal_val + native_target + healthy_controls)
    # + 2 per mismatched_target diagnostic record (in diag_img_dir)
    non_diag_img_records = len(source_training_recs) + len(internal_val_recs) + len(native_target_recs) + len(healthy_control_recs)
    diag_img_records = len(mismatched_target_recs)
    expected_img_files = non_diag_img_records * 2  # bilinear + bicubic
    expected_diag_img_files = diag_img_records * 2

    log.info(f"  images/:            {len(img_files):5d} files  (expected ~{expected_img_files})")
    log.info(f"  masks/:             {len(mask_files):5d} files")
    log.info(f"  diagnostic_masks/:  {len(diag_mask_files):5d} files")
    log.info(f"  diagnostic_images/: {len(diag_img_files):5d} files  (expected ~{expected_diag_img_files})")
    log.info(f"  metadata/:          {len(meta_files):5d} files  (expected {len(manifest_rows)})")

    # Manifest vs metadata count must match
    assert len(meta_files) == len(manifest_rows), (
        f"Metadata file count ({len(meta_files)}) != manifest rows ({len(manifest_rows)})"
    )

    # Cohort counts in manifest
    from collections import Counter
    cohort_counts = Counter(r["cohort"] for r in manifest_rows)
    log.info("\n  Manifest cohort breakdown:")
    for cohort, count in sorted(cohort_counts.items()):
        log.info(f"    {cohort}: {count}")

    end_time = datetime.now(timezone.utc)
    elapsed = (end_time - start_time).total_seconds()
    log.info(f"\nStage 3.2B completed in {elapsed:.1f}s")
    log.info(f"  Errors: {len(errors)}")
    log.info(f"  Manifest: {manifest_csv}")
    log.info("=" * 70)

    return manifest_rows, errors


if __name__ == "__main__":
    manifest_rows, errors = run_full_preprocessing()
    if errors:
        sys.exit(1)

