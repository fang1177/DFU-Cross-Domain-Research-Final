"""
Stage 1C — Image-Level Integrity, Mask Quality and Cross-Dataset Overlap Audit
=============================================================================

Performs comprehensive, non-destructive image-level auditing of dataset ZIP archives:
  - AZH: data/raw/AZH/archives/wound-segmentation-master.zip
  - WSEG: data/raw/WSEG/archives/wseg_dataset.zip
  - MENDELEY: data/raw/MENDELEY/archives/Wound Image Dataset.zip

Reads image bytes directly from ZIP entries into memory, decodes with Pillow without
extracting datasets to disk, and executes:
  1. Archive Verification: checks live archive SHA-256 hashes against verified Stage 1B reference hashes.
  2. Image Integrity: decoding success, format, width, height, mode, channels, size, aspect ratio.
  3. Mask Quality: unique pixel values, binary/binary-like/grayscale/color/palette categorization,
     rigorous foreground reliability estimation (reliable/provisional/unavailable),
     empty/full/suspicious foreground detection, complete non-zero histogram recording.
  4. Image-Mask Alignment: paired dimension checking, mismatch detection, status classification,
     normalized numeric-ID mask lookup across supported image extensions.
  5. Cross-Dataset Overlap: exact SHA-256 matching and perceptual dHash (difference hash)
     near-duplicate matching across distinct dataset and subset cohorts (separating exact byte duplicates
     from perceptual near-duplicates).
  6. Deterministic Visual Samples: generates side-by-side previews (original, mask, overlay)
     and contact sheets for a deterministic sample under results/temporary/stage_1c/visual_samples/.
     Distinguishes encoding-based reliable overlays, provisional threshold-based overlays,
     and unavailable placeholders.
  7. Multi-Manifest Reporting: writes JSON, CSV, and Markdown audit reports using safe
     staging and verified rollback-protected finalization.

Safeguards:
  - Standard-library Python + Pillow only.
  - All paths strictly anchored to PROJECT_ROOT.
  - Read-only zipfile streaming (never extracts datasets or modifies archives).
  - Protects every planned Stage 1C output path (all 7 paths, including empty directories).
  - Staging and rollback: verifies content hashes of all backups before replacing targets,
    rolls back only destinations where modification was attempted, and preserves backup directory
    whenever it contains recovery material.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
import zipfile
from typing import Dict, List, Optional, Set, Tuple, Any

import PIL
from PIL import Image, ImageOps, ImageDraw, ImageFont


# Anchor project root to script file location
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent

# Output Paths (all anchored inside PROJECT_ROOT)
MANIFESTS_DIR = PROJECT_ROOT / "data" / "manifests"
STAGE_1C_RESULTS_DIR = PROJECT_ROOT / "results" / "temporary" / "stage_1c"
VISUAL_SAMPLES_DIR = STAGE_1C_RESULTS_DIR / "visual_samples"
CONTACT_SHEET_PATH = STAGE_1C_RESULTS_DIR / "stage_1c_contact_sheet.jpg"

OUTPUT_IMAGE_INTEGRITY_JSON = MANIFESTS_DIR / "image_integrity_audit.json"
OUTPUT_PAIR_AUDIT_CSV = MANIFESTS_DIR / "image_mask_pair_audit.csv"
OUTPUT_MASK_QUALITY_JSON = MANIFESTS_DIR / "mask_quality_summary.json"
OUTPUT_DUPLICATES_JSON = MANIFESTS_DIR / "cross_dataset_duplicate_audit.json"
OUTPUT_SUMMARY_MD = STAGE_1C_RESULTS_DIR / "stage_1c_summary.md"

ALL_STAGE_1C_OUTPUTS = [
    OUTPUT_IMAGE_INTEGRITY_JSON,
    OUTPUT_PAIR_AUDIT_CSV,
    OUTPUT_MASK_QUALITY_JSON,
    OUTPUT_DUPLICATES_JSON,
    OUTPUT_SUMMARY_MD,
    CONTACT_SHEET_PATH,
    VISUAL_SAMPLES_DIR,
]

# Dataset Archives
TARGET_ARCHIVES = {
    "AZH": PROJECT_ROOT / "data" / "raw" / "AZH" / "archives" / "wound-segmentation-master.zip",
    "WSEG": PROJECT_ROOT / "data" / "raw" / "WSEG" / "archives" / "wseg_dataset.zip",
    "MENDELEY": PROJECT_ROOT / "data" / "raw" / "MENDELEY" / "archives" / "Wound Image Dataset.zip",
}

# Reference SHA-256 Hashes established in Stage 1A / Stage 1B
REFERENCE_ARCHIVE_SHA256 = {
    "AZH": "15bf68722e5a293e6852443ed356d0155ad5be91cba7d792a7c668e7e7fe0eae",
    "WSEG": "0a48dd1838305c4d518403010e94c6ee4b5e22b259740793c404cf7f5a975dcc",
    "MENDELEY": "00370cab8eebe941fb25c7d7fc0e8fd34fc513cb96a04e695d0b6b2c8610bd2c",
}

# Image extensions filter
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Perceptual hash algorithm parameter
PERCEPTUAL_HAMMING_THRESHOLD = 6


# ---------------------------------------------------------------------------
# Perceptual Hashing Utilities (dHash)
# ---------------------------------------------------------------------------

def compute_dhash(img: Image.Image, hash_size: int = 8) -> int:
    """
    Compute 64-bit difference hash (dHash) for an in-memory Pillow image.
    Resizes to (hash_size + 1, hash_size), converts to grayscale, and compares
    adjacent horizontal pixels.
    """
    gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    if hasattr(gray, "get_flattened_data"):
        pixels = list(gray.get_flattened_data())
    else:
        pixels = list(gray.getdata())
    width = hash_size + 1

    diff_bits = 0
    for row in range(hash_size):
        row_offset = row * width
        for col in range(hash_size):
            left = pixels[row_offset + col]
            right = pixels[row_offset + col + 1]
            diff_bits = (diff_bits << 1) | (1 if left > right else 0)

    return diff_bits


def hamming_distance(h1: int, h2: int) -> int:
    """Compute Hamming distance between two 64-bit integers."""
    return bin(h1 ^ h2).count("1")


# ---------------------------------------------------------------------------
# Image & Mask Decoding and Inspection
# ---------------------------------------------------------------------------

def inspect_image_bytes(raw_bytes: bytes, rel_path: str, dataset_key: str) -> dict:
    """
    Decode raw image bytes in memory and extract dimensions, mode, and format.
    """
    size_bytes = len(raw_bytes)
    sha256_hash = hashlib.sha256(raw_bytes).hexdigest()

    result = {
        "dataset_key": dataset_key,
        "relative_path": rel_path,
        "size_bytes": size_bytes,
        "sha256": sha256_hash,
        "decode_success": False,
        "format": None,
        "mode": None,
        "channels": None,
        "width": None,
        "height": None,
        "aspect_ratio": None,
        "is_zero_dimension": False,
        "is_unusual_dimension": False,
        "error_message": None,
        "dhash": None,
    }

    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            result["format"] = img.format
            result["mode"] = img.mode
            w, h = img.size
            result["width"] = w
            result["height"] = h

            if w == 0 or h == 0:
                result["is_zero_dimension"] = True
            else:
                ar = round(w / h, 4)
                result["aspect_ratio"] = ar
                if ar > 4.0 or ar < 0.25 or w < 16 or h < 16 or w > 8000 or h > 8000:
                    result["is_unusual_dimension"] = True

            mode_channel_map = {
                "1": 1, "L": 1, "P": 1, "I": 1, "F": 1,
                "RGB": 3, "YCbCr": 3, "LAB": 3, "HSV": 3,
                "RGBA": 4, "CMYK": 4,
            }
            result["channels"] = mode_channel_map.get(img.mode, len(img.getbands()) if hasattr(img, "getbands") else None)

            result["dhash"] = f"{compute_dhash(img):016x}"
            result["decode_success"] = True

    except Exception as e:
        result["error_message"] = f"{type(e).__name__}: {str(e)}"

    return result


def inspect_mask_bytes(raw_bytes: bytes, rel_path: str, dataset_key: str) -> dict:
    """
    Decode raw mask bytes in memory and extract quality, unique values,
    histogram, mask type, and foreground distribution with explicit reliability status.

    Foreground Estimation Protocol:
      - STRICT_BINARY (0, 1), (0, 255), or (0, V): foreground_estimation_status = 'RELIABLE'
          (Measures pixel-value distributions and applies documented encoding conventions.
           Does not independently establish clinical semantic meaning without verification.)
      - BINARY_LIKE_JPEG_ARTEFACTS (JPEG mask with >90% of pixels at extremes):
          foreground_estimation_status = 'PROVISIONAL' (provisional threshold = 128)
      - FULL_FOREGROUND (all pixels nonzero, uniform value):
          foreground_estimation_status = 'RELIABLE'
      - EMPTY (all pixels zero):
          foreground_estimation_status = 'RELIABLE'
      - PALETTE_BASED (Pillow mode P or palette-mapped):
          foreground_estimation_status = 'UNAVAILABLE' (unless strictly binary palette mapping is established)
      - MULTICLASS_OR_DISCRETE (<= 5 distinct values, non-binary):
          foreground_estimation_status = 'PROVISIONAL'
      - GRAYSCALE_CONTINUOUS (> 5 values across grayscale range):
          foreground_estimation_status = 'UNAVAILABLE'
      - COLOR_CODED (RGB channels differ):
          foreground_estimation_status = 'UNAVAILABLE'
    """
    size_bytes = len(raw_bytes)
    sha256_hash = hashlib.sha256(raw_bytes).hexdigest()

    result = {
        "dataset_key": dataset_key,
        "relative_path": rel_path,
        "size_bytes": size_bytes,
        "sha256": sha256_hash,
        "decode_success": False,
        "format": None,
        "mode": None,
        "is_palette_based": False,
        "palette_entry_count": 0,
        "palette_sample_rgb": [],
        "width": None,
        "height": None,
        "total_pixels": 0,
        "unique_pixel_values_sample": [],
        "num_unique_values": 0,
        "pixel_histogram_nonzero_bins": {},  # Preserves ALL non-zero histogram bins (pixel values or palette indices)
        "histogram_semantic_type": "PIXEL_VALUES",  # 'PIXEL_VALUES' or 'PALETTE_INDICES'
        "mask_type": "UNKNOWN",
        "foreground_estimation_status": "UNAVAILABLE",  # RELIABLE, PROVISIONAL, UNAVAILABLE
        "foreground_estimation_reason": None,
        "foreground_threshold_used": None,
        "foreground_pixels_estimated": None,
        "foreground_percentage_estimated": None,
        "is_completely_empty": False,
        "is_completely_full": False,
        "is_suspicious_small_foreground": False,
        "requires_visual_review": False,
        "review_reason": None,
        "error_message": None,
    }

    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            result["format"] = img.format
            result["mode"] = img.mode
            w, h = img.size
            result["width"] = w
            result["height"] = h
            total_pix = w * h
            result["total_pixels"] = total_pix

            if total_pix == 0:
                result["is_completely_empty"] = True
                result["mask_type"] = "ZERO_DIMENSION"
                result["foreground_estimation_status"] = "UNAVAILABLE"
                result["foreground_estimation_reason"] = "Mask has zero dimensions"
                result["requires_visual_review"] = True
                result["review_reason"] = "Zero dimension mask"
                return result

            # -------------------------------------------------------------
            # Mode & Palette Inspection
            # -------------------------------------------------------------
            if img.mode == "P" or img.palette is not None:
                result["is_palette_based"] = True
                result["histogram_semantic_type"] = "PALETTE_INDICES"
                palette_data = img.getpalette()  # List of [R, G, B, R, G, B, ...]
                palette_colors = []
                if palette_data:
                    palette_colors = [
                        (palette_data[i], palette_data[i + 1], palette_data[i + 2])
                        for i in range(0, min(len(palette_data), 768), 3)
                    ]
                result["palette_entry_count"] = len(palette_colors)
                result["palette_sample_rgb"] = [list(c) for c in palette_colors[:10]]

                # Palette index analysis (explicitly indexed by palette entry index)
                idx_hist = img.histogram()
                used_indices = [idx for idx in range(min(256, len(idx_hist))) if idx_hist[idx] > 0]
                result["num_unique_values"] = len(used_indices)
                result["unique_pixel_values_sample"] = used_indices[:20]
                # Preserve ALL non-zero palette index bins
                result["pixel_histogram_nonzero_bins"] = {
                    idx: idx_hist[idx] for idx in used_indices
                }

                # Evaluate palette colors corresponding to used indices
                used_colors = set()
                for idx in used_indices:
                    if idx < len(palette_colors):
                        used_colors.add(palette_colors[idx])

                result["mask_type"] = "PALETTE_BASED"
                result["requires_visual_review"] = True

                # Check if palette is strictly binary monochrome (e.g. black (0,0,0) and white (255,255,255))
                is_pure_bw_palette = (
                    len(used_indices) <= 2 and
                    used_colors.issubset({(0, 0, 0), (255, 255, 255)})
                )

                if is_pure_bw_palette and len(used_indices) == 2:
                    fg_idx = None
                    for idx in used_indices:
                        if idx < len(palette_colors) and palette_colors[idx] == (255, 255, 255):
                            fg_idx = idx
                            break
                    if fg_idx is not None:
                        result["foreground_estimation_status"] = "RELIABLE"
                        result["foreground_estimation_reason"] = (
                            f"Palette mask with verified two-color palette {used_colors}; "
                            f"palette index {fg_idx} maps to RGB (255, 255, 255). Observed index encoding is reliable; "
                            "clinical semantic validity requires visual verification."
                        )
                        result["foreground_threshold_used"] = fg_idx
                        fg_pix = idx_hist[fg_idx]
                        result["foreground_pixels_estimated"] = fg_pix
                        fg_pct = round((fg_pix / total_pix) * 100.0, 4)
                        result["foreground_percentage_estimated"] = fg_pct
                    else:
                        result["foreground_estimation_status"] = "UNAVAILABLE"
                        result["foreground_estimation_reason"] = (
                            "Palette mask has <=2 colors but foreground mapping to white is ambiguous."
                        )
                        result["review_reason"] = "Palette mask requires visual confirmation of color mapping"
                else:
                    result["foreground_estimation_status"] = "UNAVAILABLE"
                    result["foreground_estimation_reason"] = (
                        f"Palette mask (mode P) with {len(used_indices)} active indices and {len(used_colors)} distinct palette colors. "
                        "Histogram represents palette indices, not grayscale intensities or colors; foreground percentage withheld."
                    )
                    result["review_reason"] = f"Palette-based mask with {len(used_colors)} palette colors requires custom color mapping"

                result["decode_success"] = True
                return result

            # Non-palette image inspection
            result["histogram_semantic_type"] = "PIXEL_VALUES"
            if img.mode in ("RGB", "RGBA"):
                r, g, b = img.convert("RGB").split()
                r_bytes = r.tobytes()
                g_bytes = g.tobytes()
                b_bytes = b.tobytes()
                if r_bytes == g_bytes and g_bytes == b_bytes:
                    gray_img = r
                    is_rgb_color = False
                else:
                    gray_img = img.convert("L")
                    is_rgb_color = True
            else:
                gray_img = img.convert("L")
                is_rgb_color = False

            # Extract histogram of grayscale values (256 bins)
            hist = gray_img.histogram()
            non_zero_bins = [val for val in range(256) if hist[val] > 0]
            result["num_unique_values"] = len(non_zero_bins)
            result["unique_pixel_values_sample"] = non_zero_bins[:20]
            # Preserve ALL non-zero histogram bins
            result["pixel_histogram_nonzero_bins"] = {
                val: hist[val] for val in non_zero_bins
            }

            # -------------------------------------------------------------
            # Mask Classification and Foreground Estimation
            # -------------------------------------------------------------

            # Case A: Color-coded mask (RGB channels differ)
            if is_rgb_color:
                result["mask_type"] = "COLOR_CODED"
                result["foreground_estimation_status"] = "UNAVAILABLE"
                result["foreground_estimation_reason"] = (
                    "Mask contains non-identical RGB channels; arbitrary grayscale conversion "
                    "does not represent valid clinical foreground. Specific palette mapping required."
                )
                result["requires_visual_review"] = True
                result["review_reason"] = "Color-coded mask channels require specific palette mapping"

            # Case B: Completely empty mask (single value = 0)
            elif len(non_zero_bins) == 1 and non_zero_bins[0] == 0:
                result["mask_type"] = "EMPTY"
                result["foreground_estimation_status"] = "RELIABLE"
                result["foreground_estimation_reason"] = (
                    "All pixels have value 0 (completely background under standard zero-background convention)."
                )
                result["foreground_threshold_used"] = 1
                result["foreground_pixels_estimated"] = 0
                result["foreground_percentage_estimated"] = 0.0
                result["is_completely_empty"] = True
                result["requires_visual_review"] = True
                result["review_reason"] = "Completely empty mask (0 foreground pixels)"

            # Case C: Uniform nonzero mask (single value v > 0)
            elif len(non_zero_bins) == 1 and non_zero_bins[0] > 0:
                uniform_val = non_zero_bins[0]
                result["mask_type"] = "FULL_FOREGROUND"
                result["foreground_estimation_status"] = "RELIABLE"
                result["foreground_estimation_reason"] = (
                    f"All pixels share single uniform non-zero value {uniform_val} (100% foreground under nonzero foreground convention). "
                    "Pixel distribution reliably measured; clinical validity requires visual verification."
                )
                result["foreground_threshold_used"] = uniform_val
                result["foreground_pixels_estimated"] = total_pix
                result["foreground_percentage_estimated"] = 100.0
                result["is_completely_full"] = True
                result["requires_visual_review"] = True
                result["review_reason"] = f"Uniform 100% full-foreground mask (all pixels = {uniform_val})"

            # Case D: Strict binary mask with {0, 1} or {0, 255}
            elif set(non_zero_bins) == {0, 1}:
                result["mask_type"] = "BINARY_STRICT"
                result["foreground_estimation_status"] = "RELIABLE"
                result["foreground_estimation_reason"] = (
                    "Strict binary mask with observed encoding {0: background, 1: foreground}. "
                    "Pixel encoding is reliably measured under this convention; clinical semantic validity requires verification."
                )
                result["foreground_threshold_used"] = 1
                fg_pix = hist[1]
                result["foreground_pixels_estimated"] = fg_pix
                fg_pct = round((fg_pix / total_pix) * 100.0, 4)
                result["foreground_percentage_estimated"] = fg_pct

            elif set(non_zero_bins) == {0, 255}:
                result["mask_type"] = "BINARY_STRICT"
                result["foreground_estimation_status"] = "RELIABLE"
                result["foreground_estimation_reason"] = (
                    "Strict binary mask with observed encoding {0: background, 255: foreground}. "
                    "Pixel encoding is reliably measured under this convention; clinical semantic validity requires verification."
                )
                result["foreground_threshold_used"] = 255
                fg_pix = hist[255]
                result["foreground_pixels_estimated"] = fg_pix
                fg_pct = round((fg_pix / total_pix) * 100.0, 4)
                result["foreground_percentage_estimated"] = fg_pct

            # Case E: Strict binary with arbitrary non-standard 2 values {0, V}
            elif len(non_zero_bins) == 2 and 0 in non_zero_bins:
                non_zero_val = [v for v in non_zero_bins if v != 0][0]
                result["mask_type"] = "BINARY_STRICT"
                result["foreground_estimation_status"] = "RELIABLE"
                result["foreground_estimation_reason"] = (
                    f"Strict two-value mask with observed encoding {{0: background, {non_zero_val}: foreground}}. "
                    "Pixel encoding is reliably measured under this convention; clinical semantic validity requires verification."
                )
                result["foreground_threshold_used"] = non_zero_val
                fg_pix = hist[non_zero_val]
                result["foreground_pixels_estimated"] = fg_pix
                fg_pct = round((fg_pix / total_pix) * 100.0, 4)
                result["foreground_percentage_estimated"] = fg_pct

            # Case F: JPEG-like binary mask with DCT compression artefacts
            elif result["format"] == "JPEG" and ((hist[0] + hist[255]) / total_pix) > 0.90:
                result["mask_type"] = "BINARY_LIKE_JPEG_ARTEFACTS"
                result["foreground_estimation_status"] = "PROVISIONAL"
                result["foreground_estimation_reason"] = (
                    "JPEG compression artifacts detected (>90% of mass at values 0 and 255). "
                    "Foreground is provisionally estimated using threshold >= 128; requires visual verification."
                )
                threshold = 128
                result["foreground_threshold_used"] = threshold
                fg_pix = sum(hist[val] for val in range(threshold, 256))
                result["foreground_pixels_estimated"] = fg_pix
                fg_pct = round((fg_pix / total_pix) * 100.0, 4)
                result["foreground_percentage_estimated"] = fg_pct

            # Case G: Discrete / multiclass masks (<= 5 distinct values, not pure binary)
            elif len(non_zero_bins) <= 5:
                result["mask_type"] = "MULTICLASS_OR_DISCRETE"
                result["foreground_estimation_status"] = "PROVISIONAL"
                result["foreground_estimation_reason"] = (
                    f"Discrete mask with {len(non_zero_bins)} distinct values ({non_zero_bins}). "
                    "Class semantics are unverified; foreground estimate provisionally treats all nonzero values as foreground."
                )
                result["foreground_threshold_used"] = min([v for v in non_zero_bins if v > 0])
                fg_pix = sum(hist[val] for val in non_zero_bins if val > 0)
                result["foreground_pixels_estimated"] = fg_pix
                fg_pct = round((fg_pix / total_pix) * 100.0, 4)
                result["foreground_percentage_estimated"] = fg_pct
                result["requires_visual_review"] = True
                result["review_reason"] = f"Discrete non-binary mask with {len(non_zero_bins)} distinct values {non_zero_bins}"

            # Case H: Continuous grayscale gradient mask
            else:
                result["mask_type"] = "GRAYSCALE_CONTINUOUS"
                result["foreground_estimation_status"] = "UNAVAILABLE"
                result["foreground_estimation_reason"] = (
                    f"Continuous grayscale mask with {len(non_zero_bins)} distinct values across dynamic range. "
                    "Arbitrary thresholding would be scientifically unverified; foreground percentage withheld."
                )
                result["requires_visual_review"] = True
                result["review_reason"] = f"Continuous grayscale gradient mask ({len(non_zero_bins)} unique values)"

            # Check for suspicious tiny foreground (< 0.05% of image area) when foreground is available
            if result["foreground_percentage_estimated"] is not None:
                fg_pct_val = result["foreground_percentage_estimated"]
                if 0 < fg_pct_val < 0.05:
                    result["is_suspicious_small_foreground"] = True
                    result["requires_visual_review"] = True
                    prev_reason = result["review_reason"] or ""
                    result["review_reason"] = (
                        f"{prev_reason}; Suspiciously tiny foreground region ({fg_pct_val}% of image area)"
                    ).strip("; ")

            result["decode_success"] = True

    except Exception as e:
        result["error_message"] = f"{type(e).__name__}: {str(e)}"
        result["foreground_estimation_status"] = "UNAVAILABLE"
        result["foreground_estimation_reason"] = f"Decoding exception: {e}"
        result["requires_visual_review"] = True
        result["review_reason"] = f"Mask decoding failed: {e}"

    return result


# ---------------------------------------------------------------------------
# Visual Sample & Contact Sheet Generation
# ---------------------------------------------------------------------------

def generate_pair_preview(
    img_bytes: bytes,
    mask_bytes: bytes,
    out_path: Path,
    label_text: str,
    mask_audit_info: Optional[dict] = None,
    target_height: int = 300,
) -> bool:
    """
    Generate a 3-panel preview: [Original Image | Mask | Overlay] with label banner.
    Preserves original aspect ratio. Resizing is display-only.

    Requirements:
      - Preserve original visual appearance of palette-based and RGB color-coded masks in the mask panel.
      - Clearly distinguish overlay types in the banner:
          * 'PROVISIONAL OVERLAY — THRESHOLD: <val>' for provisional estimates.
          * 'ENCODING-BASED OVERLAY — CLINICAL SEMANTICS UNVERIFIED' for reliable encoding estimates.
      - Generate binary red overlay ONLY when mask foreground encoding is established
        and mask is NOT an unverified color-coded or palette-based representation.
      - If foreground encoding is UNAVAILABLE: do NOT create an overlay; show an explicit placeholder panel.
      - Retain dimension-mismatch warnings.
    """
    try:
        with Image.open(io.BytesIO(img_bytes)) as img_raw, Image.open(io.BytesIO(mask_bytes)) as mask_raw:
            orig_rgb = img_raw.convert("RGB")

            # Preserve visual appearance of mask in panel 2
            if mask_raw.mode == "P":
                mask_display_rgb = mask_raw.convert("RGB")
            elif mask_raw.mode in ("RGB", "RGBA"):
                mask_display_rgb = mask_raw.convert("RGB")
            else:
                mask_display_rgb = mask_raw.convert("L").convert("RGB")

            dim_mismatch = (orig_rgb.size != mask_raw.size)

            fg_status = mask_audit_info.get("foreground_estimation_status") if mask_audit_info else None
            mask_type = mask_audit_info.get("mask_type") if mask_audit_info else None
            thresh_val = mask_audit_info.get("foreground_threshold_used") if mask_audit_info else None

            # Overlay type logic
            can_generate_overlay = (
                fg_status in ("RELIABLE", "PROVISIONAL") and
                mask_type not in ("COLOR_CODED", "PALETTE_BASED", "GRAYSCALE_CONTINUOUS")
            )

            if fg_status == "PROVISIONAL" and can_generate_overlay:
                overlay_label = f"PROVISIONAL OVERLAY — THRESHOLD: {thresh_val or 128}"
            elif fg_status == "RELIABLE" and can_generate_overlay:
                overlay_label = "ENCODING-BASED OVERLAY — CLINICAL SEMANTICS UNVERIFIED"
            else:
                overlay_label = "OVERLAY UNAVAILABLE"

            # Build Overlay Panel
            if can_generate_overlay:
                if mask_raw.mode == "P":
                    mask_l = mask_raw.convert("RGB").convert("L")
                else:
                    mask_l = mask_raw.convert("L")

                if dim_mismatch:
                    mask_l = mask_l.resize(orig_rgb.size, Image.Resampling.NEAREST)

                threshold = thresh_val or 128
                mask_bin = mask_l.point(lambda p: 255 if p >= threshold else 0)
                red_layer = Image.new("RGB", orig_rgb.size, color=(255, 30, 30))
                blended = Image.composite(red_layer, orig_rgb, mask_bin)
                overlay_panel = Image.blend(orig_rgb, blended, alpha=0.55)
            else:
                # Informative placeholder panel: darkened image with clear text
                overlay_panel = orig_rgb.copy()
                dark_tint = Image.new("RGB", orig_rgb.size, color=(20, 20, 25))
                overlay_panel = Image.blend(overlay_panel, dark_tint, alpha=0.7)
                draw_ov = ImageDraw.Draw(overlay_panel)
                msg1 = "BINARY OVERLAY UNAVAILABLE"
                msg2 = f"Mask Type: {mask_type or mask_raw.mode}"
                msg3 = "Foreground encoding unverified"
                ow, oh = orig_rgb.size
                draw_ov.text((max(10, ow // 2 - 110), max(10, oh // 2 - 30)), msg1, fill=(255, 180, 180))
                draw_ov.text((max(10, ow // 2 - 110), max(10, oh // 2 - 10)), msg2, fill=(220, 220, 220))
                draw_ov.text((max(10, ow // 2 - 110), max(10, oh // 2 + 10)), msg3, fill=(180, 180, 180))

            # Resize display panels preserving aspect ratio to target_height
            w, h = orig_rgb.size
            scale = target_height / max(h, 1)
            display_w = max(int(w * scale), 1)
            display_h = target_height

            mw, mh = mask_raw.size
            m_scale = target_height / max(mh, 1)
            mask_display_w = max(int(mw * m_scale), 1)

            p1 = orig_rgb.resize((display_w, display_h), Image.Resampling.BILINEAR)
            p2 = mask_display_rgb.resize((mask_display_w, display_h), Image.Resampling.NEAREST)
            p3 = overlay_panel.resize((display_w, display_h), Image.Resampling.BILINEAR)

            pad = 10
            banner_h = 58 if dim_mismatch else 44
            canvas_w = display_w + mask_display_w + display_w + (pad * 4)
            canvas_h = display_h + (pad * 2) + banner_h

            bg_color = (45, 25, 25) if dim_mismatch else (30, 30, 35)
            canvas = Image.new("RGB", (canvas_w, canvas_h), color=bg_color)
            draw = ImageDraw.Draw(canvas)

            # Paste 3 panels
            canvas.paste(p1, (pad, banner_h + pad))
            canvas.paste(p2, (pad * 2 + display_w, banner_h + pad))
            canvas.paste(p3, (pad * 3 + display_w + mask_display_w, banner_h + pad))

            # Header lines (distinct and readable in contact sheet)
            line1 = f"{label_text}  |  Img: {w}x{h}  Mask: {mw}x{mh} ({mask_raw.mode})  |  [Original | Mask | Overlay]"
            draw.text((pad + 2, 4), line1, fill=(240, 240, 240))

            # Status line clearly distinguishing overlay type
            if fg_status == "PROVISIONAL" and can_generate_overlay:
                draw.text((pad + 2, 22), f"OVERLAY: {overlay_label}", fill=(255, 215, 0))  # Gold
            elif fg_status == "RELIABLE" and can_generate_overlay:
                draw.text((pad + 2, 22), f"OVERLAY: {overlay_label}", fill=(144, 238, 144))  # Light green
            else:
                draw.text((pad + 2, 22), f"OVERLAY: {overlay_label} (Colour-coded / Palette / Continuous)", fill=(255, 160, 160))  # Light red

            if dim_mismatch:
                warning_text = "CAUTION: DIMENSION MISMATCH; DISPLAY ONLY; NOT EVIDENCE OF VALID CLINICAL ALIGNMENT"
                draw.text((pad + 2, 38), warning_text, fill=(255, 100, 100))

            out_path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(out_path, format="JPEG", quality=90)
            return True

    except Exception as e:
        print(f"    [WARN] Preview generation failed for {label_text}: {e}", file=sys.stderr)
        return False


def generate_contact_sheet(
    preview_paths: List[Path],
    out_contact_sheet_path: Path,
    title: str,
) -> bool:
    """Combine preview panels into a multi-row contact sheet for easy visual audit."""
    if not preview_paths:
        return False
    try:
        images = [Image.open(p) for p in preview_paths if p.is_file()]
        if not images:
            return False

        max_w = max(img.width for img in images)
        total_h = sum(img.height for img in images) + (len(images) * 10) + 50

        sheet = Image.new("RGB", (max_w + 20, total_h), color=(20, 20, 25))
        draw = ImageDraw.Draw(sheet)
        draw.text((20, 15), title, fill=(255, 255, 255))

        curr_y = 50
        for img in images:
            sheet.paste(img, (10, curr_y))
            curr_y += img.height + 10

        out_contact_sheet_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out_contact_sheet_path, format="JPEG", quality=85)
        return True
    except Exception as e:
        print(f"    [WARN] Contact sheet generation failed: {e}", file=sys.stderr)
        return False


def compute_preview_sample_indices(total_count: int, sample_size: int = 10) -> Set[int]:
    """
    Safely determine deterministic preview sample indices across different cohort sizes:
      - total_count == 0: returns empty set
      - total_count == 1: returns {0}
      - 1 < total_count <= sample_size: returns all indices set(range(total_count))
      - total_count > sample_size: returns deterministic stride indices of size sample_size
    """
    if total_count <= 0:
        return set()
    if total_count == 1:
        return {0}
    if total_count <= sample_size:
        return set(range(total_count))
    return set([int(i * (total_count - 1) / (sample_size - 1)) for i in range(sample_size)])


# ---------------------------------------------------------------------------
# Safe Output Writing & Rollback Utilities
# ---------------------------------------------------------------------------

def compute_file_sha256(path: Path) -> str:
    """Calculate SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_directory_backup(orig_dir: Path, bkp_dir: Path) -> bool:
    """
    Recursively verify that bkp_dir is a byte-identical copy of orig_dir.
    Compares:
      1. Relative directory paths (including empty subdirectories).
      2. Relative file paths and file counts.
      3. File sizes and SHA-256 hashes.
    """
    if not orig_dir.is_dir() or not bkp_dir.is_dir():
        return False

    # 1. Compare all relative directory paths (including empty subdirectories)
    orig_dirs = {p.relative_to(orig_dir) for p in orig_dir.rglob("*") if p.is_dir()}
    bkp_dirs = {p.relative_to(bkp_dir) for p in bkp_dir.rglob("*") if p.is_dir()}
    if orig_dirs != bkp_dirs:
        return False

    # 2. Compare all relative file paths
    orig_files = {p.relative_to(orig_dir): p for p in orig_dir.rglob("*") if p.is_file()}
    bkp_files = {p.relative_to(bkp_dir): p for p in bkp_dir.rglob("*") if p.is_file()}

    if set(orig_files.keys()) != set(bkp_files.keys()):
        return False

    # 3. Compare file sizes and SHA-256 hashes
    for rel_path, o_path in orig_files.items():
        b_path = bkp_files[rel_path]
        if o_path.stat().st_size != b_path.stat().st_size:
            return False
        if compute_file_sha256(o_path) != compute_file_sha256(b_path):
            return False

    return True


# ---------------------------------------------------------------------------
# Main Audit Execution
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 1C: Image-Level Integrity, Mask Quality and Cross-Dataset Overlap Audit."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly permit overwriting existing Stage 1C manifests or preview outputs.",
    )
    args = parser.parse_args()

    project_root = PROJECT_ROOT.resolve()

    # -------------------------------------------------------------------------
    # Protect EVERY planned output path (including empty dirs)
    # -------------------------------------------------------------------------
    existing_outputs = [p for p in ALL_STAGE_1C_OUTPUTS if p.exists()]

    if existing_outputs and not args.force:
        print("=" * 80, file=sys.stderr)
        print("[SAFEGUARD] Existing Stage 1C output(s) detected:", file=sys.stderr)
        for p in existing_outputs:
            print(f"  - {p.relative_to(project_root) if p.is_relative_to(project_root) else p}", file=sys.stderr)
        print("Operation stopped to protect existing records. Use --force if overwrite is explicitly approved.", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        return 1

    if existing_outputs and args.force:
        print("[SAFEGUARD WARNING] --force supplied. The following existing Stage 1C output(s) will be replaced:")
        for p in existing_outputs:
            print(f"  - {p.relative_to(project_root) if p.is_relative_to(project_root) else p}")

    print("=" * 80)
    print("Stage 1C: Image-Level Integrity, Mask Quality and Cross-Dataset Overlap Audit")
    print(f"Executed at : {datetime.now(timezone.utc).isoformat()} UTC")
    print(f"Project Root: {project_root}")
    print("Mode        : In-memory stream decoding (zero extraction of dataset archives)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Archive Existence & Stage 1B Reference Hash Verification (Requirement 2)
    # -------------------------------------------------------------------------
    archive_runtime_hashes: Dict[str, str] = {}
    for d_key, a_path in TARGET_ARCHIVES.items():
        if not a_path.is_file():
            print(f"[FATAL INTEGRITY ERROR] Expected archive not found: {a_path}", file=sys.stderr)
            return 1
        print(f"Computing SHA-256 for {d_key} archive ({a_path.name})...")
        observed_hash = compute_file_sha256(a_path)
        archive_runtime_hashes[d_key] = observed_hash
        expected_hash = REFERENCE_ARCHIVE_SHA256[d_key]

        if observed_hash != expected_hash:
            print("=" * 80, file=sys.stderr)
            print(f"[FATAL INTEGRITY ERROR] SHA-256 mismatch for {d_key} archive!", file=sys.stderr)
            print(f"  Archive Path : {a_path}", file=sys.stderr)
            print(f"  Expected Hash: {expected_hash}", file=sys.stderr)
            print(f"  Observed Hash: {observed_hash}", file=sys.stderr)
            print("Aborting before opening ZIP entries or decoding images. Existing records untouched.", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            return 1

        print(f"  {d_key} SHA-256: {observed_hash} (VERIFIED against reference)")

    # Accumulators
    all_image_records: List[dict] = []
    all_mask_records: List[dict] = []
    all_pair_records: List[dict] = []
    excluded_non_image_entries: Dict[str, List[str]] = defaultdict(list)
    malformed_filename_entries: Dict[str, List[str]] = defaultdict(list)

    # Hashes for exact duplicates: key: sha256 -> list of occurrences
    image_hash_registry: Dict[str, List[dict]] = defaultdict(list)
    mask_hash_registry: Dict[str, List[dict]] = defaultdict(list)
    perceptual_image_list: List[dict] = []

    # Dynamic tracking of AZH challenge test count
    azh_challenge_test_image_count = 0

    # Dynamic tracking of AZH nested patches archive metadata
    azh_nested_zip_metadata: Optional[dict] = None

    # Temporary staging directory for all outputs
    temp_staging_dir = Path(tempfile.mkdtemp(prefix="stage_1c_staging_", dir=PROJECT_ROOT / "results" / "temporary"))
    temp_visual_samples_dir = temp_staging_dir / "visual_samples"
    temp_visual_samples_dir.mkdir(parents=True, exist_ok=True)
    temp_preview_paths: List[Path] = []

    try:
        # =========================================================================
        # AUDIT 1: AZH ARCHIVE
        # =========================================================================
        print("\n[1/3] Auditing AZH Archive: wound-segmentation-master.zip...")
        azh_path = TARGET_ARCHIVES["AZH"]
        with zipfile.ZipFile(azh_path, mode="r") as zf:
            namelist = zf.namelist()

            azh_subsets = [
                {
                    "name": "AZH_Challenge_Train",
                    "img_prefix": "wound-segmentation-master/data/Foot Ulcer Segmentation Challenge/train/images",
                    "mask_prefix": "wound-segmentation-master/data/Foot Ulcer Segmentation Challenge/train/labels",
                    "is_labelled": True,
                },
                {
                    "name": "AZH_Challenge_Val",
                    "img_prefix": "wound-segmentation-master/data/Foot Ulcer Segmentation Challenge/validation/images",
                    "mask_prefix": "wound-segmentation-master/data/Foot Ulcer Segmentation Challenge/validation/labels",
                    "is_labelled": True,
                },
                {
                    "name": "AZH_Medetec_Train",
                    "img_prefix": "wound-segmentation-master/data/Medetec_foot_ulcer_224/train/images",
                    "mask_prefix": "wound-segmentation-master/data/Medetec_foot_ulcer_224/train/labels",
                    "is_labelled": True,
                },
                {
                    "name": "AZH_Medetec_Test",
                    "img_prefix": "wound-segmentation-master/data/Medetec_foot_ulcer_224/test/images",
                    "mask_prefix": "wound-segmentation-master/data/Medetec_foot_ulcer_224/test/labels",
                    "is_labelled": True,
                },
                {
                    "name": "AZH_Challenge_Test",
                    "img_prefix": "wound-segmentation-master/data/Foot Ulcer Segmentation Challenge/test/images",
                    "mask_prefix": None,
                    "is_labelled": False,
                },
            ]

            for s_idx, subset in enumerate(azh_subsets, 1):
                sub_name = subset["name"]
                img_prefix = subset["img_prefix"]
                mask_prefix = subset["mask_prefix"]
                is_labelled = subset["is_labelled"]

                raw_img_entries = [n for n in namelist if n.startswith(img_prefix + "/") and not n.endswith("/")]
                img_files = []
                for entry in raw_img_entries:
                    ext = Path(entry).suffix.lower()
                    if ext in IMAGE_EXTENSIONS:
                        img_files.append(entry)
                    else:
                        excluded_non_image_entries["AZH"].append(entry)
                img_files.sort()

                if sub_name == "AZH_Challenge_Test":
                    azh_challenge_test_image_count = len(img_files)

                print(f"  ({s_idx}.{sub_name}) Auditing {len(img_files)} valid image files (excluded {len(raw_img_entries) - len(img_files)} non-images)...")

                mask_files_by_stem: Dict[str, str] = {}
                mask_stem_duplicates: Dict[str, List[str]] = defaultdict(list)
                if is_labelled and mask_prefix:
                    raw_mask_entries = [n for n in namelist if n.startswith(mask_prefix + "/") and not n.endswith("/")]
                    valid_mask_entries = []
                    for entry in raw_mask_entries:
                        ext = Path(entry).suffix.lower()
                        if ext in IMAGE_EXTENSIONS:
                            valid_mask_entries.append(entry)
                        else:
                            excluded_non_image_entries["AZH"].append(entry)

                    for m_path in valid_mask_entries:
                        stem = Path(m_path).stem
                        mask_stem_duplicates[stem].append(m_path)

                    for stem, paths in mask_stem_duplicates.items():
                        if len(paths) == 1:
                            mask_files_by_stem[stem] = paths[0]
                        else:
                            print(f"    [WARN] Duplicate mask stem '{stem}' detected in {sub_name}: {paths}", file=sys.stderr)

                preview_indices = compute_preview_sample_indices(len(img_files), sample_size=8)

                for idx, img_path in enumerate(img_files):
                    img_bytes = zf.read(img_path)
                    img_audit = inspect_image_bytes(img_bytes, img_path, "AZH")
                    img_audit["subset"] = sub_name
                    all_image_records.append(img_audit)

                    if img_audit["decode_success"]:
                        sha = img_audit["sha256"]
                        image_hash_registry[sha].append({
                            "dataset_key": "AZH",
                            "subset": sub_name,
                            "path": img_path,
                            "width": img_audit["width"],
                            "height": img_audit["height"],
                            "dhash": img_audit["dhash"],
                        })
                        perceptual_image_list.append({
                            "dataset_key": "AZH",
                            "subset": sub_name,
                            "cohort": f"AZH::{sub_name}",
                            "path": img_path,
                            "sha256": sha,
                            "dhash_int": int(img_audit["dhash"], 16),
                            "width": img_audit["width"],
                            "height": img_audit["height"],
                        })

                    if is_labelled and mask_prefix:
                        stem = Path(img_path).stem
                        pair_record = {
                            "dataset_key": "AZH",
                            "subset": sub_name,
                            "pair_id": stem,
                            "image_path": img_path,
                            "mask_path": None,
                            "pair_status": "PENDING",
                            "image_width": img_audit["width"],
                            "image_height": img_audit["height"],
                            "mask_width": None,
                            "mask_height": None,
                            "dimensions_match": False,
                            "mask_type": None,
                            "foreground_estimation_status": None,
                            "mask_foreground_pct": None,
                            "requires_visual_review": False,
                            "review_reason": None,
                        }

                        conflicting_masks = mask_stem_duplicates.get(stem, [])
                        if len(conflicting_masks) > 1:
                            pair_record["pair_status"] = "DUPLICATE_MASK_STEMS_CONFLICT"
                            pair_record["requires_visual_review"] = True
                            pair_record["review_reason"] = f"Multiple conflicting mask paths for stem '{stem}': {conflicting_masks}"
                            pair_record["mask_path"] = ";".join(conflicting_masks)
                        elif len(conflicting_masks) == 0:
                            pair_record["pair_status"] = "MASK_MISSING"
                            pair_record["requires_visual_review"] = True
                            pair_record["review_reason"] = f"No mask found for image stem '{stem}'"
                        else:
                            mask_path = conflicting_masks[0]
                            pair_record["mask_path"] = mask_path

                            mask_bytes = zf.read(mask_path)
                            mask_audit = inspect_mask_bytes(mask_bytes, mask_path, "AZH")
                            mask_audit["subset"] = sub_name
                            all_mask_records.append(mask_audit)

                            if mask_audit["decode_success"]:
                                mask_hash_registry[mask_audit["sha256"]].append({
                                    "dataset_key": "AZH",
                                    "subset": sub_name,
                                    "path": mask_path,
                                    "width": mask_audit["width"],
                                    "height": mask_audit["height"],
                                })

                            pair_record["mask_width"] = mask_audit["width"]
                            pair_record["mask_height"] = mask_audit["height"]
                            pair_record["mask_type"] = mask_audit["mask_type"]
                            pair_record["foreground_estimation_status"] = mask_audit["foreground_estimation_status"]
                            pair_record["mask_foreground_pct"] = mask_audit["foreground_percentage_estimated"]

                            if not img_audit["decode_success"] and not mask_audit["decode_success"]:
                                pair_record["pair_status"] = "BOTH_DECODE_FAILED"
                                pair_record["requires_visual_review"] = True
                                pair_record["review_reason"] = "Both image and mask failed Pillow decoding"
                            elif not img_audit["decode_success"]:
                                pair_record["pair_status"] = "IMAGE_DECODE_FAILED"
                                pair_record["requires_visual_review"] = True
                                pair_record["review_reason"] = f"Image decode failed: {img_audit['error_message']}"
                            elif not mask_audit["decode_success"]:
                                pair_record["pair_status"] = "MASK_DECODE_FAILED"
                                pair_record["requires_visual_review"] = True
                                pair_record["review_reason"] = f"Mask decode failed: {mask_audit['error_message']}"
                            elif (img_audit["width"], img_audit["height"]) != (mask_audit["width"], mask_audit["height"]):
                                pair_record["pair_status"] = "DIMENSION_MISMATCH"
                                pair_record["dimensions_match"] = False
                                pair_record["requires_visual_review"] = True
                                pair_record["review_reason"] = f"Dimension mismatch: Img ({img_audit['width']}x{img_audit['height']}) vs Mask ({mask_audit['width']}x{mask_audit['height']})"
                            else:
                                pair_record["dimensions_match"] = True
                                if mask_audit["requires_visual_review"]:
                                    pair_record["pair_status"] = "MASK_REQUIRES_REVIEW"
                                    pair_record["requires_visual_review"] = True
                                    pair_record["review_reason"] = mask_audit["review_reason"]
                                else:
                                    pair_record["pair_status"] = "VALID_DECODE_AND_DIMENSIONS"

                            if idx in preview_indices and img_audit["decode_success"] and mask_audit["decode_success"]:
                                prev_name = f"AZH_{sub_name}_{stem}.jpg"
                                prev_out = temp_visual_samples_dir / "AZH" / prev_name
                                if generate_pair_preview(img_bytes, mask_bytes, prev_out, f"AZH [{sub_name}] {stem}", mask_audit):
                                    temp_preview_paths.append(prev_out)

                        all_pair_records.append(pair_record)

            # ---------------------------------------------------------------------
            # 1B. Nested Archive Inspection: azh_wound_care_center_dataset_patches.zip
            # In-memory inspection via io.BytesIO and zipfile.ZipFile (zero disk extraction)
            # Distinct Cohorts: AZH::WoundCareCenter_Patches_Train, AZH::WoundCareCenter_Patches_Test
            # ---------------------------------------------------------------------
            nested_patch_zip_entry = "wound-segmentation-master/data/wound_dataset/azh_wound_care_center_dataset_patches.zip"
            if nested_patch_zip_entry in namelist:
                print(f"  (Nested Archive) Auditing {nested_patch_zip_entry} in-memory...")
                nested_info = zf.getinfo(nested_patch_zip_entry)
                nested_zip_bytes = zf.read(nested_patch_zip_entry)
                nested_sha256 = hashlib.sha256(nested_zip_bytes).hexdigest()

                with zipfile.ZipFile(io.BytesIO(nested_zip_bytes), mode="r") as nzf:
                    nested_namelist = nzf.namelist()

                    # Dynamic per-split metrics based on verified stem matching
                    nested_splits_summary = {}
                    for split_name, s_img_pfx, s_lbl_pfx in [("train", "train/images/", "train/labels/"), ("test", "test/images/", "test/labels/")]:
                        raw_imgs = [n for n in nested_namelist if n.startswith(s_img_pfx) and not n.endswith("/")]
                        raw_lbls = [n for n in nested_namelist if n.startswith(s_lbl_pfx) and not n.endswith("/")]

                        supp_imgs = [n for n in raw_imgs if Path(n).suffix.lower() in IMAGE_EXTENSIONS]
                        supp_lbls = [n for n in raw_lbls if Path(n).suffix.lower() in IMAGE_EXTENSIONS]

                        img_stems_list = [Path(n).stem for n in supp_imgs]
                        lbl_stems_list = [Path(n).stem for n in supp_lbls]

                        img_stem_counts = Counter(img_stems_list)
                        lbl_stem_counts = Counter(lbl_stems_list)

                        uniq_img_stems = set(img_stems_list)
                        uniq_lbl_stems = set(lbl_stems_list)

                        matched_stems = sorted(list(uniq_img_stems.intersection(uniq_lbl_stems)))
                        imgs_no_mask = sorted(list(uniq_img_stems - uniq_lbl_stems))
                        masks_no_img = sorted(list(uniq_lbl_stems - uniq_img_stems))

                        dup_img_stems = sorted([s for s, c in img_stem_counts.items() if c > 1])
                        dup_lbl_stems = sorted([s for s, c in lbl_stem_counts.items() if c > 1])

                        # A verified unique stem-matched pair requires matching stem with exactly one image and one mask
                        unique_stem_matched_pairs = [s for s in matched_stems if img_stem_counts[s] == 1 and lbl_stem_counts[s] == 1]

                        nested_splits_summary[split_name] = {
                            "supported_image_files_count": len(supp_imgs),
                            "supported_mask_files_count": len(supp_lbls),
                            "unique_image_stems_count": len(uniq_img_stems),
                            "unique_mask_stems_count": len(uniq_lbl_stems),
                            "matched_image_mask_stems_count": len(matched_stems),
                            "unique_image_stems_without_matching_mask_count": len(imgs_no_mask),
                            "unique_mask_stems_without_matching_image_count": len(masks_no_img),
                            "duplicate_image_stems_count": len(dup_img_stems),
                            "duplicate_mask_stems_count": len(dup_lbl_stems),
                            "total_unique_stem_matched_pairs": len(unique_stem_matched_pairs),
                        }

                    total_unique_stem_matched = sum(s["total_unique_stem_matched_pairs"] for s in nested_splits_summary.values())
                    total_candidate_imgs = sum(s["supported_image_files_count"] for s in nested_splits_summary.values())
                    total_candidate_masks = sum(s["supported_mask_files_count"] for s in nested_splits_summary.values())

                    azh_nested_zip_metadata = {
                        "entry_path": nested_patch_zip_entry,
                        "uncompressed_size_bytes": nested_info.file_size,
                        "compressed_size_bytes": nested_info.compress_size,
                        "crc32": f"0x{nested_info.CRC:08x}",
                        "sha256": nested_sha256,
                        "total_entries": len(nested_namelist),
                        "total_supported_image_files": total_candidate_imgs,
                        "total_supported_mask_files": total_candidate_masks,
                        "total_unique_stem_matched_pairs": total_unique_stem_matched,
                        "scientific_note": (
                            "total_unique_stem_matched_pairs reflects filename-level stem correspondence only. "
                            "Actual decoding integrity, dimension agreement, and mask-quality metrics are evaluated "
                            "separately during the full pair inspection loop."
                        ),
                        "splits": nested_splits_summary,
                    }

                    patch_subsets = [
                        {
                            "name": "AZH_WCC_Patches_Train",
                            "cohort": "AZH::WoundCareCenter_Patches_Train",
                            "img_prefix": "train/images",
                            "mask_prefix": "train/labels",
                        },
                        {
                            "name": "AZH_WCC_Patches_Test",
                            "cohort": "AZH::WoundCareCenter_Patches_Test",
                            "img_prefix": "test/images",
                            "mask_prefix": "test/labels",
                        },
                    ]

                    for p_subset in patch_subsets:
                        p_sub_name = p_subset["name"]
                        p_cohort = p_subset["cohort"]
                        p_img_prefix = p_subset["img_prefix"]
                        p_mask_prefix = p_subset["mask_prefix"]

                        raw_p_img_entries = [n for n in nested_namelist if n.startswith(p_img_prefix + "/") and not n.endswith("/")]
                        p_img_files = []
                        for entry in raw_p_img_entries:
                            ext = Path(entry).suffix.lower()
                            if ext in IMAGE_EXTENSIONS:
                                p_img_files.append(entry)
                            else:
                                excluded_non_image_entries["AZH"].append(f"{nested_patch_zip_entry}::{entry}")
                        p_img_files.sort()

                        print(f"    ({p_sub_name}) Auditing {len(p_img_files)} valid patch image files in-memory...")

                        p_mask_files_by_stem: Dict[str, str] = {}
                        p_mask_stem_duplicates: Dict[str, List[str]] = defaultdict(list)
                        raw_p_mask_entries = [n for n in nested_namelist if n.startswith(p_mask_prefix + "/") and not n.endswith("/")]
                        valid_p_mask_entries = []
                        for entry in raw_p_mask_entries:
                            ext = Path(entry).suffix.lower()
                            if ext in IMAGE_EXTENSIONS:
                                valid_p_mask_entries.append(entry)
                            else:
                                excluded_non_image_entries["AZH"].append(f"{nested_patch_zip_entry}::{entry}")

                        for m_path in valid_p_mask_entries:
                            stem = Path(m_path).stem
                            p_mask_stem_duplicates[stem].append(m_path)

                        for stem, paths in p_mask_stem_duplicates.items():
                            if len(paths) == 1:
                                p_mask_files_by_stem[stem] = paths[0]
                            else:
                                print(f"      [WARN] Duplicate patch mask stem '{stem}' detected in {p_sub_name}: {paths}", file=sys.stderr)

                        p_preview_indices = compute_preview_sample_indices(len(p_img_files), sample_size=8)

                        for idx, img_path in enumerate(p_img_files):
                            img_bytes = nzf.read(img_path)
                            composite_img_path = f"{nested_patch_zip_entry}::{img_path}"
                            img_audit = inspect_image_bytes(img_bytes, composite_img_path, "AZH")
                            img_audit["subset"] = p_sub_name
                            all_image_records.append(img_audit)

                            if img_audit["decode_success"]:
                                sha = img_audit["sha256"]
                                image_hash_registry[sha].append({
                                    "dataset_key": "AZH",
                                    "subset": p_sub_name,
                                    "path": composite_img_path,
                                    "width": img_audit["width"],
                                    "height": img_audit["height"],
                                    "dhash": img_audit["dhash"],
                                })
                                perceptual_image_list.append({
                                    "dataset_key": "AZH",
                                    "subset": p_sub_name,
                                    "cohort": p_cohort,
                                    "path": composite_img_path,
                                    "sha256": sha,
                                    "dhash_int": int(img_audit["dhash"], 16),
                                    "width": img_audit["width"],
                                    "height": img_audit["height"],
                                })

                            stem = Path(img_path).stem
                            pair_record = {
                                "dataset_key": "AZH",
                                "subset": p_sub_name,
                                "pair_id": stem,
                                "image_path": composite_img_path,
                                "mask_path": None,
                                "pair_status": "PENDING",
                                "image_width": img_audit["width"],
                                "image_height": img_audit["height"],
                                "mask_width": None,
                                "mask_height": None,
                                "dimensions_match": False,
                                "mask_type": None,
                                "foreground_estimation_status": None,
                                "mask_foreground_pct": None,
                                "requires_visual_review": False,
                                "review_reason": None,
                            }

                            conflicting_masks = p_mask_stem_duplicates.get(stem, [])
                            if len(conflicting_masks) > 1:
                                pair_record["pair_status"] = "DUPLICATE_MASK_STEMS_CONFLICT"
                                pair_record["requires_visual_review"] = True
                                pair_record["review_reason"] = f"Multiple conflicting mask paths for stem '{stem}': {conflicting_masks}"
                                pair_record["mask_path"] = ";".join(conflicting_masks)
                            elif len(conflicting_masks) == 0:
                                pair_record["pair_status"] = "MASK_MISSING"
                                pair_record["requires_visual_review"] = True
                                pair_record["review_reason"] = f"No mask found for patch image stem '{stem}'"
                            else:
                                mask_path = conflicting_masks[0]
                                composite_mask_path = f"{nested_patch_zip_entry}::{mask_path}"
                                pair_record["mask_path"] = composite_mask_path

                                mask_bytes = nzf.read(mask_path)
                                mask_audit = inspect_mask_bytes(mask_bytes, composite_mask_path, "AZH")
                                mask_audit["subset"] = p_sub_name
                                all_mask_records.append(mask_audit)

                                if mask_audit["decode_success"]:
                                    mask_hash_registry[mask_audit["sha256"]].append({
                                        "dataset_key": "AZH",
                                        "subset": p_sub_name,
                                        "path": composite_mask_path,
                                        "width": mask_audit["width"],
                                        "height": mask_audit["height"],
                                    })

                                pair_record["mask_width"] = mask_audit["width"]
                                pair_record["mask_height"] = mask_audit["height"]
                                pair_record["mask_type"] = mask_audit["mask_type"]
                                pair_record["foreground_estimation_status"] = mask_audit["foreground_estimation_status"]
                                pair_record["mask_foreground_pct"] = mask_audit["foreground_percentage_estimated"]

                                if not img_audit["decode_success"] and not mask_audit["decode_success"]:
                                    pair_record["pair_status"] = "BOTH_DECODE_FAILED"
                                    pair_record["requires_visual_review"] = True
                                    pair_record["review_reason"] = "Both patch image and mask failed Pillow decoding"
                                elif not img_audit["decode_success"]:
                                    pair_record["pair_status"] = "IMAGE_DECODE_FAILED"
                                    pair_record["requires_visual_review"] = True
                                    pair_record["review_reason"] = f"Patch image decode failed: {img_audit['error_message']}"
                                elif not mask_audit["decode_success"]:
                                    pair_record["pair_status"] = "MASK_DECODE_FAILED"
                                    pair_record["requires_visual_review"] = True
                                    pair_record["review_reason"] = f"Patch mask decode failed: {mask_audit['error_message']}"
                                elif (img_audit["width"], img_audit["height"]) != (mask_audit["width"], mask_audit["height"]):
                                    pair_record["pair_status"] = "DIMENSION_MISMATCH"
                                    pair_record["dimensions_match"] = False
                                    pair_record["requires_visual_review"] = True
                                    pair_record["review_reason"] = f"Dimension mismatch: Img ({img_audit['width']}x{img_audit['height']}) vs Mask ({mask_audit['width']}x{mask_audit['height']})"
                                else:
                                    pair_record["dimensions_match"] = True
                                    if mask_audit["requires_visual_review"]:
                                        pair_record["pair_status"] = "MASK_REQUIRES_REVIEW"
                                        pair_record["requires_visual_review"] = True
                                        pair_record["review_reason"] = mask_audit["review_reason"]
                                    else:
                                        pair_record["pair_status"] = "VALID_DECODE_AND_DIMENSIONS"

                                if idx in p_preview_indices and img_audit["decode_success"] and mask_audit["decode_success"]:
                                    prev_name = f"AZH_{p_sub_name}_{stem}.jpg"
                                    prev_out = temp_visual_samples_dir / "AZH" / prev_name
                                    if generate_pair_preview(img_bytes, mask_bytes, prev_out, f"AZH [{p_sub_name}] {stem}", mask_audit):
                                        temp_preview_paths.append(prev_out)

                            all_pair_records.append(pair_record)

        # =========================================================================
        # AUDIT 2: WSEG ARCHIVE
        # =========================================================================
        print("\n[2/3] Auditing WSEG Archive: wseg_dataset.zip...")
        wseg_path = TARGET_ARCHIVES["WSEG"]
        with zipfile.ZipFile(wseg_path, mode="r") as zf:
            namelist = zf.namelist()

            raw_sample_files = [n for n in namelist if n.startswith("wseg_dataset/sample/") and not n.endswith("/")]
            sample_files = []
            for entry in raw_sample_files:
                ext = Path(entry).suffix.lower()
                if ext in IMAGE_EXTENSIONS:
                    sample_files.append(entry)
                else:
                    excluded_non_image_entries["WSEG"].append(entry)
            sample_files.sort()

            print(f"  Auditing {len(sample_files)} valid sample/mask pairs in WSEG (excluded {len(raw_sample_files) - len(sample_files)} non-images)...")

            wseg_preview_indices = compute_preview_sample_indices(len(sample_files), sample_size=10)

            for idx, sample_path in enumerate(sample_files):
                fname = Path(sample_path).name
                mask_path = f"wseg_dataset/mask/{fname}"

                img_bytes = zf.read(sample_path)
                img_audit = inspect_image_bytes(img_bytes, sample_path, "WSEG")
                img_audit["subset"] = "wseg_dataset"
                all_image_records.append(img_audit)

                if img_audit["decode_success"]:
                    sha = img_audit["sha256"]
                    image_hash_registry[sha].append({
                        "dataset_key": "WSEG",
                        "subset": "wseg_dataset",
                        "path": sample_path,
                        "width": img_audit["width"],
                        "height": img_audit["height"],
                        "dhash": img_audit["dhash"],
                    })
                    perceptual_image_list.append({
                        "dataset_key": "WSEG",
                        "subset": "wseg_dataset",
                        "cohort": "WSEG",
                        "path": sample_path,
                        "sha256": sha,
                        "dhash_int": int(img_audit["dhash"], 16),
                        "width": img_audit["width"],
                        "height": img_audit["height"],
                    })

                pair_record = {
                    "dataset_key": "WSEG",
                    "subset": "wseg_dataset",
                    "pair_id": Path(fname).stem,
                    "image_path": sample_path,
                    "mask_path": mask_path,
                    "pair_status": "PENDING",
                    "image_width": img_audit["width"],
                    "image_height": img_audit["height"],
                    "mask_width": None,
                    "mask_height": None,
                    "dimensions_match": False,
                    "mask_type": None,
                    "foreground_estimation_status": None,
                    "mask_foreground_pct": None,
                    "requires_visual_review": False,
                    "review_reason": None,
                }

                try:
                    mask_bytes = zf.read(mask_path)
                    mask_audit = inspect_mask_bytes(mask_bytes, mask_path, "WSEG")
                    mask_audit["subset"] = "wseg_dataset"
                    all_mask_records.append(mask_audit)

                    if mask_audit["decode_success"]:
                        mask_hash_registry[mask_audit["sha256"]].append({
                            "dataset_key": "WSEG",
                            "subset": "wseg_dataset",
                            "path": mask_path,
                            "width": mask_audit["width"],
                            "height": mask_audit["height"],
                        })

                    pair_record["mask_width"] = mask_audit["width"]
                    pair_record["mask_height"] = mask_audit["height"]
                    pair_record["mask_type"] = mask_audit["mask_type"]
                    pair_record["foreground_estimation_status"] = mask_audit["foreground_estimation_status"]
                    pair_record["mask_foreground_pct"] = mask_audit["foreground_percentage_estimated"]

                    if not img_audit["decode_success"] and not mask_audit["decode_success"]:
                        pair_record["pair_status"] = "BOTH_DECODE_FAILED"
                        pair_record["requires_visual_review"] = True
                        pair_record["review_reason"] = "Both sample and mask failed decoding"
                    elif not img_audit["decode_success"]:
                        pair_record["pair_status"] = "IMAGE_DECODE_FAILED"
                        pair_record["requires_visual_review"] = True
                        pair_record["review_reason"] = img_audit["error_message"]
                    elif not mask_audit["decode_success"]:
                        pair_record["pair_status"] = "MASK_DECODE_FAILED"
                        pair_record["requires_visual_review"] = True
                        pair_record["review_reason"] = mask_audit["error_message"]
                    elif (img_audit["width"], img_audit["height"]) != (mask_audit["width"], mask_audit["height"]):
                        pair_record["pair_status"] = "DIMENSION_MISMATCH"
                        pair_record["dimensions_match"] = False
                        pair_record["requires_visual_review"] = True
                        pair_record["review_reason"] = f"Dimension mismatch: Img ({img_audit['width']}x{img_audit['height']}) vs Mask ({mask_audit['width']}x{mask_audit['height']})"
                    else:
                        pair_record["dimensions_match"] = True
                        if mask_audit["requires_visual_review"]:
                            pair_record["pair_status"] = "MASK_REQUIRES_REVIEW"
                            pair_record["requires_visual_review"] = True
                            pair_record["review_reason"] = mask_audit["review_reason"]
                        else:
                            pair_record["pair_status"] = "VALID_DECODE_AND_DIMENSIONS"

                    if idx in wseg_preview_indices and img_audit["decode_success"] and mask_audit["decode_success"]:
                        prev_name = f"WSEG_{Path(fname).stem}.jpg"
                        prev_out = temp_visual_samples_dir / "WSEG" / prev_name
                        if generate_pair_preview(img_bytes, mask_bytes, prev_out, f"WSEG {fname}", mask_audit):
                            temp_preview_paths.append(prev_out)

                except KeyError:
                    pair_record["pair_status"] = "MASK_MISSING"
                    pair_record["requires_visual_review"] = True
                    pair_record["review_reason"] = f"Mask entry missing from ZIP: {mask_path}"

                all_pair_records.append(pair_record)

        # =========================================================================
        # AUDIT 3: MENDELEY ARCHIVE
        # =========================================================================
        print("\n[3/3] Auditing Mendeley Archive: Wound Image Dataset.zip...")
        mendeley_path = TARGET_ARCHIVES["MENDELEY"]
        with zipfile.ZipFile(mendeley_path, mode="r") as zf:
            namelist = zf.namelist()

            raw_wound_main = [n for n in namelist if n.startswith("wound_main/") and not n.endswith("/")]
            wound_main_files = []
            for entry in raw_wound_main:
                ext = Path(entry).suffix.lower()
                if ext in IMAGE_EXTENSIONS:
                    wound_main_files.append(entry)
                else:
                    excluded_non_image_entries["MENDELEY"].append(entry)
            wound_main_files.sort()

            raw_nomal = [n for n in namelist if n.startswith("Nomal/") and not n.endswith("/")]
            nomal_files = []
            for entry in raw_nomal:
                ext = Path(entry).suffix.lower()
                if ext in IMAGE_EXTENSIONS:
                    nomal_files.append(entry)
                else:
                    excluded_non_image_entries["MENDELEY"].append(entry)
            nomal_files.sort()

            # Robust Strict Normalized Numeric-ID Mask Lookup across supported image extensions
            # Strict regex matching: exact stem match ^wound_mask-(\d+)$
            re_mask_id = re.compile(r"^wound_mask-(\d+)$", re.IGNORECASE)
            raw_wound_mask = [n for n in namelist if n.startswith("wound_mask/") and not n.endswith("/")]
            mendeley_masks_by_norm_id: Dict[str, List[str]] = defaultdict(list)
            for m_entry in raw_wound_mask:
                ext = Path(m_entry).suffix.lower()
                if ext in IMAGE_EXTENSIONS:
                    m_stem = Path(m_entry).stem
                    m_match = re_mask_id.match(m_stem)
                    if m_match:
                        raw_digits = m_match.group(1)
                        norm_id = str(int(raw_digits))
                        mendeley_masks_by_norm_id[norm_id].append(m_entry)
                    else:
                        malformed_filename_entries["MENDELEY"].append(m_entry)
                        print(f"    [WARN] Malformed Mendeley mask filename: '{m_entry}'", file=sys.stderr)
                else:
                    excluded_non_image_entries["MENDELEY"].append(m_entry)

            print(f"  Auditing {len(wound_main_files)} wound_main/mask pairs ({len(mendeley_masks_by_norm_id)} unique normalized mask IDs indexed) and {len(nomal_files)} healthy controls in Mendeley...")

            mendeley_preview_indices = compute_preview_sample_indices(len(wound_main_files), sample_size=10)
            # Strict regex matching: exact stem match ^wound_main-(\d+)$
            re_main_id = re.compile(r"^wound_main-(\d+)$", re.IGNORECASE)

            # Pre-pass: Index wound_main images by normalized numeric ID to detect duplicate image identifiers
            mendeley_images_by_norm_id: Dict[str, List[str]] = defaultdict(list)
            for main_path in wound_main_files:
                stem = Path(main_path).stem
                m = re_main_id.match(stem)
                if m:
                    norm_id = str(int(m.group(1)))
                    mendeley_images_by_norm_id[norm_id].append(main_path)

            # 3A. Wound Main & Masks
            for idx, main_path in enumerate(wound_main_files):
                stem = Path(main_path).stem
                m = re_main_id.match(stem)
                if m:
                    raw_id = m.group(1)
                    norm_id = str(int(raw_id))
                else:
                    malformed_filename_entries["MENDELEY"].append(main_path)
                    print(f"    [WARN] Malformed Mendeley main image filename: '{main_path}'", file=sys.stderr)
                    raw_id = None
                    norm_id = None

                img_bytes = zf.read(main_path)
                img_audit = inspect_image_bytes(img_bytes, main_path, "MENDELEY")
                img_audit["subset"] = "wound_main"
                all_image_records.append(img_audit)

                if img_audit["decode_success"]:
                    sha = img_audit["sha256"]
                    image_hash_registry[sha].append({
                        "dataset_key": "MENDELEY",
                        "subset": "wound_main",
                        "path": main_path,
                        "width": img_audit["width"],
                        "height": img_audit["height"],
                        "dhash": img_audit["dhash"],
                    })
                    perceptual_image_list.append({
                        "dataset_key": "MENDELEY",
                        "subset": "wound_main",
                        "cohort": "Mendeley::wound_main",
                        "path": main_path,
                        "sha256": sha,
                        "dhash_int": int(img_audit["dhash"], 16),
                        "width": img_audit["width"],
                        "height": img_audit["height"],
                    })

                pair_record = {
                    "dataset_key": "MENDELEY",
                    "subset": "wound_main",
                    "pair_id": raw_id,
                    "normalized_pair_id": norm_id,
                    "image_path": main_path,
                    "mask_path": None,
                    "pair_status": "PENDING",
                    "image_width": img_audit["width"],
                    "image_height": img_audit["height"],
                    "mask_width": None,
                    "mask_height": None,
                    "dimensions_match": False,
                    "mask_type": None,
                    "foreground_estimation_status": None,
                    "mask_foreground_pct": None,
                    "requires_visual_review": False,
                    "review_reason": None,
                }

                if norm_id is None:
                    pair_record["pair_status"] = "MALFORMED_IMAGE_FILENAME"
                    pair_record["requires_visual_review"] = True
                    pair_record["review_reason"] = f"Image filename stem '{stem}' does not match pattern ^wound_main-(\\d+)$"
                else:
                    conflicting_images = mendeley_images_by_norm_id.get(norm_id, [])
                    if len(conflicting_images) > 1:
                        pair_record["pair_status"] = "DUPLICATE_IMAGE_IDENTIFIERS_CONFLICT"
                        pair_record["requires_visual_review"] = True
                        pair_record["review_reason"] = (
                            f"Multiple wound_main images share normalized ID '{norm_id}': {conflicting_images}"
                        )
                    else:
                        candidate_masks = mendeley_masks_by_norm_id.get(norm_id, [])
                        if len(candidate_masks) == 0:
                            pair_record["pair_status"] = "MASK_MISSING"
                            pair_record["requires_visual_review"] = True
                            pair_record["review_reason"] = f"No mask found for normalized ID '{norm_id}' (raw: '{raw_id}')"
                        elif len(candidate_masks) > 1:
                            pair_record["pair_status"] = "DUPLICATE_MASK_IDENTIFIERS_CONFLICT"
                            pair_record["requires_visual_review"] = True
                            pair_record["review_reason"] = f"Multiple conflicting mask entries for normalized ID '{norm_id}': {candidate_masks}"
                            pair_record["mask_path"] = ";".join(candidate_masks)
                        else:
                            mask_path = candidate_masks[0]
                            pair_record["mask_path"] = mask_path
                            try:
                                mask_bytes = zf.read(mask_path)
                                mask_audit = inspect_mask_bytes(mask_bytes, mask_path, "MENDELEY")
                                mask_audit["subset"] = "wound_mask"
                                all_mask_records.append(mask_audit)

                                if mask_audit["decode_success"]:
                                    mask_hash_registry[mask_audit["sha256"]].append({
                                        "dataset_key": "MENDELEY",
                                        "subset": "wound_mask",
                                        "path": mask_path,
                                        "width": mask_audit["width"],
                                        "height": mask_audit["height"],
                                    })

                                pair_record["mask_width"] = mask_audit["width"]
                                pair_record["mask_height"] = mask_audit["height"]
                                pair_record["mask_type"] = mask_audit["mask_type"]
                                pair_record["foreground_estimation_status"] = mask_audit["foreground_estimation_status"]
                                pair_record["mask_foreground_pct"] = mask_audit["foreground_percentage_estimated"]

                                if not img_audit["decode_success"] and not mask_audit["decode_success"]:
                                    pair_record["pair_status"] = "BOTH_DECODE_FAILED"
                                    pair_record["requires_visual_review"] = True
                                    pair_record["review_reason"] = "Both wound_main and wound_mask failed decoding"
                                elif not img_audit["decode_success"]:
                                    pair_record["pair_status"] = "IMAGE_DECODE_FAILED"
                                    pair_record["requires_visual_review"] = True
                                    pair_record["review_reason"] = img_audit["error_message"]
                                elif not mask_audit["decode_success"]:
                                    pair_record["pair_status"] = "MASK_DECODE_FAILED"
                                    pair_record["requires_visual_review"] = True
                                    pair_record["review_reason"] = mask_audit["error_message"]
                                elif (img_audit["width"], img_audit["height"]) != (mask_audit["width"], mask_audit["height"]):
                                    pair_record["pair_status"] = "DIMENSION_MISMATCH"
                                    pair_record["dimensions_match"] = False
                                    pair_record["requires_visual_review"] = True
                                    pair_record["review_reason"] = f"Dimension mismatch: Img ({img_audit['width']}x{img_audit['height']}) vs Mask ({mask_audit['width']}x{mask_audit['height']})"
                                else:
                                    pair_record["dimensions_match"] = True
                                    if mask_audit["requires_visual_review"]:
                                        pair_record["pair_status"] = "MASK_REQUIRES_REVIEW"
                                        pair_record["requires_visual_review"] = True
                                        pair_record["review_reason"] = mask_audit["review_reason"]
                                    else:
                                        pair_record["pair_status"] = "VALID_DECODE_AND_DIMENSIONS"

                                if idx in mendeley_preview_indices and img_audit["decode_success"] and mask_audit["decode_success"]:
                                    prev_name = f"MENDELEY_wound_{raw_id}.jpg"
                                    prev_out = temp_visual_samples_dir / "MENDELEY" / prev_name
                                    if generate_pair_preview(img_bytes, mask_bytes, prev_out, f"Mendeley ID {raw_id}", mask_audit):
                                        temp_preview_paths.append(prev_out)

                            except KeyError:
                                pair_record["pair_status"] = "MASK_MISSING"
                                pair_record["requires_visual_review"] = True
                                pair_record["review_reason"] = f"Mask entry missing from ZIP: {mask_path}"

                all_pair_records.append(pair_record)

            # 3B. Nomal/ healthy control images (no masks)
            for nomal_path in nomal_files:
                img_bytes = zf.read(nomal_path)
                img_audit = inspect_image_bytes(img_bytes, nomal_path, "MENDELEY")
                img_audit["subset"] = "Nomal_Healthy_Controls"
                all_image_records.append(img_audit)

                if img_audit["decode_success"]:
                    sha = img_audit["sha256"]
                    image_hash_registry[sha].append({
                        "dataset_key": "MENDELEY",
                        "subset": "Nomal_Healthy_Controls",
                        "path": nomal_path,
                        "width": img_audit["width"],
                        "height": img_audit["height"],
                        "dhash": img_audit["dhash"],
                    })
                    perceptual_image_list.append({
                        "dataset_key": "MENDELEY",
                        "subset": "Nomal_Healthy_Controls",
                        "cohort": "Mendeley::healthy_controls",
                        "path": nomal_path,
                        "sha256": sha,
                        "dhash_int": int(img_audit["dhash"], 16),
                        "width": img_audit["width"],
                        "height": img_audit["height"],
                    })

        # =========================================================================
        # AUDIT 4: CROSS-DATASET & CROSS-COHORT DUPLICATE ANALYSIS
        # =========================================================================
        print("\n[4/4] Performing Cross-Dataset & Cross-Cohort Duplicate Analysis...")

        # 4A. Exact SHA-256 Duplicates across datasets or cohorts
        exact_cross_image_duplicates = []
        for sha, occurrences in image_hash_registry.items():
            datasets = {o["dataset_key"] for o in occurrences}
            cohorts = {f"{o['dataset_key']}::{o['subset']}" for o in occurrences}
            if len(datasets) > 1 or len(cohorts) > 1:
                exact_cross_image_duplicates.append({
                    "sha256": sha,
                    "datasets_involved": sorted(list(datasets)),
                    "cohorts_involved": sorted(list(cohorts)),
                    "total_occurrences": len(occurrences),
                    "occurrences": occurrences,
                })

        exact_cross_mask_duplicates = []
        for sha, occurrences in mask_hash_registry.items():
            datasets = {o["dataset_key"] for o in occurrences}
            cohorts = {f"{o['dataset_key']}::{o['subset']}" for o in occurrences}
            if len(datasets) > 1 or len(cohorts) > 1:
                exact_cross_mask_duplicates.append({
                    "sha256": sha,
                    "datasets_involved": sorted(list(datasets)),
                    "cohorts_involved": sorted(list(cohorts)),
                    "total_occurrences": len(occurrences),
                    "occurrences": occurrences,
                })

        # 4B. Perceptual Near-Duplicates via dHash (Hamming distance <= 6)
        perceptual_matches = []
        cohort_groups = defaultdict(list)
        for item in perceptual_image_list:
            cohort_groups[item["cohort"]].append(item)

        cohort_names = sorted(list(cohort_groups.keys()))
        print(f"  Comparing perceptual dHash across {len(cohort_names)} distinct cohorts:")
        for cn in cohort_names:
            print(f"    - {cn} ({len(cohort_groups[cn])} images)")

        for i in range(len(cohort_names)):
            for j in range(i + 1, len(cohort_names)):
                c1 = cohort_names[i]
                c2 = cohort_names[j]
                list1 = cohort_groups[c1]
                list2 = cohort_groups[c2]

                for item1 in list1:
                    h1 = item1["dhash_int"]
                    for item2 in list2:
                        h2 = item2["dhash_int"]
                        dist = hamming_distance(h1, h2)
                        if dist <= PERCEPTUAL_HAMMING_THRESHOLD:
                            # Exclude exact byte duplicates
                            if item1["sha256"] == item2["sha256"]:
                                continue

                            perceptual_matches.append({
                                "hamming_distance": dist,
                                "is_exact_byte_duplicate": False,
                                "candidate_classification": "PERCEPTUAL_NEAR_DUPLICATE_CANDIDATE",
                                "scientific_caveat": (
                                    f"dHash match with Hamming distance {dist} <= {PERCEPTUAL_HAMMING_THRESHOLD}. "
                                    "Non-identical byte content. DOES NOT constitute confirmed duplicate; requires manual visual verification."
                                ),
                                "dataset_1": item1["dataset_key"],
                                "subset_1": item1["subset"],
                                "cohort_1": item1["cohort"],
                                "path_1": item1["path"],
                                "dimensions_1": f"{item1['width']}x{item1['height']}",
                                "dataset_2": item2["dataset_key"],
                                "subset_2": item2["subset"],
                                "cohort_2": item2["cohort"],
                                "path_2": item2["path"],
                                "dimensions_2": f"{item2['width']}x{item2['height']}",
                            })

        print(f"  Exact cross-cohort image matches       : {len(exact_cross_image_duplicates)}")
        print(f"  Exact cross-cohort mask matches        : {len(exact_cross_mask_duplicates)}")
        print(f"  Perceptual near-duplicates (non-exact) : {len(perceptual_matches)} (Hamming threshold <= {PERCEPTUAL_HAMMING_THRESHOLD})")

        # Contact Sheet Generation with strict error handling
        print("\nGenerating contact sheets for manual visual review...")
        temp_contact_sheet = temp_staging_dir / "stage_1c_contact_sheet.jpg"
        cs_ok = generate_contact_sheet(temp_preview_paths, temp_contact_sheet, "Stage 1C: Representative Dataset Preview Samples")
        if not cs_ok or not temp_contact_sheet.is_file():
            print(f"[FATAL ERROR] Contact sheet generation failed at: {temp_contact_sheet}", file=sys.stderr)
            raise RuntimeError("Contact sheet generation failed. Stopping before finalization.")

        print(f"  Generated {len(temp_preview_paths)} preview images in temporary staging and contact sheet at {temp_contact_sheet}.")

        # =========================================================================
        # Reproducibility Metadata & Scientific Caveats
        # =========================================================================
        audit_timestamp_utc = datetime.now(timezone.utc).isoformat()
        reproducibility_metadata = {
            "title": "Stage 1C Audit Reproducibility Metadata",
            "audit_timestamp_utc": audit_timestamp_utc,
            "python_version": platform.python_version(),
            "pillow_version": PIL.__version__,
            "platform": platform.platform(),
            "perceptual_hash_algorithm": "difference_hash_64bit (dHash)",
            "perceptual_hamming_threshold": PERCEPTUAL_HAMMING_THRESHOLD,
            "perceptual_interpretation_rule": (
                f"Hamming distance <= {PERCEPTUAL_HAMMING_THRESHOLD} on non-identical byte files indicates "
                "a perceptual near-duplicate candidate; requires manual visual confirmation. "
                "Exact byte duplicates are tracked separately in exact_cross_cohort_image_duplicates."
            ),
            "reference_hash_verification": {
                d_key: {
                    "reference_sha256": REFERENCE_ARCHIVE_SHA256[d_key],
                    "runtime_sha256": archive_runtime_hashes[d_key],
                    "status": "VERIFIED_MATCH",
                }
                for d_key in TARGET_ARCHIVES
            },
            "foreground_interpretation_caveat": (
                "The script measures pixel-value distributions and applies documented encoding conventions. "
                "It cannot independently establish the clinical semantic meaning of foreground without dataset documentation or visual verification."
            ),
            "histogram_reporting_note": (
                "pixel_histogram_nonzero_bins records all non-zero histogram bins observed for each decoded mask. "
                "For mode P (palette-based) masks, bins represent palette index frequencies (histogram_semantic_type: PALETTE_INDICES), "
                "not grayscale intensities or RGB color values."
            ),
            "foreground_estimation_rules": {
                "STRICT_BINARY": "Observed binary encoding {0, 1}, {0, 255}, or {0, V}; foreground_estimation_status = RELIABLE (measures pixel distribution under convention; clinical validity unverified).",
                "BINARY_LIKE_JPEG_ARTEFACTS": "JPEG format with >90% pixel mass at 0 and 255; provisional threshold >= 128; foreground_estimation_status = PROVISIONAL.",
                "FULL_FOREGROUND": "Uniform nonzero value across all pixels; foreground_estimation_status = RELIABLE.",
                "EMPTY": "All pixels zero; foreground_estimation_status = RELIABLE.",
                "PALETTE_BASED": "Pillow mode P; foreground_estimation_status = UNAVAILABLE unless unambiguous binary palette mapping to monochrome is established.",
                "MULTICLASS_OR_DISCRETE": "<= 5 distinct values, non-binary; provisional nonzero threshold; foreground_estimation_status = PROVISIONAL.",
                "GRAYSCALE_CONTINUOUS": "> 5 values across grayscale; foreground percentage withheld; foreground_estimation_status = UNAVAILABLE.",
                "COLOR_CODED": "Non-identical RGB channels; arbitrary grayscale conversion prohibited; foreground_estimation_status = UNAVAILABLE.",
            },
            "preview_sampling_rule": "Deterministic stride sampling: AZH (8 samples/subset), WSEG (10 samples), Mendeley (10 samples).",
            "azh_challenge_test_image_count_dynamic": azh_challenge_test_image_count,
            "dataset_archives": {
                d_key: {
                    "path": str(a_path.relative_to(project_root)),
                    "size_bytes": a_path.stat().st_size,
                    "sha256": archive_runtime_hashes[d_key],
                }
                for d_key, a_path in TARGET_ARCHIVES.items()
            },
            "excluded_non_image_entries_count": {
                k: len(v) for k, v in excluded_non_image_entries.items()
            },
            "malformed_filename_entries_count": {
                k: len(v) for k, v in malformed_filename_entries.items()
            },
            "azh_nested_patches_zip_metadata": azh_nested_zip_metadata,
        }

        # =========================================================================
        # Staging All Outputs in Temp Directory
        # =========================================================================
        print("\nWriting output manifests via safe temporary staging...")

        temp_img_json = temp_staging_dir / "image_integrity_audit.json"
        temp_csv = temp_staging_dir / "image_mask_pair_audit.csv"
        temp_mask_json = temp_staging_dir / "mask_quality_summary.json"
        temp_dup_json = temp_staging_dir / "cross_dataset_duplicate_audit.json"
        temp_md = temp_staging_dir / "stage_1c_summary.md"

        # 1. Image Integrity JSON
        image_integrity_doc = {
            "metadata": {
                **reproducibility_metadata,
                "title": "Stage 1C: Image-Level Integrity Audit Report",
                "total_images_inspected": len(all_image_records),
                "decoding_success_count": sum(1 for r in all_image_records if r["decode_success"]),
                "decoding_failed_count": sum(1 for r in all_image_records if not r["decode_success"]),
                "scientific_disclaimer": "Recorded from in-memory Pillow decoding of raw archive bytes without full extraction.",
            },
            "images": all_image_records,
            "excluded_non_image_entries": excluded_non_image_entries,
            "malformed_filename_entries": malformed_filename_entries,
        }
        with open(temp_img_json, "w", encoding="utf-8") as f:
            json.dump(image_integrity_doc, f, indent=2)

        # 2. Pair Audit CSV
        with open(temp_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "dataset_key", "subset", "pair_id", "normalized_pair_id", "pair_status", "dimensions_match",
                "image_width", "image_height", "mask_width", "mask_height",
                "mask_type", "foreground_estimation_status", "mask_foreground_pct",
                "requires_visual_review", "review_reason", "image_path", "mask_path"
            ])
            writer.writeheader()
            for p in all_pair_records:
                writer.writerow(p)

        # 3. Mask Quality JSON
        mask_quality_doc = {
            "metadata": {
                **reproducibility_metadata,
                "title": "Stage 1C: Mask Quality Summary Report",
                "total_masks_inspected": len(all_mask_records),
                "mask_type_distribution": dict(Counter(m["mask_type"] for m in all_mask_records)),
                "foreground_status_distribution": dict(Counter(m["foreground_estimation_status"] for m in all_mask_records)),
                "completely_empty_masks_count": sum(1 for m in all_mask_records if m["is_completely_empty"]),
                "completely_full_masks_count": sum(1 for m in all_mask_records if m["is_completely_full"]),
                "suspicious_small_foreground_count": sum(1 for m in all_mask_records if m["is_suspicious_small_foreground"]),
                "requires_visual_review_count": sum(1 for m in all_mask_records if m["requires_visual_review"]),
            },
            "masks": all_mask_records,
        }
        with open(temp_mask_json, "w", encoding="utf-8") as f:
            json.dump(mask_quality_doc, f, indent=2)

        # 4. Duplicate JSON
        duplicates_doc = {
            "metadata": {
                **reproducibility_metadata,
                "title": "Stage 1C: Cross-Dataset & Cross-Cohort Duplicate Audit Report",
                "exact_cross_cohort_image_duplicates_count": len(exact_cross_image_duplicates),
                "exact_cross_cohort_mask_duplicates_count": len(exact_cross_mask_duplicates),
                "perceptual_near_duplicate_candidates_count": len(perceptual_matches),
            },
            "exact_cross_cohort_image_duplicates": exact_cross_image_duplicates,
            "exact_cross_cohort_mask_duplicates": exact_cross_mask_duplicates,
            "perceptual_near_duplicate_candidates": perceptual_matches,
        }
        with open(temp_dup_json, "w", encoding="utf-8") as f:
            json.dump(duplicates_doc, f, indent=2)

        # 5. Markdown Summary Report
        md_summary = f"""# Stage 1C Audit Summary: Image Integrity, Mask Quality & Cross-Dataset Overlap

**Execution Date:** {audit_timestamp_utc} UTC  
**Project Root:** `{project_root}`  
**Method:** In-memory stream decoding with Pillow and perceptual dHash (zero dataset disk extraction).  
**Python Version:** {platform.python_version()} | **Pillow Version:** {PIL.__version__}

> **Clinical Interpretation Notice:**  
> The script measures pixel-value distributions and applies documented encoding conventions. It cannot independently establish the clinical semantic meaning of foreground without dataset documentation or visual verification.

---

## 1. Archive Hashes & Reference Verification
| Dataset | Archive Path | Size (Bytes) | SHA-256 Hash | Verification vs Reference |
| :--- | :--- | :--- | :--- | :--- |
| **AZH** | `{TARGET_ARCHIVES['AZH'].relative_to(project_root)}` | {TARGET_ARCHIVES['AZH'].stat().st_size} | `{archive_runtime_hashes['AZH']}` | **MATCH** (`{REFERENCE_ARCHIVE_SHA256['AZH'][:16]}...`) |
| **WSEG** | `{TARGET_ARCHIVES['WSEG'].relative_to(project_root)}` | {TARGET_ARCHIVES['WSEG'].stat().st_size} | `{archive_runtime_hashes['WSEG']}` | **MATCH** (`{REFERENCE_ARCHIVE_SHA256['WSEG'][:16]}...`) |
| **MENDELEY** | `{TARGET_ARCHIVES['MENDELEY'].relative_to(project_root)}` | {TARGET_ARCHIVES['MENDELEY'].stat().st_size} | `{archive_runtime_hashes['MENDELEY']}` | **MATCH** (`{REFERENCE_ARCHIVE_SHA256['MENDELEY'][:16]}...`) |

---

## 2. File & Decoding Overview

- **Total Images Inspected:** {len(all_image_records)}
  - Successfully Decoded: {sum(1 for r in all_image_records if r["decode_success"])}
  - Failed Decodings: {sum(1 for r in all_image_records if not r["decode_success"])}
- **Total Candidate Masks Inspected:** {len(all_mask_records)}
  - Successfully Decoded: {sum(1 for r in all_mask_records if r["decode_success"])}
  - Failed Decodings: {sum(1 for r in all_mask_records if not r["decode_success"])}
- **Total Labelled Candidate Pairs Evaluated:** {len(all_pair_records)}
- **Excluded Non-Image Entries:** {sum(len(v) for v in excluded_non_image_entries.values())} total entries filtered by extension.
- **Malformed Filename Entries:** {sum(len(v) for v in malformed_filename_entries.values())} image/mask filenames with malformed naming conventions (disqualified from pair matching).

---

## 3. Image-Mask Alignment & Pair Status Breakdown

| Pair Status | Count | Percentage |
| :--- | :--- | :--- |
"""
        pair_status_counts = Counter(p["pair_status"] for p in all_pair_records)
        for status, cnt in pair_status_counts.most_common():
            pct = round(cnt / len(all_pair_records) * 100, 2) if all_pair_records else 0
            md_summary += f"| `{status}` | {cnt} | {pct}% |\n"

        md_summary += f"""
- **Dimension Agreement:** {sum(1 for p in all_pair_records if p['dimensions_match'])} pairs share identical width and height.
- **Dimension Mismatches:** {sum(1 for p in all_pair_records if not p['dimensions_match'] and p['pair_status'] == 'DIMENSION_MISMATCH')} pairs.
- **Requires Visual Review:** {sum(1 for p in all_pair_records if p['requires_visual_review'])} pairs.

---

## 4. Mask Quality & Foreground Reliability

| Mask Type / Characteristic | Count |
| :--- | :--- |
"""
        mask_type_counts = Counter(m["mask_type"] for m in all_mask_records)
        for mtype, cnt in mask_type_counts.most_common():
            md_summary += f"| `{mtype}` | {cnt} |\n"

        fg_status_counts = Counter(m["foreground_estimation_status"] for m in all_mask_records)
        md_summary += f"""
### Foreground Estimation Status Breakdown:
- **Reliable Foreground Estimates:** {fg_status_counts.get('RELIABLE', 0)} (Observed pixel distributions under documented conventions; clinical validity unverified)
- **Provisional Foreground Estimates:** {fg_status_counts.get('PROVISIONAL', 0)} (e.g. JPEG artefacts with threshold >= 128, discrete masks)
- **Unavailable Foreground Estimates:** {fg_status_counts.get('UNAVAILABLE', 0)} (continuous grayscale, color-coded, or multi-color palette masks)

### Extreme Foreground Regions:
- **Completely Empty Masks (0 FG pixels):** {sum(1 for m in all_mask_records if m['is_completely_empty'])}
- **Completely Full-Foreground Masks (100% FG):** {sum(1 for m in all_mask_records if m['is_completely_full'])}
- **Suspiciously Small Foreground (< 0.05% of image area):** {sum(1 for m in all_mask_records if m['is_suspicious_small_foreground'])}

---

## 5. Cross-Dataset & Cross-Cohort Overlap Findings

- **Exact SHA-256 Image Duplicates across Cohorts:** {len(exact_cross_image_duplicates)}
- **Exact SHA-256 Mask Duplicates across Cohorts:** {len(exact_cross_mask_duplicates)}
- **Perceptual Near-Duplicate Candidates (dHash Hamming distance $\\le$ {PERCEPTUAL_HAMMING_THRESHOLD}, excluding exact byte matches):** {len(perceptual_matches)}
  - *Scientific Note:* A dHash match with small Hamming distance represents a non-identical candidate near-duplicate; manual visual verification is required before drawing conclusions regarding dataset independence.

---

## 6. Discrepancy Reconciliation

- **AZH Foot Ulcer Challenge Test Set Entry Count:** Dynamically evaluated at **{azh_challenge_test_image_count} images** in `wound-segmentation-master/data/Foot Ulcer Segmentation Challenge/test/images/` after filtering by image extensions.
- **AZH Nested Archive Verification (`azh_wound_care_center_dataset_patches.zip`):**
  - Outer ZIP Entry: `{azh_nested_zip_metadata['entry_path'] if azh_nested_zip_metadata else 'N/A'}`
  - Uncompressed Size: `{azh_nested_zip_metadata['uncompressed_size_bytes'] if azh_nested_zip_metadata else 'N/A'} bytes` ({round(azh_nested_zip_metadata['uncompressed_size_bytes']/(1024*1024), 2) if azh_nested_zip_metadata else 0} MB)
  - Compressed Size: `{azh_nested_zip_metadata['compressed_size_bytes'] if azh_nested_zip_metadata else 'N/A'} bytes` ({round(azh_nested_zip_metadata['compressed_size_bytes']/(1024*1024), 2) if azh_nested_zip_metadata else 0} MB)
  - CRC32: `{azh_nested_zip_metadata['crc32'] if azh_nested_zip_metadata else 'N/A'}`
  - SHA-256: `{azh_nested_zip_metadata['sha256'] if azh_nested_zip_metadata else 'N/A'}`
  - Total Internal Entries: `{azh_nested_zip_metadata['total_entries'] if azh_nested_zip_metadata else 'N/A'}`
  - Verified Pairs & Stems:
    - Train Split: `{azh_nested_zip_metadata['splits']['train']['supported_image_files_count'] if azh_nested_zip_metadata else 0} supported images, {azh_nested_zip_metadata['splits']['train']['supported_mask_files_count'] if azh_nested_zip_metadata else 0} supported masks -> {azh_nested_zip_metadata['splits']['train']['total_unique_stem_matched_pairs'] if azh_nested_zip_metadata else 0} unique stem-matched pairs ({azh_nested_zip_metadata['splits']['train']['duplicate_image_stems_count'] if azh_nested_zip_metadata else 0} duplicate image stems, {azh_nested_zip_metadata['splits']['train']['unique_image_stems_without_matching_mask_count'] if azh_nested_zip_metadata else 0} unmatched image stems)`.
    - Test Split: `{azh_nested_zip_metadata['splits']['test']['supported_image_files_count'] if azh_nested_zip_metadata else 0} supported images, {azh_nested_zip_metadata['splits']['test']['supported_mask_files_count'] if azh_nested_zip_metadata else 0} supported masks -> {azh_nested_zip_metadata['splits']['test']['total_unique_stem_matched_pairs'] if azh_nested_zip_metadata else 0} unique stem-matched pairs ({azh_nested_zip_metadata['splits']['test']['duplicate_image_stems_count'] if azh_nested_zip_metadata else 0} duplicate image stems, {azh_nested_zip_metadata['splits']['test']['unique_image_stems_without_matching_mask_count'] if azh_nested_zip_metadata else 0} unmatched image stems)`.
    - Total Unique Stem-Matched Pairs: `{azh_nested_zip_metadata['total_unique_stem_matched_pairs'] if azh_nested_zip_metadata else 0}` across both nested cohorts (`AZH::WoundCareCenter_Patches_Train` and `AZH::WoundCareCenter_Patches_Test`).
    - *Scientific Note:* `total_unique_stem_matched_pairs` establishes unique 1:1 filename stem alignment only; image decoding validity, spatial dimension agreement, and foreground mask quality are evaluated separately during the pair audit loop.
  - *Reconciliation Note:* In Stage 1B, the outer ZIP entry for `azh_wound_care_center_dataset_patches.zip` recorded `compressed_size_bytes: 17,708,456` (stored compressed in the master archive) and `uncompressed_size_bytes: 17,970,628` (the byte length of the nested zip file itself). The internal contents expand to 2,218 supported PNG images and masks (224x224). All byte sizes, CRC32, hashes, and pair verification metrics are dynamically computed at runtime directly from `zf.getinfo()`, `io.BytesIO`, and `zipfile.ZipFile(nzf)`.

---

## 7. Generated Manifests & Artifacts

- `data/manifests/image_integrity_audit.json`
- `data/manifests/image_mask_pair_audit.csv`
- `data/manifests/mask_quality_summary.json`
- `data/manifests/cross_dataset_duplicate_audit.json`
- `results/temporary/stage_1c/stage_1c_summary.md`
- `results/temporary/stage_1c/visual_samples/` (Deterministic preview images)
- `results/temporary/stage_1c/stage_1c_contact_sheet.jpg` (Full visual review contact sheet)
"""
        with open(temp_md, "w", encoding="utf-8") as f:
            f.write(md_summary)

        # ---------------------------------------------------------------------
        # Finalization Phase with Phased Rollback Protection (Requirement 1)
        # Note: Rollback-protected finalization (not guaranteed transaction atomicity)
        # ---------------------------------------------------------------------
        print("\nFinalizing outputs from staging to destination locations with phased rollback protection...")
        MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
        STAGE_1C_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        backup_staging_dir = Path(tempfile.mkdtemp(prefix="stage_1c_backup_", dir=PROJECT_ROOT / "results" / "temporary"))
        backed_up_targets: Dict[Path, Path] = {}
        commit_attempted_targets: List[Path] = []
        commit_started = False
        preserve_backup_directory = False

        move_plan: List[Tuple[Path, Path, bool]] = [
            (temp_img_json, OUTPUT_IMAGE_INTEGRITY_JSON, False),
            (temp_csv, OUTPUT_PAIR_AUDIT_CSV, False),
            (temp_mask_json, OUTPUT_MASK_QUALITY_JSON, False),
            (temp_dup_json, OUTPUT_DUPLICATES_JSON, False),
            (temp_md, OUTPUT_SUMMARY_MD, False),
            (temp_contact_sheet, CONTACT_SHEET_PATH, False),
            (temp_visual_samples_dir, VISUAL_SAMPLES_DIR, True),
        ]

        try:
            # -----------------------------------------------------------------
            # Phase A: Back up all existing destination paths
            # -----------------------------------------------------------------
            for src, dst, is_dir in move_plan:
                if dst.exists():
                    backup_dst = backup_staging_dir / dst.name
                    if is_dir:
                        shutil.copytree(dst, backup_dst)
                    else:
                        shutil.copy2(dst, backup_dst)
                    backed_up_targets[dst] = backup_dst

            # -----------------------------------------------------------------
            # Phase B: Rigorous Content Verification of Every Backup
            # -----------------------------------------------------------------
            for dst, bkp_dst in backed_up_targets.items():
                if not bkp_dst.exists():
                    raise RuntimeError(f"Backup verification failed: {bkp_dst} does not exist.")
                if dst.is_dir():
                    if not verify_directory_backup(dst, bkp_dst):
                        raise RuntimeError(f"Backup content verification failed for directory: {dst}")
                else:
                    if dst.stat().st_size != bkp_dst.stat().st_size:
                        raise RuntimeError(f"Backup size mismatch for file: {dst}")
                    if compute_file_sha256(dst) != compute_file_sha256(bkp_dst):
                        raise RuntimeError(f"Backup SHA-256 hash mismatch for file: {dst}")

            # -----------------------------------------------------------------
            # Phase C: Replace destinations (commit phase)
            # -----------------------------------------------------------------
            commit_started = True
            for src, dst, is_dir in move_plan:
                if not src.exists():
                    continue

                # Track destination immediately before attempting modification
                commit_attempted_targets.append(dst)

                if dst.exists():
                    if dst.is_dir():
                        shutil.rmtree(dst)
                    else:
                        dst.unlink()

                shutil.move(str(src), str(dst))

            print("All Stage 1C manifests finalized successfully.")

            # On full success, discard temporary backup
            shutil.rmtree(backup_staging_dir, ignore_errors=True)

        except Exception as finalization_err:
            print(f"\n[CRITICAL FAILURE] Output finalization failed: {finalization_err}", file=sys.stderr)
            print("Initiating rollback recovery to preserve workspace state...", file=sys.stderr)

            restored_destinations = []
            failed_restorations = []
            cleaned_destinations = []
            failed_cleanups = []

            if not commit_started:
                # Failure occurred during Phase A or B: original destinations were never modified
                print("  [Rollback] Failure occurred before commit started; original destinations were untouched.", file=sys.stderr)
                if backed_up_targets:
                    preserve_backup_directory = True
            else:
                # Failure occurred during Phase C: clean ONLY destinations that commit attempted to modify
                for dst in commit_attempted_targets:
                    try:
                        if dst.exists():
                            if dst.is_dir():
                                shutil.rmtree(dst)
                            else:
                                dst.unlink()
                            cleaned_destinations.append(dst.name)
                            print(f"  [Rollback Cleanup] Removed modified destination: {dst.name}", file=sys.stderr)
                    except Exception as del_err:
                        # Critical safeguard: If cleanup of a commit-attempted destination fails,
                        # preserve backup directory immediately, record the destination, and flag for manual recovery
                        preserve_backup_directory = True
                        failed_cleanups.append(f"{dst.name} ({del_err})")
                        print(f"  [Rollback Warning] Could not remove {dst.name}: {del_err}", file=sys.stderr)
                        print(f"  [Rollback Alert] Destination {dst.name} could not be cleaned; manual recovery may be necessary.", file=sys.stderr)

                # Restore backed-up originals for destinations that were modified
                for original_dst, bkp_src in backed_up_targets.items():
                    if original_dst in commit_attempted_targets:
                        try:
                            if bkp_src.exists():
                                # Ensure destination is completely clear before restoration
                                if original_dst.exists():
                                    if original_dst.is_dir():
                                        shutil.rmtree(original_dst)
                                    else:
                                        original_dst.unlink()

                                if bkp_src.is_dir():
                                    shutil.copytree(bkp_src, original_dst)
                                else:
                                    shutil.copy2(bkp_src, original_dst)

                                restored_destinations.append(original_dst.name)
                                print(f"  [Rollback Restored] Successfully restored: {original_dst.name}", file=sys.stderr)
                            else:
                                failed_restorations.append(original_dst.name)
                                preserve_backup_directory = True
                        except Exception as restore_err:
                            failed_restorations.append(f"{original_dst.name} ({restore_err})")
                            preserve_backup_directory = True
                            print(f"  [Rollback Error] Failed to restore {original_dst.name}: {restore_err}", file=sys.stderr)

            # Reporting rollback summary
            print(f"  [Rollback Report] Cleaned modified targets : {cleaned_destinations or 'None'}", file=sys.stderr)
            if failed_cleanups:
                print(f"  [Rollback Report] FAILED cleanups          : {failed_cleanups}", file=sys.stderr)
                print("  [Rollback Report] Manual recovery may be necessary for uncleaned destinations.", file=sys.stderr)
            print(f"  [Rollback Report] Restored targets         : {restored_destinations or 'None'}", file=sys.stderr)
            if failed_restorations:
                print(f"  [Rollback Report] FAILED restorations      : {failed_restorations}", file=sys.stderr)
                preserve_backup_directory = True

            if preserve_backup_directory:
                print(f"  [Rollback Notice] Preserving backup directory for recovery at: {backup_staging_dir}", file=sys.stderr)
            else:
                shutil.rmtree(backup_staging_dir, ignore_errors=True)

            raise RuntimeError(
                f"Output finalization failed and rollback was executed. "
                f"Commit started: {commit_started}; Restored: {restored_destinations}; Failed: {failed_restorations}. "
                f"Original error: {finalization_err}"
            )

    finally:
        # Clean up temporary staging directory if it still exists
        if temp_staging_dir.exists():
            shutil.rmtree(temp_staging_dir, ignore_errors=True)

    print("\n" + "=" * 80)
    print("Stage 1C Audit Complete. Manifests successfully generated:")
    print(f"  1. {OUTPUT_IMAGE_INTEGRITY_JSON.relative_to(project_root)}")
    print(f"  2. {OUTPUT_PAIR_AUDIT_CSV.relative_to(project_root)}")
    print(f"  3. {OUTPUT_MASK_QUALITY_JSON.relative_to(project_root)}")
    print(f"  4. {OUTPUT_DUPLICATES_JSON.relative_to(project_root)}")
    print(f"  5. {OUTPUT_SUMMARY_MD.relative_to(project_root)}")
    print(f"  6. {CONTACT_SHEET_PATH.relative_to(project_root)}")
    print(f"  7. {VISUAL_SAMPLES_DIR.relative_to(project_root)}/")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
