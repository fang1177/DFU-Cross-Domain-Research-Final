"""
Stage 3.2A: Controlled Preprocessing Pilot Execution Script.
Applies deterministic letterboxing, interpolation separation, mask binarization,
diagnostic mask preservation, metadata generation, and provenance tracking.
"""

import os
import sys
import csv
import json
import hashlib
import numpy as np
from pathlib import Path
from PIL import Image

# Ensure project root is in path
project_root = Path("E:/DFU_Cross_Domain_Research")
sys.path.insert(0, str(project_root))

from src.data.transforms import (
    calculate_letterbox_geometry,
    apply_letterbox_image,
    apply_letterbox_mask,
    compute_tolerance_boundary_fields
)

def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def compute_array_sha256(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(arr.tobytes())
    return h.hexdigest()

def get_bounding_box(mask_arr: np.ndarray):
    """Returns (ymin, xmin, ymax, xmax) or None if empty."""
    pos = np.argwhere(mask_arr > 0)
    if len(pos) == 0:
        return None
    ymin, xmin = pos.min(axis=0)
    ymax, xmax = pos.max(axis=0)
    return int(ymin), int(xmin), int(ymax), int(xmax)

def run_pilot():
    print("[INFO] Starting Stage 3.2A Controlled Preprocessing Pilot...")
    
    sample_csv = project_root / "results/temporary/stage_3_2A/pilot_sample_selection.csv"
    with open(sample_csv, mode="r", encoding="utf-8") as f:
        samples = list(csv.DictReader(f))
        
    print(f"[INFO] Loaded {len(samples)} pilot sample records.")
    
    out_dir = project_root / "data/processed/stage_3_2A_pilot"
    img_out_dir = out_dir / "images"
    mask_out_dir = out_dir / "masks"
    diag_mask_out_dir = out_dir / "diagnostic_masks"
    manifest_out_dir = out_dir / "manifests"
    metadata_out_dir = out_dir / "metadata"
    
    for d in [img_out_dir, mask_out_dir, diag_mask_out_dir, manifest_out_dir, metadata_out_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    manifest_records = []
    
    for idx, row in enumerate(samples):
        cohort = row["pilot_cohort"]
        pair_id = row["pair_id"]
        unique_content_id = row["unique_content_id"]
        img_disk_path = project_root / row["image_disk_path"]
        mask_disk_path = project_root / row["mask_disk_path"] if row["mask_disk_path"] else None
        
        orig_img_sha = compute_sha256(img_disk_path)
        
        # Load Image
        with Image.open(img_disk_path) as raw_img:
            img_rgb = raw_img.convert("RGB")
            orig_img_w, orig_img_h = img_rgb.size
            
            # Geometry
            geom = calculate_letterbox_geometry(orig_img_w, orig_img_h, 512, 512)
            
            # Letterbox Image: Bilinear (Canonical)
            img_bilinear = apply_letterbox_image(img_rgb, (512, 512), interpolation_mode=Image.BILINEAR)
            # Letterbox Image: Bicubic (Comparison)
            img_bicubic = apply_letterbox_image(img_rgb, (512, 512), interpolation_mode=Image.BICUBIC)
            
        img_filename = f"{cohort}_{pair_id}_bilinear.png"
        img_out_path = img_out_dir / img_filename
        img_bilinear.save(img_out_path, format="PNG")
        proc_img_sha = compute_sha256(img_out_path)
        
        img_bicubic_filename = f"{cohort}_{pair_id}_bicubic.png"
        img_bicubic_path = img_out_dir / img_bicubic_filename
        img_bicubic.save(img_bicubic_path, format="PNG")
        proc_img_bicubic_sha = compute_sha256(img_bicubic_path)
        
        # Compute image statistics (bilinear)
        arr_img = np.array(img_bilinear, dtype=np.float32) / 255.0
        active_h_slice = slice(geom["pad_top"], geom["pad_top"] + geom["scaled_height"])
        active_w_slice = slice(geom["pad_left"], geom["pad_left"] + geom["scaled_width"])
        active_pixels = arr_img[active_h_slice, active_w_slice, :]
        img_mean = active_pixels.mean(axis=(0, 1)).tolist()
        img_std = active_pixels.std(axis=(0, 1)).tolist()
        
        # Now process Mask
        mask_status = "none"
        orig_mask_sha = ""
        orig_mask_w, orig_mask_h = 0, 0
        proc_mask_filename = ""
        proc_mask_sha = ""
        orig_fg_count = 0
        proc_fg_count = 0
        area_ratio = 1.0
        bbox_orig = None
        bbox_proc = None
        unique_mask_values = []
        is_diagnostic = False
        
        if mask_disk_path and mask_disk_path.exists():
            orig_mask_sha = compute_sha256(mask_disk_path)
            with Image.open(mask_disk_path) as raw_mask:
                mask_l = raw_mask.convert("L")
                orig_mask_w, orig_mask_h = mask_l.size
                arr_orig_mask = np.array(mask_l)
                
                # Check Medetec [0, 1] normalization: standardize to [0, 255]
                orig_unique = np.unique(arr_orig_mask).tolist()
                unique_mask_values = orig_unique
                if orig_unique == [0, 1]:
                    # Standardize Medetec {0, 1} to canonical {0, 255}
                    mask_l = Image.fromarray((arr_orig_mask * 255).astype(np.uint8))
                    arr_orig_mask = np.array(mask_l)
                    
                bbox_orig = get_bounding_box(arr_orig_mask)
                
                # Apply Letterbox to mask: STRICTLY NEAREST-NEIGHBOR
                mask_nn = apply_letterbox_mask(mask_l, (512, 512), fill_value=0)
                arr_mask_nn = np.array(mask_nn)
                
                # Evaluate Mask Type and Destination
                # 1. Multiclass check (e.g. 0233)
                if pair_id == "0233" or (len(orig_unique) > 2 and cohort not in ["native_target", "mismatched_target"]):
                    is_diagnostic = True
                    mask_status = "diagnostic_multiclass"
                    proc_mask_filename = f"{cohort}_{pair_id}_multiclass_raw.png"
                    proc_mask_path = diag_mask_out_dir / proc_mask_filename
                    mask_nn.save(proc_mask_path, format="PNG")
                    proc_mask_sha = compute_sha256(proc_mask_path)
                    
                # 2. Mismatched target (Mendeley continuous grayscale, 331x331 -> 224x224 mask)
                elif cohort == "mismatched_target":
                    is_diagnostic = True
                    mask_status = "diagnostic_continuous_grayscale"
                    proc_mask_filename = f"{cohort}_{pair_id}_continuous_raw.png"
                    proc_mask_path = diag_mask_out_dir / proc_mask_filename
                    mask_nn.save(proc_mask_path, format="PNG")
                    proc_mask_sha = compute_sha256(proc_mask_path)
                    
                    # Generate provisional thresholded copies in diagnostic_masks
                    for t in [64, 96, 128, 160, 192]:
                        t_mask = (arr_mask_nn >= t).astype(np.uint8) * 255
                        t_path = diag_mask_out_dir / f"{cohort}_{pair_id}_thresh_{t}.png"
                        Image.fromarray(t_mask).save(t_path, format="PNG")
                        
                # 3. Categorical mask (AZH clean binary or WSEG JPEG-artefact)
                else:
                    if "JPEG" in row["mask_type"]:
                        # WSEG JPEG artefact: binarize at T=128 for canonical, save raw NN to diagnostic
                        mask_status = "canonical_binarized_jpeg"
                        diag_raw_path = diag_mask_out_dir / f"{cohort}_{pair_id}_jpeg_raw_nn.png"
                        mask_nn.save(diag_raw_path, format="PNG")
                        
                        # Canonical thresholded mask at T=128
                        canonical_bin = (arr_mask_nn >= 128).astype(np.uint8) * 255
                        proc_mask_filename = f"{cohort}_{pair_id}_canonical_mask.png"
                        proc_mask_path = mask_out_dir / proc_mask_filename
                        Image.fromarray(canonical_bin).save(proc_mask_path, format="PNG")
                        proc_mask_sha = compute_sha256(proc_mask_path)
                        
                        # Generate sensitivity sweep copies in diagnostic_masks
                        for t in [64, 96, 128, 160, 192]:
                            t_mask = (arr_mask_nn >= t).astype(np.uint8) * 255
                            t_path = diag_mask_out_dir / f"{cohort}_{pair_id}_sweep_{t}.png"
                            Image.fromarray(t_mask).save(t_path, format="PNG")
                            
                        orig_fg_count = int((arr_orig_mask >= 128).sum())
                        proc_fg_count = int((canonical_bin == 255).sum())
                        bbox_proc = get_bounding_box(canonical_bin)
                        
                    elif float(row["mask_foreground_pct"]) == 0 or len(orig_unique) == 1:
                        # Empty mask (e.g. 0092, 0128)
                        mask_status = "authentic_negative_empty"
                        proc_mask_filename = f"{cohort}_{pair_id}_empty_mask.png"
                        proc_mask_path = mask_out_dir / proc_mask_filename
                        mask_nn.save(proc_mask_path, format="PNG")
                        proc_mask_sha = compute_sha256(proc_mask_path)
                        orig_fg_count = 0
                        proc_fg_count = int((arr_mask_nn > 0).sum())
                        bbox_proc = None
                        
                    else:
                        # Clean strict binary (AZH or standardized Medetec)
                        mask_status = "strict_binary"
                        proc_mask_filename = f"{cohort}_{pair_id}_canonical_mask.png"
                        proc_mask_path = mask_out_dir / proc_mask_filename
                        mask_nn.save(proc_mask_path, format="PNG")
                        proc_mask_sha = compute_sha256(proc_mask_path)
                        orig_fg_count = int((arr_orig_mask == 255).sum())
                        proc_fg_count = int((arr_mask_nn == 255).sum())
                        bbox_proc = get_bounding_box(arr_mask_nn)
                        
                    expected_scaled_fg = orig_fg_count * (geom["scale"] ** 2)
                    area_ratio = proc_fg_count / expected_scaled_fg if expected_scaled_fg > 0 else 1.0
                    
        elif cohort == "healthy_controls":
            mask_status = "authentic_negative_healthy"
            zero_mask = Image.new("L", (512, 512), 0)
            proc_mask_filename = f"{cohort}_{pair_id}_zero_mask.png"
            proc_mask_path = mask_out_dir / proc_mask_filename
            zero_mask.save(proc_mask_path, format="PNG")
            proc_mask_sha = compute_sha256(proc_mask_path)
            orig_fg_count = 0
            proc_fg_count = 0
            area_ratio = 1.0
            bbox_orig = None
            bbox_proc = None
            unique_mask_values = [0]
            
        # Distance fields for boundary loss
        phi_fp_sha = ""
        phi_fn_sha = ""
        if mask_status in ["strict_binary", "canonical_binarized_jpeg", "authentic_negative_empty", "authentic_negative_healthy"]:
            target_mask_arr = np.array(Image.open(proc_mask_path))
            bin_bool = (target_mask_arr == 255)
            phi_fp, phi_fn = compute_tolerance_boundary_fields(bin_bool, tau=2.0, max_cap=20.0)
            phi_fp_sha = compute_array_sha256(phi_fp)
            phi_fn_sha = compute_array_sha256(phi_fn)
            
        # Metadata JSON per sample
        metadata_entry = {
            "pair_id": pair_id,
            "pilot_cohort": cohort,
            "cluster_id": row["cluster_id"],
            "unique_content_id": unique_content_id,
            "is_primary_reference": row["is_primary_reference"] == "True",
            "image": {
                "original_path": row["image_path"],
                "original_disk_path": row["image_disk_path"],
                "original_dimensions": [orig_img_w, orig_img_h],
                "original_sha256": orig_img_sha,
                "processed_bilinear_path": str(img_out_path.relative_to(project_root)).replace("\\", "/"),
                "processed_bilinear_sha256": proc_img_sha,
                "processed_bicubic_path": str(img_bicubic_path.relative_to(project_root)).replace("\\", "/"),
                "processed_bicubic_sha256": proc_img_bicubic_sha,
                "active_roi_mean_rgb": img_mean,
                "active_roi_std_rgb": img_std
            },
            "geometry": geom,
            "mask": {
                "original_path": row["mask_path"],
                "original_disk_path": row["mask_disk_path"],
                "original_dimensions": [orig_mask_w, orig_mask_h] if mask_disk_path else None,
                "original_sha256": orig_mask_sha,
                "processed_path": str(proc_mask_path.relative_to(project_root)).replace("\\", "/") if proc_mask_filename else "",
                "processed_sha256": proc_mask_sha,
                "mask_status": mask_status,
                "is_diagnostic": is_diagnostic,
                "original_unique_values": unique_mask_values,
                "original_foreground_pixels": orig_fg_count,
                "processed_foreground_pixels": proc_fg_count,
                "area_preservation_ratio": float(area_ratio),
                "original_bbox_ymin_xmin_ymax_xmax": bbox_orig,
                "processed_bbox_ymin_xmin_ymax_xmax": bbox_proc,
                "phi_fp_sha256": phi_fp_sha,
                "phi_fn_sha256": phi_fn_sha
            }
        }
        
        meta_file = metadata_out_dir / f"{cohort}_{pair_id}_metadata.json"
        with open(meta_file, mode="w", encoding="utf-8") as f:
            json.dump(metadata_entry, f, indent=2)
            
        manifest_records.append({
            "pilot_cohort": cohort,
            "pair_id": pair_id,
            "cluster_id": row["cluster_id"],
            "unique_content_id": unique_content_id,
            "is_primary_reference": row["is_primary_reference"],
            "orig_img_path": row["image_path"],
            "orig_img_w": orig_img_w,
            "orig_img_h": orig_img_h,
            "orig_img_sha256": orig_img_sha,
            "proc_img_bilinear_sha256": proc_img_sha,
            "proc_img_bicubic_sha256": proc_img_bicubic_sha,
            "scale": geom["scale"],
            "pad_left": geom["pad_left"],
            "pad_top": geom["pad_top"],
            "pad_right": geom["pad_right"],
            "pad_bottom": geom["pad_bottom"],
            "orig_mask_path": row["mask_path"],
            "orig_mask_w": orig_mask_w,
            "orig_mask_h": orig_mask_h,
            "orig_mask_sha256": orig_mask_sha,
            "proc_mask_filename": proc_mask_filename,
            "proc_mask_sha256": proc_mask_sha,
            "mask_status": mask_status,
            "is_diagnostic": is_diagnostic,
            "orig_fg_count": orig_fg_count,
            "proc_fg_count": proc_fg_count,
            "area_ratio": area_ratio,
            "phi_fp_sha256": phi_fp_sha,
            "phi_fn_sha256": phi_fn_sha
        })
        
    # Write manifest CSV
    manifest_csv = manifest_out_dir / "pilot_manifest.csv"
    with open(manifest_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_records[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_records)
        
    print(f"[INFO] Stage 3.2A Pilot complete. Generated {len(manifest_records)} records in {manifest_csv}")

if __name__ == "__main__":
    run_pilot()
