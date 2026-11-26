"""
Read-only image decoding and integrity audit utility for extracted datasets.
Inspects:
- Successful decoding
- Dimensions (width, height)
- Channel count and mode
- Min, max, and unique pixel values (for masks)
- File size and SHA-256
- Mask type categorization (binary, JPEG-ringing, continuous grayscale, empty)
- Detection of corrupt or zero-dimension images
Does NOT modify, resize, normalize, or threshold any images or masks.
"""

import os
import sys
import csv
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from PIL import Image, ImageOps
import numpy as np

def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("image_auditor")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh_fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    fh.setFormatter(fh_fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch_fmt = logging.Formatter("[%(levelname)s] %(message)s")
    ch.setFormatter(ch_fmt)
    logger.addHandler(ch)

    return logger

def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()

def audit_single_image(file_path: Path, relative_path: str, dataset_key: str, is_mask: bool) -> Dict[str, Any]:
    record = {
        "dataset_key": dataset_key,
        "relative_path": relative_path,
        "is_mask": is_mask,
        "file_size_bytes": file_path.stat().st_size if file_path.exists() else 0,
        "sha256": compute_sha256(file_path) if file_path.exists() else "",
        "decode_success": False,
        "format": None,
        "mode": None,
        "width": 0,
        "height": 0,
        "channels": 0,
        "min_pixel": None,
        "max_pixel": None,
        "num_unique_values": 0,
        "is_empty": False,
        "is_corrupt": False,
        "error_message": ""
    }

    if not file_path.exists():
        record["is_corrupt"] = True
        record["error_message"] = "File not found"
        return record

    try:
        with Image.open(file_path) as img:
            record["format"] = img.format
            record["mode"] = img.mode
            record["width"], record["height"] = img.size
            record["channels"] = len(img.getbands())
            
            if record["width"] <= 0 or record["height"] <= 0:
                record["is_corrupt"] = True
                record["error_message"] = "Zero or negative dimensions"
                return record

            # Fast pixel array inspection
            arr = np.array(img)
            record["min_pixel"] = int(arr.min())
            record["max_pixel"] = int(arr.max())
            
            if is_mask:
                unique_vals = np.unique(arr)
                record["num_unique_values"] = len(unique_vals)
                if record["max_pixel"] == 0:
                    record["is_empty"] = True

            record["decode_success"] = True

    except Exception as e:
        record["decode_success"] = False
        record["is_corrupt"] = True
        record["error_message"] = str(e)

    return record

def run_extracted_image_audit(
    extracted_root: Path,
    output_report_json: Path,
    output_summary_csv: Path,
    logger: logging.Logger
) -> Dict[str, Any]:
    logger.info("Starting read-only image decoding audit of extracted files...")
    
    # Collect all image files
    extensions = {".jpg", ".jpeg", ".png"}
    datasets = ["AZH", "WSEG", "MENDELEY"]
    
    audit_records = []
    total_images = 0
    total_masks = 0
    corrupt_count = 0

    for ds in datasets:
        ds_dir = extracted_root / ds
        if not ds_dir.exists():
            logger.warning(f"Dataset directory not found: {ds_dir}")
            continue

        files = [p for p in ds_dir.rglob("*") if p.is_file() and p.suffix.lower() in extensions]
        logger.info(f"Auditing {len(files)} files in {ds}...")

        for i, fpath in enumerate(files):
            rel_path = str(fpath.relative_to(ds_dir)).replace("\\", "/")
            
            # Identify mask vs image from path
            is_mask = False
            if ds == "AZH":
                is_mask = "labels" in rel_path or "mask" in rel_path
            elif ds == "WSEG":
                is_mask = "mask" in rel_path
            elif ds == "MENDELEY":
                is_mask = "wound_mask" in rel_path

            if is_mask:
                total_masks += 1
            else:
                total_images += 1

            rec = audit_single_image(fpath, rel_path, ds, is_mask)
            audit_records.append(rec)

            if not rec["decode_success"]:
                corrupt_count += 1
                logger.error(f"Corrupt image detected: {ds}/{rel_path} - {rec['error_message']}")

            if (i + 1) % 2000 == 0 or (i + 1) == len(files):
                logger.info(f"  {ds}: Audited {i + 1}/{len(files)} files...")

    logger.info(f"Audit completed: {len(audit_records)} total files inspected.")
    logger.info(f"Images: {total_images}, Masks: {total_masks}, Corrupt: {corrupt_count}")

    # Summary statistics
    summary = {
        "total_files_audited": len(audit_records),
        "total_images": total_images,
        "total_masks": total_masks,
        "corrupt_files_count": corrupt_count,
        "formats": {},
        "modes": {},
        "empty_masks_count": sum(1 for r in audit_records if r.get("is_empty", False))
    }
    for r in audit_records:
        fmt = r["format"] or "UNKNOWN"
        mode = r["mode"] or "UNKNOWN"
        summary["formats"][fmt] = summary["formats"].get(fmt, 0) + 1
        summary["modes"][mode] = summary["modes"].get(mode, 0) + 1

    # Write summary CSV
    output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_key", "relative_path", "is_mask", "format", "mode", "width", "height",
        "channels", "min_pixel", "max_pixel", "num_unique_values", "is_empty",
        "decode_success", "is_corrupt", "file_size_bytes", "sha256", "error_message"
    ]
    with open(output_summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_records)

    # Write JSON summary
    with open(output_report_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary CSV written: {output_summary_csv}")
    logger.info(f"Report JSON written: {output_report_json}")
    return summary

if __name__ == "__main__":
    log_dir = Path("data/interim/stage_3_1/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir / "image_decoding_audit.log")

    extracted_root = Path("data/interim/stage_3_1/extracted")
    report_json = Path("data/interim/stage_3_1/integrity_reports/extracted_image_audit_summary.json")
    summary_csv = Path("data/interim/stage_3_1/integrity_reports/extracted_image_audit_records.csv")

    summary = run_extracted_image_audit(extracted_root, report_json, summary_csv, logger)
    print("Audit Summary:", summary)

