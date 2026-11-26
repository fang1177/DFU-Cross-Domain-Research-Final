"""
Verify that every source file for all 6,598 records actually exists on disk.
"""
import csv
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path("E:/DFU_Cross_Domain_Research")
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.run_full_dataset_preprocessing import (
    PAIR_AUDIT_CSV, MENDELEY_MANIFEST_CSV, SRC_DUP_MAP_CSV, VAL_RECON_CSV,
    build_source_training_records, build_internal_validation_records,
    build_native_target_records, build_mismatched_target_records,
    build_healthy_control_records, PROJECT_ROOT
)

all_pairs = list(csv.DictReader(open(PAIR_AUDIT_CSV, encoding="utf-8")))
mendeley_manifest = list(csv.DictReader(open(MENDELEY_MANIFEST_CSV, encoding="utf-8")))
src_dups = list(csv.DictReader(open(SRC_DUP_MAP_CSV, encoding="utf-8")))
val_recon = list(csv.DictReader(open(VAL_RECON_CSV, encoding="utf-8")))

recs = (
    build_source_training_records(all_pairs, src_dups)
    + build_internal_validation_records(all_pairs, val_recon)
    + build_native_target_records(all_pairs)
    + build_mismatched_target_records(all_pairs)
    + build_healthy_control_records(mendeley_manifest)
)

print(f"Total built: {len(recs)}")

missing_images = []
missing_masks = []
for r in recs:
    img = PROJECT_ROOT / r["image_disk_path"]
    if not img.exists():
        missing_images.append((r["cohort"], r["pair_id"], r["image_disk_path"]))
    if r["mask_disk_path"]:
        mask = PROJECT_ROOT / r["mask_disk_path"]
        if not mask.exists():
            missing_masks.append((r["cohort"], r["pair_id"], r["mask_disk_path"]))

print(f"Missing images: {len(missing_images)}")
if missing_images:
    print("  First 3:", missing_images[:3])
print(f"Missing masks:  {len(missing_masks)}")
if missing_masks:
    print("  First 3:", missing_masks[:3])
print(f"ALL DISK PATHS VALID: {len(missing_images) == 0 and len(missing_masks) == 0}")

