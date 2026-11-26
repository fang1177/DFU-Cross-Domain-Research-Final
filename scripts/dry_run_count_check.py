"""
Dry-run record count verification for Stage 3.2B.
Builds all cohort record sets and reports counts WITHOUT touching output files.
"""
import csv
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path("E:/DFU_Cross_Domain_Research")
sys.path.insert(0, str(PROJECT_ROOT))

PAIR_AUDIT_CSV        = PROJECT_ROOT / "data/manifests/image_mask_pair_audit.csv"
MENDELEY_MANIFEST_CSV = PROJECT_ROOT / "data/interim/stage_3_1/extraction_manifests/mendeley_extraction_manifest.csv"
SRC_DUP_MAP_CSV       = PROJECT_ROOT / "results/temporary/stage_1d_r2/source_training_duplicate_mapping.csv"
VAL_RECON_CSV         = PROJECT_ROOT / "results/temporary/stage_1d_r2/validation_record_reconciliation.csv"

all_pairs = list(csv.DictReader(open(PAIR_AUDIT_CSV, encoding="utf-8")))
mendeley_manifest = list(csv.DictReader(open(MENDELEY_MANIFEST_CSV, encoding="utf-8")))
src_dups = list(csv.DictReader(open(SRC_DUP_MAP_CSV, encoding="utf-8")))
val_recon = list(csv.DictReader(open(VAL_RECON_CSV, encoding="utf-8")))

# -- Source Training --
dup_image_paths = {r["image_path"] for r in src_dups}
clusters = defaultdict(list)
for r in src_dups:
    clusters[r["image_sha256"]].append(r)
dup_count = sum(len(v) for v in clusters.values())
single_source = [
    p for p in all_pairs
    if p["subset"] in ("AZH_Challenge_Train", "AZH_Medetec_Train")
    and p["image_path"] not in dup_image_paths
]
src_total = dup_count + len(single_source)
print(f"Source Training:   {src_total}  (dup cluster records={dup_count}, singles={len(single_source)})")

# -- Internal Validation --
independent_ids = {r["pair_id"] for r in val_recon if r["train_overlap_status"] == "INDEPENDENT"}
val_rows = [
    p for p in all_pairs
    if p["subset"] == "AZH_Challenge_Val" and p["pair_id"] in independent_ids
]
print(f"Internal Validation: {len(val_rows)}  (excluded 7 leaked)")

# -- Native Target --
wseg_clean = [
    p for p in all_pairs
    if p["dataset_key"] == "WSEG"
    and p["dimensions_match"] == "True"
    and p["mask_type"] == "BINARY_LIKE_JPEG_ARTEFACTS"
]
print(f"Native Target (WSEG): {len(wseg_clean)}")

# -- Mismatched Target --
mend_mismatch = [
    p for p in all_pairs
    if p["dataset_key"] == "MENDELEY"
    and p["dimensions_match"] == "False"
]
print(f"Mismatched Target: {len(mend_mismatch)}")

# -- Healthy Controls --
nomal = [
    m for m in mendeley_manifest
    if "Nomal" in m["extracted_relative_path"]
    and m["extension"].lower() in (".jpg", ".jpeg", ".png")
    and m["extraction_status"] == "SUCCESS"
]
print(f"Healthy Controls:  {len(nomal)}")

total = src_total + len(val_rows) + len(wseg_clean) + len(mend_mismatch) + len(nomal)
print(f"\nTOTAL: {total}  (expected 6598)")
print(f"Match expected: {total == 6598}")

