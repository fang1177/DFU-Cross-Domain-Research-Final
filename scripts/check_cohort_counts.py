import csv
from pathlib import Path
from collections import Counter

pair_csv = Path("data/manifests/image_mask_pair_audit.csv")
rows = list(csv.DictReader(open(pair_csv, encoding="utf-8")))

# Source training: AZH_Challenge_Train + AZH_Medetec_Train
src_train = [r for r in rows if r["subset"] in ("AZH_Challenge_Train", "AZH_Medetec_Train")]
print(f"Source Training (all): {len(src_train)}")
for sub in ("AZH_Challenge_Train", "AZH_Medetec_Train"):
    sub_rows = [r for r in src_train if r["subset"] == sub]
    statuses = dict(Counter(r["pair_status"] for r in sub_rows))
    print(f"  {sub}: {len(sub_rows)} | statuses: {statuses}")

print()
val = [r for r in rows if r["subset"] == "AZH_Challenge_Val"]
print(f"Internal Validation: {len(val)}")
statuses = dict(Counter(r["pair_status"] for r in val))
print(f"  statuses: {statuses}")

print()
wseg_ok = [r for r in rows if r["dataset_key"] == "WSEG" and r["dimensions_match"] == "True" and r["mask_type"] == "BINARY_LIKE_JPEG_ARTEFACTS"]
print(f"Native Target (WSEG clean): {len(wseg_ok)}")

print()
mend_mismatch = [r for r in rows if r["dataset_key"] == "MENDELEY" and r["dimensions_match"] == "False"]
print(f"Mismatched Target (MENDELEY 331x224): {len(mend_mismatch)}")

print()
mend_ext = Path("data/interim/stage_3_1/extraction_manifests/mendeley_extraction_manifest.csv")
ext_rows = list(csv.DictReader(open(mend_ext, encoding="utf-8")))
nomal = [r for r in ext_rows if "Nomal" in r["extracted_relative_path"] and r["extension"].lower() in (".jpg", ".jpeg", ".png")]
print(f"Healthy Controls (Mendeley Nomal images): {len(nomal)}")

print()
total = len(src_train) + len(val) + len(wseg_ok) + len(mend_mismatch) + len(nomal)
print(f"TOTAL to process: {total}")

