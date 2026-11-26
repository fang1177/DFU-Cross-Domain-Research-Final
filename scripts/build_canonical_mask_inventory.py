import csv
import hashlib
from pathlib import Path

manifest_path = Path("data/processed/stage_3_2A_pilot/manifests/pilot_manifest.csv")
audit_pairs_path = Path("data/manifests/image_mask_pair_audit.csv")
output_csv = Path("results/temporary/stage_3_2A_R4/canonical_mask_file_inventory.csv")

with open(manifest_path, mode="r", encoding="utf-8") as f:
    manifest_rows = list(csv.DictReader(f))

with open(audit_pairs_path, mode="r", encoding="utf-8") as f:
    audit_rows = list(csv.DictReader(f))

audit_by_img = {r["image_path"]: r for r in audit_rows}

active_masks = [r for r in manifest_rows if r["proc_mask_filename"] and r["is_diagnostic"] != "True"]
print(f"Total active canonical mask records: {len(active_masks)}")

inventory_records = []
for idx, r in enumerate(active_masks, 1):
    cohort = r["pilot_cohort"]
    pair_id = r["pair_id"]
    filename = r["proc_mask_filename"]
    rel_path = f"data/processed/stage_3_2A_pilot/masks/{filename}"
    file_path = Path(rel_path)
    
    # Calculate sha256
    h = hashlib.sha256()
    with open(file_path, "rb") as bf:
        while chunk := bf.read(1024 * 1024):
            h.update(chunk)
    file_sha = h.hexdigest()
    
    # Dataset
    orig_ds = "AZH" if "source" in cohort or "internal" in cohort else ("WSEG" if "native" in cohort else "MENDELEY")
    orig_img_p = r["orig_img_path"]
    orig_mask_p = r["orig_mask_path"]
    img_sha = r["orig_img_sha256"]
    
    # Category definition
    is_derived = "True" if cohort == "healthy_controls" else "False"
    
    # Precise category
    if cohort == "healthy_controls":
        cat = "Derived Operational Negative-Control (All Zeros)"
    elif r["mask_status"] == "authentic_negative_empty":
        cat = "Authentic Negative Empty Mask (Zero Foreground)"
    elif cohort == "native_target":
        cat = "Provisional Operational Reference Mask (T>=128 JPEG)"
    elif "foot-ulcer" in pair_id:
        cat = "Medetec Standardized Strict Binary Mask ({0, 1} -> {0, 255})"
    else:
        cat = "Challenge Strict Binary Mask ({0, 255})"
        
    inventory_records.append({
        "item_number": idx,
        "relative_path": rel_path,
        "filename": filename,
        "sha256": file_sha,
        "original_dataset": orig_ds,
        "original_image_path": orig_img_p,
        "original_mask_path": orig_mask_p if orig_mask_p else "N/A (No Author Mask)",
        "pair_id": pair_id,
        "unique_image_hash": img_sha,
        "annotation_identifier": f"{pair_id}_ref" if r["is_primary_reference"] == "True" else f"{pair_id}_alt",
        "mask_category": cat,
        "is_primary_reference": r["is_primary_reference"],
        "active_manifest_reference": "True",
        "is_derived_negative_control": is_derived
    })

fieldnames = [
    "item_number", "relative_path", "filename", "sha256",
    "original_dataset", "original_image_path", "original_mask_path",
    "pair_id", "unique_image_hash", "annotation_identifier",
    "mask_category", "is_primary_reference", "active_manifest_reference",
    "is_derived_negative_control"
]

output_csv.parent.mkdir(parents=True, exist_ok=True)
with open(output_csv, mode="w", newline="", encoding="utf-8") as out_f:
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(inventory_records)

print(f"Successfully wrote {len(inventory_records)} records to {output_csv}")

