import csv
from pathlib import Path

manifest_path = Path("data/processed/stage_3_2A_pilot/manifests/pilot_manifest.csv")
output_csv = Path("results/temporary/stage_3_2A_R4/pair_record_to_mask_mapping.csv")

with open(manifest_path, mode="r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Total active pair records: {len(rows)}")

mapping_rows = []
for idx, r in enumerate(rows, 1):
    cohort = r["pilot_cohort"]
    pair_id = r["pair_id"]
    is_diag = r["is_diagnostic"] == "True"
    mask_fn = r["proc_mask_filename"]
    
    # Destination classification
    if not is_diag:
        dest = "data/processed/stage_3_2A_pilot/masks/"
        dest_cat = "Primary Canonical / Reference Mask"
        has_canon = "True"
    else:
        dest = "data/processed/stage_3_2A_pilot/diagnostic_masks/"
        dest_cat = "Diagnostic-Only Mask (Excluded from Primary Benchmarks)"
        has_canon = "False"
        
    # Reason
    if pair_id == "0233":
        reason = "Discrete multiclass labels {0, 128, 255}; preserved without forced binarization in diagnostic_masks"
    elif cohort == "mismatched_target":
        reason = "Continuous grayscale mask with uncalibrated dimension mismatch; quarantined to diagnostic_masks"
    elif cohort == "healthy_controls":
        reason = "Derived operational negative-control mask (all zeros) for standalone specificity benchmark"
    elif r["mask_status"] == "authentic_negative_empty":
        reason = "Authentic empty lesion mask (zero foreground) in training/validation"
    elif cohort == "native_target":
        reason = "Provisional operational reference binary mask (T>=128 JPEG standardized)"
    else:
        reason = "Strict binary ground truth mask ({0, 255})"
        
    mapping_rows.append({
        "item_number": idx,
        "pilot_cohort": cohort,
        "pair_id": pair_id,
        "cluster_id": r["cluster_id"],
        "is_primary_reference": r["is_primary_reference"],
        "has_canonical_mask": has_canon,
        "canonical_mask_filename": mask_fn if not is_diag else "None",
        "diagnostic_mask_filename": mask_fn if is_diag else "None",
        "output_directory": dest,
        "mask_status": r["mask_status"],
        "classification_category": dest_cat,
        "methodological_justification": reason
    })

fieldnames = [
    "item_number", "pilot_cohort", "pair_id", "cluster_id",
    "is_primary_reference", "has_canonical_mask",
    "canonical_mask_filename", "diagnostic_mask_filename",
    "output_directory", "mask_status", "classification_category",
    "methodological_justification"
]

output_csv.parent.mkdir(parents=True, exist_ok=True)
with open(output_csv, mode="w", newline="", encoding="utf-8") as out_f:
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(mapping_rows)

print(f"Wrote mapping of {len(mapping_rows)} pair records to {output_csv}")
n_canon = len([r for r in mapping_rows if r["has_canonical_mask"] == "True"])
n_diag = len([r for r in mapping_rows if r["has_canonical_mask"] == "False"])
print(f"Reconciliation check: {n_canon} pairs with canonical masks + {n_diag} diagnostic-only pairs = {n_canon + n_diag} total active pairs")

