import csv
from pathlib import Path

manifest_path = Path("data/processed/stage_3_2A_pilot/manifests/pilot_manifest.csv")
output_csv = Path("results/temporary/stage_3_2A_R5/canonical_alternate_diagnostic_role_inventory.csv")

with open(manifest_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

inventory = []
for idx, r in enumerate(rows, 1):
    cohort = r["pilot_cohort"]
    pair_id = r["pair_id"]
    cluster_id = r["cluster_id"]
    is_primary = r["is_primary_reference"] == "True"
    is_diag = r["is_diagnostic"] == "True"
    mask_fn = r["proc_mask_filename"]
    
    # Classify Role:
    # 1. Canonical Reference Mask (Primary Operational Reference)
    # 2. Alternate Annotation (Valid Alternate in Multi-Mask Cluster)
    # 3. Diagnostic-Only Mask (Multiclass or Dimension Mismatched)
    # 4. Derived Negative-Control Label (Healthy Volunteers)
    if cohort == "healthy_controls":
        role = "Derived Negative-Control Label"
        primary_bench = "Excluded (Evaluated in Separate Specificity Benchmark)"
        dir_loc = "data/processed/stage_3_2A_pilot/masks/"
    elif is_diag:
        role = "Diagnostic-Only Mask"
        primary_bench = "Excluded (Quarantined / Diagnostic Inspection Only)"
        dir_loc = "data/processed/stage_3_2A_pilot/diagnostic_masks/"
    elif is_primary:
        role = "Canonical Reference Mask (Primary Operational Reference)"
        if cohort in ["source_training", "internal_validation"]:
            primary_bench = "Eligible Reference (Training / Val Benchmark)"
        elif cohort == "native_target":
            primary_bench = "Primary Target Benchmark (Single Reference Sample)"
        dir_loc = "data/processed/stage_3_2A_pilot/masks/"
    else:
        role = "Alternate Annotation Mask"
        if cohort == "source_training":
            primary_bench = "Training Alternate (Sampled via Cluster-Balanced Sampler)"
        elif cohort == "internal_validation":
            primary_bench = "Validation Alternate (Evaluated in Sensitivity Sweep)"
        elif cohort == "native_target":
            primary_bench = "Excluded from Primary Target Benchmark (Sensitivity Sweep Only)"
        dir_loc = "data/processed/stage_3_2A_pilot/masks/"
        
    inventory.append({
        "item_number": idx,
        "pilot_cohort": cohort,
        "pair_id": pair_id,
        "cluster_id": cluster_id,
        "unique_content_id": r["unique_content_id"],
        "unique_image_sha256": r["orig_img_sha256"],
        "annotation_role": role,
        "is_primary_operational_reference": "True" if is_primary else "False",
        "output_directory": dir_loc,
        "output_mask_filename": mask_fn,
        "target_benchmark_eligibility": primary_bench,
        "risk_of_leakage_mitigation": "Manifest-driven filter: assert is_primary_reference == True"
    })

fieldnames = [
    "item_number", "pilot_cohort", "pair_id", "cluster_id",
    "unique_content_id", "unique_image_sha256", "annotation_role",
    "is_primary_operational_reference", "output_directory",
    "output_mask_filename", "target_benchmark_eligibility",
    "risk_of_leakage_mitigation"
]

output_csv.parent.mkdir(parents=True, exist_ok=True)
with open(output_csv, "w", newline="", encoding="utf-8") as out_f:
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(inventory)

print(f"Wrote {len(inventory)} role records to {output_csv}")
from collections import Counter
print("Role counts:")
for k, v in Counter(x["annotation_role"] for x in inventory).items():
    print(f"  {k}: {v}")

