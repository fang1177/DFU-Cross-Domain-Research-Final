import csv
import hashlib
from pathlib import Path

project_root = Path("E:/DFU_Cross_Domain_Research")
pilot_dir = project_root / "data/processed/stage_3_2A_pilot"
manifest_path = pilot_dir / "manifests/pilot_manifest.csv"
output_csv = project_root / "results/temporary/stage_3_2A_R2/orphan_file_inventory.csv"

with open(manifest_path, mode="r", encoding="utf-8") as f:
    active_rows = list(csv.DictReader(f))

active_pairs = set(f"{r['pilot_cohort']}_{r['pair_id']}" for r in active_rows)
print(f"Total active pair IDs in pilot manifest: {len(active_pairs)}")

# Map of referenced files in active manifest
referenced_files = set()
for r in active_rows:
    cohort = r["pilot_cohort"]
    pair_id = r["pair_id"]
    # Image files
    referenced_files.add(f"images/{cohort}_{pair_id}_bilinear.png")
    referenced_files.add(f"images/{cohort}_{pair_id}_bicubic.png")
    # Metadata file
    referenced_files.add(f"metadata/{cohort}_{pair_id}_metadata.json")
    # Mask file
    if r["proc_mask_filename"]:
        if r["is_diagnostic"] == "True":
            referenced_files.add(f"diagnostic_masks/{r['proc_mask_filename']}")
        else:
            referenced_files.add(f"masks/{r['proc_mask_filename']}")
    # Diagnostic sweeps / raw nn
    if cohort == "native_target":
        referenced_files.add(f"diagnostic_masks/{cohort}_{pair_id}_jpeg_raw_nn.png")
        for t in [64, 96, 128, 160, 192]:
            referenced_files.add(f"diagnostic_masks/{cohort}_{pair_id}_sweep_{t}.png")
    elif cohort == "mismatched_target":
        for t in [64, 96, 128, 160, 192]:
            referenced_files.add(f"diagnostic_masks/{cohort}_{pair_id}_thresh_{t}.png")

referenced_files.add("manifests/pilot_manifest.csv")

print(f"Total explicitly referenced active files: {len(referenced_files)}")

# Collect all files in pilot_dir
all_pilot_files = []
for p in sorted(pilot_dir.rglob("*")):
    if p.is_file():
        rel_p = str(p.relative_to(pilot_dir)).replace("\\", "/")
        all_pilot_files.append((rel_p, p))

print(f"Total physical files in pilot directory: {len(all_pilot_files)}")

orphan_records = []
for rel_p, p in all_pilot_files:
    if rel_p not in referenced_files:
        # Compute SHA-256
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        digest = h.hexdigest()
        size_bytes = p.stat().st_size
        
        # Categorize
        parent_dir = p.parent.name
        filename = p.name
        
        # Identify superseded Pair ID
        superseded_pair_id = "UNKNOWN"
        for part in filename.split("_"):
            # check if part is all digits or digit-with-suffix
            clean_part = part.replace(".png", "").replace(".json", "")
            if any(clean_part.startswith(prefix) for prefix in ["5", "6", "7"]) and len(clean_part) >= 8:
                superseded_pair_id = clean_part
                break
            elif "56523561" in filename:
                superseded_pair_id = "56523561"
                break

        orphan_records.append({
            "relative_path": f"data/processed/stage_3_2A_pilot/{rel_p}",
            "filename": filename,
            "file_size_bytes": size_bytes,
            "sha256": digest,
            "file_category": parent_dir,
            "associated_superseded_pair_id": superseded_pair_id,
            "reason_orphaned": "Generated during Stage 3.2A Iteration 1 before WSEG 1:1 dimension mismatch filtering; superseded by active sample selection",
            "referenced_in_active_manifest": "False"
        })

print(f"Total orphan files identified: {len(orphan_records)}")

fieldnames = [
    "relative_path", "filename", "file_size_bytes", "sha256", 
    "file_category", "associated_superseded_pair_id", 
    "reason_orphaned", "referenced_in_active_manifest"
]

with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(orphan_records)

print(f"Wrote orphan file inventory to {output_csv}")

