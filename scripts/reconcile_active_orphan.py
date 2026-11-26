import csv
import hashlib
from pathlib import Path
from collections import defaultdict

project_root = Path("E:/DFU_Cross_Domain_Research")
pilot_dir = project_root / "data/processed/stage_3_2A_pilot"
manifest_path = pilot_dir / "manifests/pilot_manifest.csv"

with open(manifest_path, mode="r", encoding="utf-8") as f:
    active_manifest_rows = list(csv.DictReader(f))

print(f"Active manifest rows: {len(active_manifest_rows)}")

# Build set of all files associated with active pairs
# For each active pair, trace exactly what run_preprocessing_pilot.py generates:
active_pair_keys = set(f"{r['pilot_cohort']}_{r['pair_id']}" for r in active_manifest_rows)

active_expected_files = set()
for r in active_manifest_rows:
    cohort = r["pilot_cohort"]
    pair_id = r["pair_id"]
    
    # 1. Images: 2 per pair
    active_expected_files.add(f"images/{cohort}_{pair_id}_bilinear.png")
    active_expected_files.add(f"images/{cohort}_{pair_id}_bicubic.png")
    
    # 2. Metadata: 1 per pair
    active_expected_files.add(f"metadata/{cohort}_{pair_id}_metadata.json")
    
    # 3. Masks / Diagnostic Masks
    mask_fn = r["proc_mask_filename"]
    if mask_fn:
        if r["is_diagnostic"] == "True":
            active_expected_files.add(f"diagnostic_masks/{mask_fn}")
        else:
            active_expected_files.add(f"masks/{mask_fn}")
            
    # Additional diagnostic sweeps generated in run_preprocessing_pilot.py
    if cohort == "native_target":
        # Saves jpeg_raw_nn
        active_expected_files.add(f"diagnostic_masks/{cohort}_{pair_id}_jpeg_raw_nn.png")
        # Saves sweep 64, 96, 128, 160, 192
        for t in [64, 96, 128, 160, 192]:
            active_expected_files.add(f"diagnostic_masks/{cohort}_{pair_id}_sweep_{t}.png")
    elif cohort == "mismatched_target":
        # Saves thresh 64, 96, 128, 160, 192
        for t in [64, 96, 128, 160, 192]:
            active_expected_files.add(f"diagnostic_masks/{cohort}_{pair_id}_thresh_{t}.png")

# Manifest itself
active_expected_files.add("manifests/pilot_manifest.csv")

print(f"Total active expected files: {len(active_expected_files)}")

# Audit every physical file on disk in pilot_dir
physical_files = {}
for p in sorted(pilot_dir.rglob("*")):
    if p.is_file():
        rel = str(p.relative_to(pilot_dir)).replace("\\", "/")
        physical_files[rel] = p

print(f"Total physical files on disk: {len(physical_files)}")

# Check overlap
active_present = set()
orphan_present = set()

for rel, p in physical_files.items():
    if rel in active_expected_files:
        active_present.add(rel)
    else:
        orphan_present.add(rel)

print(f"Active files present on disk: {len(active_present)}")
print(f"Orphan files present on disk: {len(orphan_present)}")
print(f"Sum (active + orphan): {len(active_present) + len(orphan_present)}")

# Check if any expected active file is missing
missing_active = active_expected_files - set(physical_files.keys())
print(f"Missing active files: {len(missing_active)} {missing_active}")

# Inspect the orphans by directory
orphan_by_dir = defaultdict(list)
for rel in orphan_present:
    parent = rel.split("/")[0]
    orphan_by_dir[parent].append(rel)

for d, files in orphan_by_dir.items():
    print(f"Orphan files in {d}: {len(files)}")

