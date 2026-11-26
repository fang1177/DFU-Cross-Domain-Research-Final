import csv
from pathlib import Path

pilot_dir = Path("data/processed/stage_3_2A_pilot")
images = set(x.name for x in (pilot_dir / "images").iterdir() if x.is_file())
metadata = set(x.name for x in (pilot_dir / "metadata").iterdir() if x.is_file())
masks = set(x.name for x in (pilot_dir / "masks").iterdir() if x.is_file())
diag = set(x.name for x in (pilot_dir / "diagnostic_masks").iterdir() if x.is_file())

with open(pilot_dir / "manifests/pilot_manifest.csv") as f:
    rows = list(csv.DictReader(f))

active_pair_keys = set(f"{r['pilot_cohort']}_{r['pair_id']}" for r in rows)

meta_keys = set(x.replace("_metadata.json", "") for x in metadata)
diff_keys = meta_keys - active_pair_keys
print("Diff keys (11 pairs):", sorted(list(diff_keys)))

orphan_images = [x for x in images if any(k in x for k in diff_keys)]
orphan_meta = [x for x in metadata if any(k in x for k in diff_keys)]
orphan_masks = [x for x in masks if any(k in x for k in diff_keys)]
orphan_diag = [x for x in diag if any(k in x for k in diff_keys)]

print("Orphan images:", len(orphan_images))
print("Orphan metadata:", len(orphan_meta))
print("Orphan masks:", len(orphan_masks))
print("Orphan diag:", len(orphan_diag))
print("Total files from 11 superseded pairs:", len(orphan_images) + len(orphan_meta) + len(orphan_masks) + len(orphan_diag))

# Are there any other files?
total_physical = len(images) + len(metadata) + len(masks) + len(diag) + 1 # +1 manifest
print("Total physical files on disk:", total_physical)

