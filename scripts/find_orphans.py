import csv
from pathlib import Path

p = Path("data/processed/stage_3_2A_pilot")
images = [x.name for x in (p / "images").iterdir() if x.is_file()]
meta = [x.name for x in (p / "metadata").iterdir() if x.is_file()]

with open(p / "manifests/pilot_manifest.csv") as f:
    rows = list(csv.DictReader(f))

manifest_pairs = set(f"{r['pilot_cohort']}_{r['pair_id']}" for r in rows)
meta_pairs = set(x.replace("_metadata.json", "") for x in meta)
print("Pairs in manifest:", len(manifest_pairs))
print("Pairs in metadata:", len(meta_pairs))
orphan_meta = meta_pairs - manifest_pairs
print("Orphan metadata count:", len(orphan_meta), orphan_meta)

