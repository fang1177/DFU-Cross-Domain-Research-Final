import csv, json
from pathlib import Path
from PIL import Image
import numpy as np

with open("data/processed/stage_3_2A_pilot/manifests/pilot_manifest.csv") as f:
    rows = list(csv.DictReader(f))

mismatched = [r for r in rows if r["pilot_cohort"] == "mismatched_target"]
print(f"Mismatched target rows: {len(mismatched)}\n")

for r in mismatched:
    pair_id = r["pair_id"]
    meta_p = Path(f"data/processed/stage_3_2A_pilot/metadata/mismatched_target_{pair_id}_metadata.json")
    with open(meta_p) as f:
        meta = json.load(f)
    print(f"Pair {pair_id}:")
    print(f"  Img orig dims: {meta['image']['original_dimensions']} -> 512x512")
    print(f"  Mask orig dims: {meta['mask']['original_dimensions']} -> 512x512")
    print(f"  Geometry scale: {r['scale']}, Pad: L={r['pad_left']}, T={r['pad_top']}")
    print(f"  Is diagnostic: {r['is_diagnostic']}, Status: {r['mask_status']}")
    print(f"  Orig mask values count: {len(meta['mask']['original_unique_values'])}")
    print(f"  Orig FG pixels: {r['orig_fg_count']}, Proc FG: {r['proc_fg_count']}")
    print()

