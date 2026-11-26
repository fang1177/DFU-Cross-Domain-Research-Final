import csv
from collections import Counter

with open("data/processed/stage_3_2A_pilot/manifests/pilot_manifest.csv") as f:
    rows = list(csv.DictReader(f))

active_masks = [r for r in rows if r["proc_mask_filename"] and r["is_diagnostic"] != "True"]
print("Active mask rows count:", len(active_masks))

cohort_counter = Counter(r["pilot_cohort"] for r in active_masks)
status_counter = Counter(r["mask_status"] for r in active_masks)

print("\nCounts by pilot_cohort:")
for k, v in cohort_counter.items():
    print(f"  {k}: {v}")

print("\nCounts by mask_status:")
for k, v in status_counter.items():
    print(f"  {k}: {v}")

print("\nDetailed list of all 40 active mask records:")
for idx, r in enumerate(active_masks, 1):
    print(f"{idx:2d}. Cohort: {r['pilot_cohort']:20} | Pair ID: {r['pair_id']:25} | Status: {r['mask_status']:28} | File: {r['proc_mask_filename']}")

