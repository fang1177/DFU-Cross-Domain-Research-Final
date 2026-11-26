import csv
from pathlib import Path

sample_csv = Path("results/temporary/stage_3_2A/pilot_sample_selection.csv")
rows = list(csv.DictReader(open(sample_csv, encoding="utf-8")))

seen_cohorts = set()
for r in rows:
    cohort = r["pilot_cohort"]
    if cohort not in seen_cohorts:
        seen_cohorts.add(cohort)
        print(f"=== {cohort} ===")
        print(f"  dataset_key: {r['dataset_key']}")
        print(f"  subset: {r['subset']}")
        print(f"  pair_id: {r['pair_id']}")
        print(f"  mask_type: {r['mask_type']}")
        print(f"  image_disk_path: {r['image_disk_path']}")
        print(f"  mask_disk_path: {r['mask_disk_path']}")
        print(f"  is_primary_reference: {r['is_primary_reference']}")
        print(f"  cluster_id: {r['cluster_id']}")
        print(f"  unique_content_id: {r['unique_content_id']}")
        print()

