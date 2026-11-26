import csv, json
from pathlib import Path

manifest_path = Path("data/processed/stage_3_2A_pilot/manifests/pilot_manifest.csv")
with open(manifest_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

wseg_cluster_rows = [r for r in rows if r["pair_id"] in ["49422580", "49498192"]]
for r in wseg_cluster_rows:
    print(r["pilot_cohort"], r["pair_id"], r["cluster_id"], r["is_primary_reference"], r["orig_img_sha256"][:12], r["proc_mask_filename"], r["mask_status"])

for pid in ["49422580", "49498192"]:
    meta_p = Path(f"data/processed/stage_3_2A_pilot/metadata/native_target_{pid}_metadata.json")
    if meta_p.exists():
        with open(meta_p) as f:
            d = json.load(f)
            print(f"Metadata {pid}: is_primary_ref={d['is_primary_reference']}, cluster={d['cluster_id']}, unique_content={d['unique_content_id']}")

