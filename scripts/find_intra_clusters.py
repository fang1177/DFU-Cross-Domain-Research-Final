import json
from collections import defaultdict

with open("data/manifests/cross_dataset_duplicate_audit.json") as f:
    d = json.load(f)

dups = d["exact_cross_cohort_image_duplicates"]

# Group by cohorts involved
cohort_clusters = defaultdict(list)
for item in dups:
    cohort_key = tuple(sorted(item["cohorts_involved"]))
    cohort_clusters[cohort_key].append(item)

print("Cluster distributions by cohorts involved:")
for k, v in sorted(cohort_clusters.items(), key=lambda x: len(x[1]), reverse=True):
    print(f"  {k}: {len(v)} clusters")

# Let's inspect intra-WSEG multi-mask clusters:
# That is, clusters where multiple wseg samples have the same image hash
wseg_multi = []
for item in dups:
    wseg_occs = [occ for occ in item["occurrences"] if occ["dataset_key"] == "WSEG"]
    if len(wseg_occs) > 1:
        wseg_multi.append((item, wseg_occs))

print(f"\nTotal WSEG clusters with >1 image occurrence: {len(wseg_multi)}")
for item, occs in wseg_multi[:5]:
    print(f"  SHA256: {item['sha256'][:8]}... | Occurrences ({len(occs)}):")
    for o in occs:
        print(f"    {o['path']}")

