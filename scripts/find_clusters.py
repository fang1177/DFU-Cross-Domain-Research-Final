import json

with open("data/manifests/cross_dataset_duplicate_audit.json") as f:
    d = json.load(f)

dups = d["exact_cross_cohort_image_duplicates"]
print(f"Total exact image duplicate clusters: {len(dups)}")

targets = ["0087", "0111", "0179", "0303", "0233", "0092", "0128", "0635", "0386", "0889", "0339", "0183"]
for item in dups:
    paths = [occ["path"] for occ in item["occurrences"]]
    matched = [t for t in targets if any(t in p for p in paths)]
    if matched:
        print(f"Cluster matching {matched} (SHA256: {item['sha256'][:10]}..., total: {item['total_occurrences']}):")
        for occ in item["occurrences"]:
            print(f"   {occ['subset']} | {occ['path']}")

