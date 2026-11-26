import csv
from pathlib import Path

# Load orphan_file_inventory.csv generated in Stage 3.2A-R2
inventory_csv = Path("results/temporary/stage_3_2A_R2/orphan_file_inventory.csv")
with open(inventory_csv, mode="r", encoding="utf-8") as f:
    r2_orphans = list(csv.DictReader(f))

print(f"Total rows in R2 orphan inventory: {len(r2_orphans)}")

# Let's inspect the superseded pair IDs
by_pair = {}
for row in r2_orphans:
    pid = row["associated_superseded_pair_id"]
    if pid not in by_pair:
        by_pair[pid] = []
    by_pair[pid].append(row)

print(f"Superseded pair IDs count: {len(by_pair)}")
for pid, flist in sorted(by_pair.items()):
    cats = [x["file_category"] for x in flist]
    print(f"  Pair {pid} ({len(flist)} files): {dict((c, cats.count(c)) for c in set(cats))}")

