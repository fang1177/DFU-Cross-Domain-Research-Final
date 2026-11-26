import csv
import hashlib
from pathlib import Path

pilot_dir = Path("data/processed/stage_3_2A_pilot")
manifest_path = pilot_dir / "manifests/pilot_manifest.csv"
output_csv = Path("results/temporary/stage_3_2A_R3/orphan_inventory_comparison.csv")

with open(manifest_path, mode="r", encoding="utf-8") as f:
    active_rows = list(csv.DictReader(f))

active_pids = set(r["pair_id"] for r in active_rows)

# Collect all files on disk
all_files = []
for p in sorted(pilot_dir.rglob("*")):
    if p.is_file():
        rel = str(p.relative_to(pilot_dir)).replace("\\", "/")
        parent = p.parent.name
        fn = p.name
        size = p.stat().st_size
        
        # sha256
        h = hashlib.sha256()
        with open(p, "rb") as bf:
            while chunk := bf.read(1024 * 1024):
                h.update(chunk)
        digest = h.hexdigest()
        
        all_files.append({
            "rel_path": rel,
            "filename": fn,
            "parent_dir": parent,
            "size": size,
            "sha256": digest
        })

# Classify each file
rows_out = []
for f in all_files:
    rel = f["rel_path"]
    fn = f["filename"]
    parent = f["parent_dir"]
    
    # Check if active
    is_active = False
    matched_active_pid = None
    for pid in active_pids:
        if f"_{pid}_" in fn or f"_{pid}." in fn or fn == "pilot_manifest.csv":
            is_active = True
            matched_active_pid = pid
            break
            
    # Check if superseded
    superseded_pid = None
    for spid in ["51694882", "52233095", "53928342", "56523561", "59589913", "59728680", "61444678", "61446116", "75529004", "75529017", "75556609"]:
        if spid in fn:
            superseded_pid = spid
            break
            
    # What did Stage 3.2A-R classify this as?
    # In 3.2A-R:
    # images for superseded: orphan (22)
    # masks for superseded: orphan (9)
    # metadata for superseded: orphan (11)
    # diagnostic: 3.2A-R erroneously thought only 2 were orphan and 169 were active!
    status_in_3_2a_r = "Active"
    if superseded_pid:
        if parent in ["images", "masks", "metadata"]:
            status_in_3_2a_r = "Orphan"
        elif parent == "diagnostic_masks":
            # 3.2A-R reported 2 diagnostic orphans (for 56523561)
            if "56523561" in fn:
                status_in_3_2a_r = "Orphan"
            else:
                status_in_3_2a_r = "Misclassified as Active"
                
    # What did Stage 3.2A-R2 classify this as?
    status_in_3_2a_r2 = "Active" if is_active else "Orphan"
    
    # Ground truth status
    ground_truth_status = "Active" if is_active else "Orphan"
    
    rows_out.append({
        "relative_path": f"data/processed/stage_3_2A_pilot/{rel}",
        "filename": fn,
        "parent_directory": parent,
        "file_size_bytes": f["size"],
        "sha256": f["sha256"],
        "ground_truth_status": ground_truth_status,
        "status_in_stage_3_2A_R": status_in_3_2a_r,
        "status_in_stage_3_2A_R2": status_in_3_2a_r2,
        "associated_pair_id": matched_active_pid if is_active else superseded_pid,
        "classification_explanation": "Referenced in active pilot_manifest.csv (47 pairs)" if is_active else f"Superseded Iteration 1 artifact for pair {superseded_pid}; unreferenced in active manifest"
    })

fieldnames = [
    "relative_path", "filename", "parent_directory", "file_size_bytes", "sha256",
    "ground_truth_status", "status_in_stage_3_2A_R", "status_in_stage_3_2A_R2",
    "associated_pair_id", "classification_explanation"
]

output_csv.parent.mkdir(parents=True, exist_ok=True)
with open(output_csv, mode="w", newline="", encoding="utf-8") as out_f:
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_out)

print(f"Wrote {len(rows_out)} comparison rows to {output_csv}")
gt_active = len([r for r in rows_out if r["ground_truth_status"] == "Active"])
gt_orphan = len([r for r in rows_out if r["ground_truth_status"] == "Orphan"])
print(f"Ground truth breakdown: {gt_active} Active, {gt_orphan} Orphan (Total {gt_active + gt_orphan})")

