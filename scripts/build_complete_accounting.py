import csv
import hashlib
from pathlib import Path

pilot_dir = Path("data/processed/stage_3_2A_pilot")
manifest_path = pilot_dir / "manifests/pilot_manifest.csv"
output_csv = Path("results/temporary/stage_3_2A_R3/complete_pilot_file_accounting.csv")

with open(manifest_path, mode="r", encoding="utf-8") as f:
    active_rows = list(csv.DictReader(f))

active_pids = set(r["pair_id"] for r in active_rows)

accounting_rows = []
for p in sorted(pilot_dir.rglob("*")):
    if p.is_file():
        rel = str(p.relative_to(pilot_dir)).replace("\\", "/")
        parent = p.parent.name
        fn = p.name
        size = p.stat().st_size
        
        h = hashlib.sha256()
        with open(p, "rb") as bf:
            while chunk := bf.read(1024 * 1024):
                h.update(chunk)
        digest = h.hexdigest()
        
        # Check if active
        is_active = False
        matched_pid = None
        for pid in active_pids:
            if f"_{pid}_" in fn or f"_{pid}." in fn or fn == "pilot_manifest.csv":
                is_active = True
                matched_pid = pid
                break
                
        # If not active, find superseded pid
        if not is_active:
            for spid in ["51694882", "52233095", "53928342", "56523561", "59589913", "59728680", "61444678", "61446116", "75529004", "75529017", "75556609"]:
                if spid in fn:
                    matched_pid = spid
                    break
                    
        # File type / content role
        if parent == "images":
            role = "RGB Letterboxed Image (Bilinear)" if "bilinear" in fn else "RGB Letterboxed Image (Bicubic)"
        elif parent == "masks":
            role = "Canonical / Reference Binary Mask"
        elif parent == "diagnostic_masks":
            if "raw_nn" in fn:
                role = "Diagnostic Unthresholded Raw Nearest-Neighbor Mask"
            elif "sweep" in fn:
                role = "Diagnostic Threshold Sweep Binary Mask"
            elif "thresh" in fn:
                role = "Diagnostic Threshold Copy Mask"
            elif "multiclass" in fn:
                role = "Diagnostic Multiclass Raw Mask"
            elif "continuous" in fn:
                role = "Diagnostic Continuous Grayscale Mask"
            else:
                role = "Diagnostic Mask Artifact"
        elif parent == "manifests":
            role = "Pilot Provenance Manifest CSV"
        elif parent == "metadata":
            role = "Per-Sample Provenance Metadata JSON"
            
        accounting_rows.append({
            "relative_path": f"data/processed/stage_3_2A_pilot/{rel}",
            "filename": fn,
            "subdirectory": parent,
            "content_role": role,
            "file_size_bytes": size,
            "sha256": digest,
            "lifecycle_status": "ACTIVE" if is_active else "ORPHAN_SUPERSEDED",
            "associated_pair_id": matched_pid,
            "in_active_manifest": "True" if is_active else "False"
        })

fieldnames = [
    "relative_path", "filename", "subdirectory", "content_role",
    "file_size_bytes", "sha256", "lifecycle_status",
    "associated_pair_id", "in_active_manifest"
]

with open(output_csv, mode="w", newline="", encoding="utf-8") as out_f:
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(accounting_rows)

print(f"Wrote complete accounting of {len(accounting_rows)} files to {output_csv}")

