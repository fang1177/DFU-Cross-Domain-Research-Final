"""
Double-Run Determinism Verification Script for Stage 3.2A Pilot.
Reruns the preprocessing pilot and asserts 100% bitwise SHA-256 equivalence across all generated outputs.
"""

import sys
import csv
import hashlib
from pathlib import Path

project_root = Path("E:/DFU_Cross_Domain_Research")
sys.path.insert(0, str(project_root))

from scripts.data.run_preprocessing_pilot import run_pilot

def get_dir_hashes(directory: Path):
    hashes = {}
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            h = hashlib.sha256()
            with open(p, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    h.update(chunk)
            hashes[str(p.relative_to(directory)).replace("\\", "/")] = h.hexdigest()
    return hashes

def main():
    pilot_dir = project_root / "data/processed/stage_3_2A_pilot"
    print("[INFO] Computing baseline hashes from run 1...")
    baseline_hashes = get_dir_hashes(pilot_dir)
    print(f"[INFO] Indexed {len(baseline_hashes)} baseline files.")
    
    print("[INFO] Executing Run 2 of Preprocessing Pilot...")
    run_pilot()
    
    print("[INFO] Computing verification hashes from run 2...")
    second_run_hashes = get_dir_hashes(pilot_dir)
    print(f"[INFO] Indexed {len(second_run_hashes)} second run files.")
    
    mismatches = []
    for fpath, h1 in baseline_hashes.items():
        if fpath not in second_run_hashes:
            mismatches.append((fpath, "MISSING_IN_RUN2", h1))
        elif second_run_hashes[fpath] != h1:
            mismatches.append((fpath, "HASH_MISMATCH", h1, second_run_hashes[fpath]))
            
    if mismatches:
        print(f"[ERROR] Non-deterministic execution detected! {len(mismatches)} mismatches:")
        for m in mismatches:
            print(" ", m)
        sys.exit(1)
    else:
        print("[SUCCESS] 100% Deterministic Reproducibility Confirmed! All files match bitwise across consecutive runs.")

if __name__ == "__main__":
    main()

