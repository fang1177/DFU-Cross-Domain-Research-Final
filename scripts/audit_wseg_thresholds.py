import csv
from pathlib import Path
from PIL import Image
import numpy as np
from scipy.ndimage import label

project_root = Path("E:/DFU_Cross_Domain_Research")
manifest_csv = project_root / "data/processed/stage_3_2A_pilot/manifests/pilot_manifest.csv"

with open(manifest_csv, "r") as f:
    rows = list(csv.DictReader(f))

wseg_rows = [r for r in rows if r["pilot_cohort"] == "native_target"]

print(f"Auditing all {len(wseg_rows)} WSEG native target masks:\n")

for r in wseg_rows:
    pair_id = r["pair_id"]
    p = project_root / "data/interim/stage_3_1/extracted/WSEG" / r["orig_mask_path"]
    arr = np.array(Image.open(p).convert("L"))
    uniques = np.unique(arr)
    
    # Check original counts
    orig_counts = {t: int((arr >= t).sum()) for t in [64, 96, 128, 160, 192]}
    
    # Check processed letterboxed counts (on 512x512 raw NN)
    raw_nn_p = project_root / "data/processed/stage_3_2A_pilot/diagnostic_masks" / f"native_target_{pair_id}_jpeg_raw_nn.png"
    arr_nn = np.array(Image.open(raw_nn_p).convert("L"))
    proc_counts = {t: int((arr_nn >= t).sum()) for t in [64, 96, 128, 160, 192]}
    
    # Connected components at each threshold on processed mask
    ccs = {}
    for t in [64, 96, 128, 160, 192]:
        bin_m = (arr_nn >= t).astype(np.uint8)
        _, num_f = label(bin_m)
        ccs[t] = num_f
        
    diff_orig = max(orig_counts.values()) - min(orig_counts.values())
    diff_proc = max(proc_counts.values()) - min(proc_counts.values())
    
    print(f"Pair ID: {pair_id}")
    print(f"  Orig unique values count: {len(uniques)} (min: {uniques.min()}, max: {uniques.max()})")
    print(f"  Orig counts [64..192]: {orig_counts} (diff={diff_orig})")
    print(f"  Proc counts [64..192]: {proc_counts} (diff={diff_proc})")
    print(f"  Proc Connected Components: {ccs}")
    print()

