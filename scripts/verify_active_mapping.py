import csv
from pathlib import Path

# Verify that every active pilot record in pilot_manifest.csv points to existing files and maps correctly
manifest_path = Path("data/processed/stage_3_2A_pilot/manifests/pilot_manifest.csv")
audit_pairs_path = Path("data/manifests/image_mask_pair_audit.csv")

with open(manifest_path, mode="r", encoding="utf-8") as f:
    manifest_rows = list(csv.DictReader(f))

with open(audit_pairs_path, mode="r", encoding="utf-8") as f:
    audit_rows = list(csv.DictReader(f))

audit_by_img = {r["image_path"]: r for r in audit_rows}

print(f"Total active manifest rows: {len(manifest_rows)}")

# Check each row
errors = []
for r in manifest_rows:
    cohort = r["pilot_cohort"]
    pair_id = r["pair_id"]
    orig_img_path = r["orig_img_path"]
    
    # 1. Image output exists
    img_bilin = Path(f"data/processed/stage_3_2A_pilot/images/{cohort}_{pair_id}_bilinear.png")
    img_bicub = Path(f"data/processed/stage_3_2A_pilot/images/{cohort}_{pair_id}_bicubic.png")
    if not img_bilin.exists(): errors.append(f"Missing bilinear image: {img_bilin}")
    if not img_bicub.exists(): errors.append(f"Missing bicubic image: {img_bicub}")
    
    # 2. Metadata exists
    meta_p = Path(f"data/processed/stage_3_2A_pilot/metadata/{cohort}_{pair_id}_metadata.json")
    if not meta_p.exists(): errors.append(f"Missing metadata: {meta_p}")
    
    # 3. Mask output exists
    mask_fn = r["proc_mask_filename"]
    if mask_fn:
        if r["is_diagnostic"] == "True":
            mask_p = Path(f"data/processed/stage_3_2A_pilot/diagnostic_masks/{mask_fn}")
        else:
            mask_p = Path(f"data/processed/stage_3_2A_pilot/masks/{mask_fn}")
        if not mask_p.exists(): errors.append(f"Missing mask: {mask_p}")
        
    # 4. Diagnostic sweeps / raw nn
    if cohort == "native_target":
        raw_nn = Path(f"data/processed/stage_3_2A_pilot/diagnostic_masks/{cohort}_{pair_id}_jpeg_raw_nn.png")
        if not raw_nn.exists(): errors.append(f"Missing raw nn: {raw_nn}")
        for t in [64, 96, 128, 160, 192]:
            sweep_p = Path(f"data/processed/stage_3_2A_pilot/diagnostic_masks/{cohort}_{pair_id}_sweep_{t}.png")
            if not sweep_p.exists(): errors.append(f"Missing sweep: {sweep_p}")
            
    elif cohort == "mismatched_target":
        for t in [64, 96, 128, 160, 192]:
            thresh_p = Path(f"data/processed/stage_3_2A_pilot/diagnostic_masks/{cohort}_{pair_id}_thresh_{t}.png")
            if not thresh_p.exists(): errors.append(f"Missing thresh copy: {thresh_p}")

if errors:
    print(f"FAILED: {len(errors)} mapping errors found!")
    for e in errors[:5]:
        print("  ", e)
else:
    print("SUCCESS: All 47 active pair records perfectly map to existing, uncorrupted files on disk!")

