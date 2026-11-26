"""
Visual Alignment and Overlay Verification Script for Stage 3.2A.
Generates RGB composite overlays (green mask on grayscale/RGB image)
and saves them to results/temporary/stage_3_2A/visual_overlays/
"""

import csv
import numpy as np
from pathlib import Path
from PIL import Image

def generate_overlays():
    project_root = Path("E:/DFU_Cross_Domain_Research")
    manifest_csv = project_root / "data/processed/stage_3_2A_pilot/manifests/pilot_manifest.csv"
    out_dir = project_root / "results/temporary/stage_3_2A/visual_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(manifest_csv, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        
    print(f"[INFO] Generating visual overlays for {len(rows)} samples...")
    
    for row in rows:
        cohort = row["pilot_cohort"]
        pair_id = row["pair_id"]
        
        # Load processed bilinear image
        img_path = project_root / "data/processed/stage_3_2A_pilot/images" / f"{cohort}_{pair_id}_bilinear.png"
        img = Image.open(img_path).convert("RGB")
        img_arr = np.array(img, dtype=np.uint8)
        
        # Check mask
        mask_filename = row["proc_mask_filename"]
        if mask_filename:
            if row["is_diagnostic"] == "True":
                mask_path = project_root / "data/processed/stage_3_2A_pilot/diagnostic_masks" / mask_filename
            else:
                mask_path = project_root / "data/processed/stage_3_2A_pilot/masks" / mask_filename
                
            mask = Image.open(mask_path).convert("L")
            mask_arr = np.array(mask, dtype=np.uint8)
            
            # Create RGB overlay: image + semi-transparent green mask where mask > 0
            overlay = img_arr.copy()
            fg_mask = (mask_arr > 0)
            
            # Blend green [0, 255, 0] with 50% opacity
            overlay[fg_mask, 0] = (0.5 * overlay[fg_mask, 0] + 0.5 * 0).astype(np.uint8)
            overlay[fg_mask, 1] = (0.5 * overlay[fg_mask, 1] + 0.5 * 255).astype(np.uint8)
            overlay[fg_mask, 2] = (0.5 * overlay[fg_mask, 2] + 0.5 * 0).astype(np.uint8)
            
            # Save side-by-side: Image | Mask | Overlay
            canvas = Image.new("RGB", (512 * 3, 512))
            canvas.paste(img, (0, 0))
            canvas.paste(mask.convert("RGB"), (512, 0))
            canvas.paste(Image.fromarray(overlay), (1024, 0))
            
            out_file = out_dir / f"overlay_{cohort}_{pair_id}.png"
            canvas.save(out_file, format="PNG")
        else:
            # Healthy controls: side-by-side Image | Black Mask | Image
            canvas = Image.new("RGB", (512 * 3, 512))
            canvas.paste(img, (0, 0))
            canvas.paste(Image.new("RGB", (512, 512), (0, 0, 0)), (512, 0))
            canvas.paste(img, (1024, 0))
            out_file = out_dir / f"overlay_{cohort}_{pair_id}.png"
            canvas.save(out_file, format="PNG")
            
    print(f"[INFO] Visual overlays saved to {out_dir}")

if __name__ == "__main__":
    generate_overlays()

