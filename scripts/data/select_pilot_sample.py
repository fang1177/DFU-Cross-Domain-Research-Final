"""
Pilot Sample Selection Script for Stage 3.2A.
Strictly reproducible with seed 42.
Ensures EXACT unique content images:
- 12 Source Training unique images (2 multi-mask clusters [0111/0386/0889 and 0087/0635], empty mask 0092, + 9 single images)
- 6 Internal Validation unique images (multi-mask cluster 0303/0339, empty mask 0128, diagnostic multiclass 0233, + 3 clean holdout images)
- 12 Native Target unique images (WSEG 1:1 resolution, JPEG-artefact masks, multi-mask cluster 49422580/49498192, + 11 clean 1:1 JPEG single images)
- 6 Mismatched Target unique images (Mendeley wound_main, 331x331 vs 224x224, continuous grayscale, diagnostic only)
- 6 Healthy Controls (Mendeley Nomal/)
"""

import csv
import json
import random
from pathlib import Path
from collections import defaultdict

def main():
    random.seed(42)
    
    project_root = Path("E:/DFU_Cross_Domain_Research")
    audit_pairs_csv = project_root / "data/manifests/image_mask_pair_audit.csv"
    mendeley_manifest_csv = project_root / "data/interim/stage_3_1/extraction_manifests/mendeley_extraction_manifest.csv"
    source_map_csv = project_root / "results/temporary/stage_1d_r2/source_training_duplicate_mapping.csv"
    val_recon_csv = project_root / "results/temporary/stage_1d_r2/validation_record_reconciliation.csv"
    output_csv = project_root / "results/temporary/stage_3_2A/pilot_sample_selection.csv"
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    with open(audit_pairs_csv, mode="r", encoding="utf-8") as f:
        all_pairs = list(csv.DictReader(f))
    pairs_by_path = {p["image_path"]: p for p in all_pairs}
    
    with open(source_map_csv, mode="r", encoding="utf-8") as f:
        source_dups = list(csv.DictReader(f))
    source_clusters = defaultdict(list)
    for r in source_dups:
        source_clusters[r["image_sha256"]].append(r)
        
    with open(val_recon_csv, mode="r", encoding="utf-8") as f:
        val_recons = list(csv.DictReader(f))
        
    def get_disk_paths(dataset_key, img_rel, mask_rel):
        if dataset_key == "AZH":
            img_disk = Path("data/interim/stage_3_1/extracted/AZH") / img_rel
            mask_disk = Path("data/interim/stage_3_1/extracted/AZH") / mask_rel if mask_rel else Path("")
        elif dataset_key == "WSEG":
            img_disk = Path("data/interim/stage_3_1/extracted/WSEG") / img_rel
            mask_disk = Path("data/interim/stage_3_1/extracted/WSEG") / mask_rel if mask_rel else Path("")
        elif dataset_key == "MENDELEY":
            img_disk = Path("data/interim/stage_3_1/extracted") / img_rel
            mask_disk = Path("data/interim/stage_3_1/extracted") / mask_rel if mask_rel else Path("")
            if not (project_root / img_disk).exists():
                img_disk = Path("data/interim/stage_3_1/extracted/MENDELEY") / img_rel
            if mask_rel and not (project_root / mask_disk).exists():
                mask_disk = Path("data/interim/stage_3_1/extracted/MENDELEY") / mask_rel
        return str(img_disk).replace("\\", "/"), str(mask_disk).replace("\\", "/") if mask_rel else ""

    selected_records = []
    
    # 1. Source Training: 12 unique content images
    # Image 1: Cluster 0111 (pairs 0111, 0386, 0889)
    # Image 2: Cluster 0087 (pairs 0087, 0635)
    # Image 3: Empty mask 0092 (pair 0092)
    # Images 4-12: 9 clean single pairs
    c_0111_sha = [r for r in source_dups if "0111" in r["pair_id"]][0]["image_sha256"]
    for r in source_clusters[c_0111_sha]:
        p = dict(pairs_by_path[r["image_path"]])
        p["pilot_cohort"] = "source_training"
        p["cluster_id"] = "source_cluster_0111"
        p["unique_content_id"] = "source_img_0111"
        p["is_primary_reference"] = "True" if r["pair_id"] == "0111" else "False"
        img_d, mask_d = get_disk_paths("AZH", p["image_path"], p["mask_path"])
        p["image_disk_path"] = img_d
        p["mask_disk_path"] = mask_d
        selected_records.append(p)
        
    c_0087_sha = [r for r in source_dups if "0087" in r["pair_id"]][0]["image_sha256"]
    for r in source_clusters[c_0087_sha]:
        p = dict(pairs_by_path[r["image_path"]])
        p["pilot_cohort"] = "source_training"
        p["cluster_id"] = "source_cluster_0087"
        p["unique_content_id"] = "source_img_0087"
        p["is_primary_reference"] = "True" if r["pair_id"] == "0087" else "False"
        img_d, mask_d = get_disk_paths("AZH", p["image_path"], p["mask_path"])
        p["image_disk_path"] = img_d
        p["mask_disk_path"] = mask_d
        selected_records.append(p)
        
    p_0092 = dict(pairs_by_path["wound-segmentation-master/data/Foot Ulcer Segmentation Challenge/train/images/0092.png"])
    p_0092["pilot_cohort"] = "source_training"
    p_0092["cluster_id"] = "source_cluster_0092"
    p_0092["unique_content_id"] = "source_img_0092"
    p_0092["is_primary_reference"] = "True"
    img_d, mask_d = get_disk_paths("AZH", p_0092["image_path"], p_0092["mask_path"])
    p_0092["image_disk_path"] = img_d
    p_0092["mask_disk_path"] = mask_d
    selected_records.append(p_0092)
    
    dup_paths = set(r["image_path"] for r in source_dups)
    clean_single_source = [
        p for p in all_pairs 
        if p["subset"] in ["AZH_Challenge_Train", "AZH_Medetec_Train"] 
        and p["image_path"] not in dup_paths 
        and float(p["mask_foreground_pct"]) > 0
    ]
    clean_single_source = sorted(clean_single_source, key=lambda x: x["image_path"])
    random.seed(42)
    random.shuffle(clean_single_source)
    for i, p in enumerate(clean_single_source[:9]):
        p_copy = dict(p)
        p_copy["pilot_cohort"] = "source_training"
        p_copy["cluster_id"] = f"source_single_{p_copy['pair_id']}"
        p_copy["unique_content_id"] = f"source_single_{p_copy['pair_id']}"
        p_copy["is_primary_reference"] = "True"
        img_d, mask_d = get_disk_paths("AZH", p_copy["image_path"], p_copy["mask_path"])
        p_copy["image_disk_path"] = img_d
        p_copy["mask_disk_path"] = mask_d
        selected_records.append(p_copy)
        
    # 2. Internal Validation: 6 unique content images
    # Image 1: Cluster 0303/0339 (pairs 0303, 0339)
    # Image 2: Empty mask 0128 (pair 0128)
    # Image 3: Multiclass 0233 (pair 0233)
    # Images 4-6: 3 clean holdout images
    p_0303 = dict(pairs_by_path["wound-segmentation-master/data/Foot Ulcer Segmentation Challenge/validation/images/0303.png"])
    p_0303["pilot_cohort"] = "internal_validation"
    p_0303["cluster_id"] = "val_cluster_0303_0339"
    p_0303["unique_content_id"] = "val_img_0303"
    p_0303["is_primary_reference"] = "True"
    img_d, mask_d = get_disk_paths("AZH", p_0303["image_path"], p_0303["mask_path"])
    p_0303["image_disk_path"] = img_d
    p_0303["mask_disk_path"] = mask_d
    selected_records.append(p_0303)
    
    p_0339 = dict(pairs_by_path["wound-segmentation-master/data/Foot Ulcer Segmentation Challenge/validation/images/0339.png"])
    p_0339["pilot_cohort"] = "internal_validation"
    p_0339["cluster_id"] = "val_cluster_0303_0339"
    p_0339["unique_content_id"] = "val_img_0303"
    p_0339["is_primary_reference"] = "False"
    img_d, mask_d = get_disk_paths("AZH", p_0339["image_path"], p_0339["mask_path"])
    p_0339["image_disk_path"] = img_d
    p_0339["mask_disk_path"] = mask_d
    selected_records.append(p_0339)
    
    p_0128 = dict(pairs_by_path["wound-segmentation-master/data/Foot Ulcer Segmentation Challenge/validation/images/0128.png"])
    p_0128["pilot_cohort"] = "internal_validation"
    p_0128["cluster_id"] = "val_single_0128"
    p_0128["unique_content_id"] = "val_img_0128"
    p_0128["is_primary_reference"] = "True"
    img_d, mask_d = get_disk_paths("AZH", p_0128["image_path"], p_0128["mask_path"])
    p_0128["image_disk_path"] = img_d
    p_0128["mask_disk_path"] = mask_d
    selected_records.append(p_0128)
    
    p_0233 = dict(pairs_by_path["wound-segmentation-master/data/Foot Ulcer Segmentation Challenge/validation/images/0233.png"])
    p_0233["pilot_cohort"] = "internal_validation"
    p_0233["cluster_id"] = "val_single_0233"
    p_0233["unique_content_id"] = "val_img_0233"
    p_0233["is_primary_reference"] = "True"
    img_d, mask_d = get_disk_paths("AZH", p_0233["image_path"], p_0233["mask_path"])
    p_0233["image_disk_path"] = img_d
    p_0233["mask_disk_path"] = mask_d
    selected_records.append(p_0233)
    
    clean_val = [
        r for r in val_recons 
        if r["train_overlap_status"] == "INDEPENDENT" 
        and r["pair_id"] not in ["0303", "0339", "0128", "0233"]
    ]
    clean_val = sorted(clean_val, key=lambda x: x["pair_id"])
    random.seed(42)
    random.shuffle(clean_val)
    for r in clean_val[:3]:
        p = dict(pairs_by_path[r["image_path"]])
        p["pilot_cohort"] = "internal_validation"
        p["cluster_id"] = f"val_single_{p['pair_id']}"
        p["unique_content_id"] = f"val_single_{p['pair_id']}"
        p["is_primary_reference"] = "True"
        img_d, mask_d = get_disk_paths("AZH", p["image_path"], p["mask_path"])
        p["image_disk_path"] = img_d
        p["mask_disk_path"] = mask_d
        selected_records.append(p)
        
    # 3. Native Target (WSEG): 12 unique content images
    # Image 1: Cluster 49422580 / 49498192 (2 pairs, native 1:1, JPEG artefact)
    # Images 2-12: 11 clean 1:1 WSEG JPEG artefact images
    p_4942 = dict(pairs_by_path["wseg_dataset/sample/49422580.jpg"])
    p_4942["pilot_cohort"] = "native_target"
    p_4942["cluster_id"] = "wseg_cluster_49422580"
    p_4942["unique_content_id"] = "wseg_img_49422580"
    p_4942["is_primary_reference"] = "True"
    img_d, mask_d = get_disk_paths("WSEG", p_4942["image_path"], p_4942["mask_path"])
    p_4942["image_disk_path"] = img_d
    p_4942["mask_disk_path"] = mask_d
    selected_records.append(p_4942)
    
    p_4949 = dict(pairs_by_path["wseg_dataset/sample/49498192.jpg"])
    p_4949["pilot_cohort"] = "native_target"
    p_4949["cluster_id"] = "wseg_cluster_49422580"
    p_4949["unique_content_id"] = "wseg_img_49422580"
    p_4949["is_primary_reference"] = "False"
    img_d, mask_d = get_disk_paths("WSEG", p_4949["image_path"], p_4949["mask_path"])
    p_4949["image_disk_path"] = img_d
    p_4949["mask_disk_path"] = mask_d
    selected_records.append(p_4949)
    
    # Filter strictly to WSEG pairs that are native 1:1 match and BINARY_LIKE_JPEG_ARTEFACTS
    wseg_clean_1to1 = [
        p for p in all_pairs 
        if p["dataset_key"] == "WSEG" 
        and p["dimensions_match"] == "True"
        and p["mask_type"] == "BINARY_LIKE_JPEG_ARTEFACTS"
        and p["pair_id"] not in ["49422580", "49498192"]
    ]
    wseg_clean_1to1 = sorted(wseg_clean_1to1, key=lambda x: x["pair_id"])
    random.seed(42)
    random.shuffle(wseg_clean_1to1)
    for p in wseg_clean_1to1[:11]:
        p_copy = dict(p)
        p_copy["pilot_cohort"] = "native_target"
        p_copy["cluster_id"] = f"wseg_single_{p_copy['pair_id']}"
        p_copy["unique_content_id"] = f"wseg_single_{p_copy['pair_id']}"
        p_copy["is_primary_reference"] = "True"
        img_d, mask_d = get_disk_paths("WSEG", p_copy["image_path"], p_copy["mask_path"])
        p_copy["image_disk_path"] = img_d
        p_copy["mask_disk_path"] = mask_d
        selected_records.append(p_copy)
        
    # 4. Mismatched Target (Mendeley): 6 unique content images
    mendeley_pairs = [p for p in all_pairs if p["dataset_key"] == "MENDELEY"]
    mendeley_pairs = sorted(mendeley_pairs, key=lambda x: x["pair_id"])
    random.seed(42)
    random.shuffle(mendeley_pairs)
    for p in mendeley_pairs[:6]:
        p_copy = dict(p)
        p_copy["pilot_cohort"] = "mismatched_target"
        p_copy["cluster_id"] = f"mendeley_single_{p_copy['pair_id']}"
        p_copy["unique_content_id"] = f"mendeley_single_{p_copy['pair_id']}"
        p_copy["is_primary_reference"] = "True"
        img_d, mask_d = get_disk_paths("MENDELEY", p_copy["image_path"], p_copy["mask_path"])
        p_copy["image_disk_path"] = img_d
        p_copy["mask_disk_path"] = mask_d
        selected_records.append(p_copy)
        
    # 5. Healthy Controls (Mendeley Nomal/): 6 unique content images
    with open(mendeley_manifest_csv, mode="r", encoding="utf-8") as f:
        mendeley_manifest = list(csv.DictReader(f))
    normal_entries = [m for m in mendeley_manifest if "nomal" in m["extracted_relative_path"].lower() and m["extension"] in [".jpg", ".png", ".jpeg"]]
    normal_entries = sorted(normal_entries, key=lambda x: x["extracted_relative_path"])
    random.seed(42)
    random.shuffle(normal_entries)
    for norm in normal_entries[:6]:
        img_disk = f"data/interim/stage_3_1/extracted/{norm['extracted_relative_path']}"
        rec = {
            "dataset_key": "MENDELEY",
            "subset": "Nomal",
            "pair_id": norm["filename"],
            "normalized_pair_id": norm["filename"],
            "pair_status": "healthy_control_no_mask",
            "dimensions_match": "N/A",
            "image_width": "N/A",
            "image_height": "N/A",
            "mask_width": "N/A",
            "mask_height": "N/A",
            "mask_type": "none",
            "foreground_estimation_status": "authentic_negative",
            "mask_foreground_pct": "0.0",
            "requires_visual_review": "False",
            "review_reason": "none",
            "image_path": norm["extracted_relative_path"],
            "mask_path": "",
            "pilot_cohort": "healthy_controls",
            "cluster_id": f"healthy_{norm['filename']}",
            "unique_content_id": f"healthy_{norm['filename']}",
            "is_primary_reference": "True",
            "image_disk_path": img_disk,
            "mask_disk_path": ""
        }
        selected_records.append(rec)
        
    # Verify all disk paths exist
    print("Verifying disk paths for all selected records...")
    missing_paths = []
    for r in selected_records:
        img_file = project_root / r["image_disk_path"]
        if not img_file.exists():
            missing_paths.append(str(img_file))
        if r["mask_disk_path"]:
            mask_file = project_root / r["mask_disk_path"]
            if not mask_file.exists():
                missing_paths.append(str(mask_file))
    if missing_paths:
        raise RuntimeError(f"Missing disk paths: {missing_paths}")
    else:
        print("All 47 records have verified existing disk files!")
        
    # Save CSV
    fieldnames = list(selected_records[0].keys())
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_records)
        
    print(f"Total records written: {len(selected_records)}")
    cohort_content_counts = defaultdict(set)
    for r in selected_records:
        cohort_content_counts[r["pilot_cohort"]].add(r["unique_content_id"])
    for c, s in cohort_content_counts.items():
        recs = [r for r in selected_records if r["pilot_cohort"] == c]
        print(f"  - {c}: {len(s)} unique content images, {len(recs)} pair records")

if __name__ == "__main__":
    main()
