"""
Stage 1B: Detailed Dataset Structure Audit Utility
==================================================

Performs in-depth, non-destructive structural auditing of the three dataset
ZIP archives in project storage:
  - data/raw/AZH/archives/wound-segmentation-master.zip
  - data/raw/WSEG/archives/wseg_dataset.zip
  - data/raw/MENDELEY/archives/Wound Image Dataset.zip

Key Audits & Features:
  1. Complete Directory Hierarchy:
     - Derives and records all ancestor directories for every file path, plus
       explicit empty directory entries from the ZIP central directory.
     - Preserves unique directories without duplicates.
  2. Complete File Inventory:
     - For every archive, records every single file entry with its complete
       archive-relative path, basename, extension, parent directory, compressed
       size in bytes, and uncompressed size in bytes.
     - Never decodes or reads file payloads.
  3. AZH Pairing:
     - Compares image and label directories using stem groupings (multimaps)
       to prevent overwriting duplicate stems.
     - Reports duplicate stems, matches, and unmatched filenames.
     - Identifies nested ZIP files (path, compressed/uncompressed sizes).
  4. WSEG Pairing & Precise Basename Audit:
     - Tracks WSEG basenames separately for sample/ and mask/ trees.
     - Records occurrence count, relative paths, and unique parent directories.
     - Reports duplicate basenames within sample/, within mask/, basenames in both
       sample and mask, basenames appearing across multiple distinct parent directories,
       and duplicate full relative paths in the ZIP central directory.
     - Preserves exact relative-path matching and unmatched analysis.
  5. Mendeley Pairing:
     - Matches numeric identifiers between 'wound_main/' and 'wound_mask/'.
     - Reports duplicate and unmatched identifiers while preserving filenames.
     - Preserves clinical caveat regarding wound etiology.
  6. Robust Error Handling:
     - If an archive is missing, reports ARCHIVE_NOT_FOUND and continues.
     - If an archive cannot be opened (bad ZIP/permission), reports a clear error
       and continues auditing the remaining archives without crashing.
  7. Strict Safeguards:
     - Standard-library Python only.
     - Anchors all paths to PROJECT_ROOT.
     - Protects existing output reports from unintentional overwrites.
     - Zero extraction, zero payload decoding, zero full CRC scan.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import zipfile


# Anchor project root to script file location
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent

DEFAULT_OUTPUT_JSON_PATH = PROJECT_ROOT / "data" / "manifests" / "dataset_structure_audit.json"

TARGET_ARCHIVES = [
    {
        "dataset_key": "AZH",
        "archive_rel": Path("data/raw/AZH/archives/wound-segmentation-master.zip"),
        "description": "AZH repository wound segmentation archive",
    },
    {
        "dataset_key": "WSEG",
        "archive_rel": Path("data/raw/WSEG/archives/wseg_dataset.zip"),
        "description": "WSEG dataset archive",
    },
    {
        "dataset_key": "MENDELEY",
        "archive_rel": Path("data/raw/MENDELEY/archives/Wound Image Dataset.zip"),
        "description": "Mendeley lower-limb wound collection archive",
    },
]


def format_size(size_bytes: int) -> str:
    """Format bytes into MB or KB string."""
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024**3):.2f} GB"
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024**2):.2f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    return f"{size_bytes} bytes"


def audit_azh_structure(zf: zipfile.ZipFile, infolist: list) -> dict:
    """
    Specific audit for AZH:
    - Identifies nested ZIP files (path, compressed/uncompressed sizes).
    - Audits image-to-label pairing using stem grouping multimaps to prevent overwriting duplicates.
    """
    nested_zips = []
    cohort_dirs = defaultdict(lambda: {"files": 0, "extensions": Counter()})

    for info in infolist:
        norm = info.filename.replace("\\", "/")
        if info.is_dir() or norm.endswith("/"):
            continue

        p = Path(norm)
        # Check for nested zip files
        if p.suffix.lower() == ".zip":
            nested_zips.append({
                "path": norm,
                "compressed_size_bytes": info.compress_size,
                "compressed_size_formatted": format_size(info.compress_size),
                "uncompressed_size_bytes": info.file_size,
                "uncompressed_size_formatted": format_size(info.file_size),
            })

        # Categorize by sub-cohort
        parts = [part for part in norm.split("/") if part]
        if len(parts) >= 3 and parts[1] == "data":
            cohort_name = parts[2]
            cohort_dirs[cohort_name]["files"] += 1
            cohort_dirs[cohort_name]["extensions"][p.suffix.lower()] += 1

    candidate_subsets = [
        ("Foot Ulcer Challenge - Train", "wound-segmentation-master/data/Foot Ulcer Segmentation Challenge/train/images", "wound-segmentation-master/data/Foot Ulcer Segmentation Challenge/train/labels"),
        ("Foot Ulcer Challenge - Val", "wound-segmentation-master/data/Foot Ulcer Segmentation Challenge/validation/images", "wound-segmentation-master/data/Foot Ulcer Segmentation Challenge/validation/labels"),
        ("Medetec 224 - Train", "wound-segmentation-master/data/Medetec_foot_ulcer_224/train/images", "wound-segmentation-master/data/Medetec_foot_ulcer_224/train/labels"),
        ("Medetec 224 - Test", "wound-segmentation-master/data/Medetec_foot_ulcer_224/test/images", "wound-segmentation-master/data/Medetec_foot_ulcer_224/test/labels"),
    ]

    all_names = [info.filename.replace("\\", "/") for info in infolist if not info.is_dir() and not info.filename.endswith("/")]

    pairing_analysis = {}

    for label, img_prefix, mask_prefix in candidate_subsets:
        # Use lists for grouping to never overwrite duplicate stems
        img_stem_to_files = defaultdict(list)
        mask_stem_to_files = defaultdict(list)
        all_img_files = []
        all_mask_files = []

        for path in all_names:
            if path.startswith(img_prefix + "/"):
                fname = path[len(img_prefix) + 1:]
                stem = Path(fname).stem
                img_stem_to_files[stem].append(fname)
                all_img_files.append(fname)
            elif path.startswith(mask_prefix + "/"):
                fname = path[len(mask_prefix) + 1:]
                stem = Path(fname).stem
                mask_stem_to_files[stem].append(fname)
                all_mask_files.append(fname)

        img_stems = set(img_stem_to_files.keys())
        mask_stems = set(mask_stem_to_files.keys())

        matched_stems = sorted(list(img_stems.intersection(mask_stems)))
        unmatched_img_stems = sorted(list(img_stems - mask_stems))
        unmatched_mask_stems = sorted(list(mask_stems - img_stems))

        duplicate_img_stems = {stem: files for stem, files in img_stem_to_files.items() if len(files) > 1}
        duplicate_mask_stems = {stem: files for stem, files in mask_stem_to_files.items() if len(files) > 1}

        unmatched_img_filenames = [f for stem in unmatched_img_stems for f in img_stem_to_files[stem]]
        unmatched_mask_filenames = [f for stem in unmatched_mask_stems for f in mask_stem_to_files[stem]]

        pairing_analysis[label] = {
            "image_dir": img_prefix,
            "mask_dir": mask_prefix,
            "total_image_files": len(all_img_files),
            "total_mask_files": len(all_mask_files),
            "unique_image_stems_count": len(img_stems),
            "unique_mask_stems_count": len(mask_stems),
            "exact_stem_matches_count": len(matched_stems),
            "duplicate_image_stems": duplicate_img_stems,
            "duplicate_mask_stems": duplicate_mask_stems,
            "unmatched_images_count": len(unmatched_img_filenames),
            "unmatched_masks_count": len(unmatched_mask_filenames),
            "unmatched_image_samples": unmatched_img_filenames[:10],
            "unmatched_mask_samples": unmatched_mask_filenames[:10],
        }

    return {
        "nested_zip_archives": nested_zips,
        "cohort_directories": {k: {"files": v["files"], "extensions": dict(v["extensions"])} for k, v in cohort_dirs.items()},
        "candidate_pairing_analysis": pairing_analysis,
    }


def audit_wseg_structure(zf: zipfile.ZipFile, infolist: list) -> dict:
    """
    Specific audit for WSEG:
    - Tracks sample/ and mask/ relative paths and basenames separately.
    - Records occurrences, paths, and unique parent directories.
    - Reports duplicate basenames within sample/, within mask/, appearing in both,
      and across multiple distinct directories (based on unique parent directories).
    - Preserves exact relative-path matching and unmatched analysis.
    """
    # Track files by relative subpath within sample/ and mask/
    sample_subpath_to_info = defaultdict(list)
    mask_subpath_to_info = defaultdict(list)

    # Detailed basename tracking across the whole WSEG archive
    basename_details = defaultdict(lambda: {
        "occurrences": 0,
        "relative_paths": [],
        "parent_directories": set(),
        "in_sample": False,
        "in_mask": False,
    })

    # Track duplicate full relative paths in the ZIP central directory
    relative_path_counts = Counter()

    for info in infolist:
        norm = info.filename.replace("\\", "/")
        if info.is_dir() or norm.endswith("/"):
            continue

        relative_path_counts[norm] += 1
        p = Path(norm)
        base_name = p.name
        parent_dir = str(p.parent).replace("\\", "/")

        b_entry = basename_details[base_name]
        b_entry["occurrences"] += 1
        b_entry["relative_paths"].append(norm)
        b_entry["parent_directories"].add(parent_dir)

        if norm.startswith("wseg_dataset/sample/"):
            subpath = norm[len("wseg_dataset/sample/"):]
            sample_subpath_to_info[subpath].append(info)
            b_entry["in_sample"] = True
        elif norm.startswith("wseg_dataset/mask/"):
            subpath = norm[len("wseg_dataset/mask/"):]
            mask_subpath_to_info[subpath].append(info)
            b_entry["in_mask"] = True

    # 1. Exact relative path matching between sample/ and mask/
    sample_subpaths = set(sample_subpath_to_info.keys())
    mask_subpaths = set(mask_subpath_to_info.keys())

    exact_matches = sorted(list(sample_subpaths.intersection(mask_subpaths)))
    unmatched_samples = sorted(list(sample_subpaths - mask_subpaths))
    unmatched_masks = sorted(list(mask_subpaths - sample_subpaths))

    # 2. Duplicate full relative paths in the ZIP
    duplicate_relative_paths = {path: count for path, count in relative_path_counts.items() if count > 1}

    # 3. Basenames tracking & duplicate analysis
    duplicate_within_sample = {}
    duplicate_within_mask = {}
    basenames_in_both_sample_and_mask = []
    basenames_in_multiple_distinct_dirs = {}

    for base_name, data in basename_details.items():
        unique_parents = sorted(list(data["parent_directories"]))
        sample_paths = [p for p in data["relative_paths"] if p.startswith("wseg_dataset/sample/")]
        mask_paths = [p for p in data["relative_paths"] if p.startswith("wseg_dataset/mask/")]

        # Duplicate basename within sample/ tree
        if len(sample_paths) > 1:
            duplicate_within_sample[base_name] = {
                "occurrences_in_sample": len(sample_paths),
                "paths": sample_paths,
            }

        # Duplicate basename within mask/ tree
        if len(mask_paths) > 1:
            duplicate_within_mask[base_name] = {
                "occurrences_in_mask": len(mask_paths),
                "paths": mask_paths,
            }

        # Basename appears in both sample and mask trees
        if data["in_sample"] and data["in_mask"]:
            basenames_in_both_sample_and_mask.append(base_name)

        # Basename appears across multiple distinct directories (len(unique_parents) > 1)
        if len(unique_parents) > 1:
            basenames_in_multiple_distinct_dirs[base_name] = {
                "total_occurrences": data["occurrences"],
                "unique_directories_count": len(unique_parents),
                "directories": unique_parents,
                "paths": data["relative_paths"],
            }

    # Size comparison sample of 5 matched entries
    size_comparisons = []
    for subp in exact_matches[:5]:
        s_info = sample_subpath_to_info[subp][0]
        m_info = mask_subpath_to_info[subp][0]
        size_comparisons.append({
            "subpath": subp,
            "sample_uncompressed_bytes": s_info.file_size,
            "mask_uncompressed_bytes": m_info.file_size,
        })

    return {
        "sample_dir": "wseg_dataset/sample",
        "mask_dir": "wseg_dataset/mask",
        "total_sample_files": sum(len(v) for v in sample_subpath_to_info.values()),
        "total_mask_files": sum(len(v) for v in mask_subpath_to_info.values()),
        "unique_sample_relative_paths": len(sample_subpaths),
        "unique_mask_relative_paths": len(mask_subpaths),
        "exact_relative_path_matches_count": len(exact_matches),
        "unmatched_sample_files_count": len(unmatched_samples),
        "unmatched_mask_files_count": len(unmatched_masks),
        "unmatched_sample_samples": unmatched_samples[:10],
        "unmatched_mask_samples": unmatched_masks[:10],
        "duplicate_relative_paths_in_zip_count": len(duplicate_relative_paths),
        "duplicate_relative_paths_in_zip": duplicate_relative_paths,
        "duplicate_basenames_within_sample_count": len(duplicate_within_sample),
        "duplicate_basenames_within_sample": duplicate_within_sample,
        "duplicate_basenames_within_mask_count": len(duplicate_within_mask),
        "duplicate_basenames_within_mask": duplicate_within_mask,
        "basenames_appearing_in_both_sample_and_mask_count": len(basenames_in_both_sample_and_mask),
        "basenames_appearing_in_both_sample_and_mask": {
            name: {
                "sample_paths": [
                    p for p in basename_details[name]["relative_paths"]
                    if p.startswith("wseg_dataset/sample/")
                ],
                "mask_paths": [
                    p for p in basename_details[name]["relative_paths"]
                    if p.startswith("wseg_dataset/mask/")
                ],
            }
            for name in basenames_in_both_sample_and_mask
        },
        "basenames_in_multiple_distinct_directories_count": len(basenames_in_multiple_distinct_dirs),
        "basenames_in_multiple_distinct_directories_sample": dict(list(basenames_in_multiple_distinct_dirs.items())[:10]),
        "sample_size_comparison": size_comparisons,
        "verification_caveat": (
            "Relative-path matching between sample/ and mask/ indicates structural naming alignment only. "
            "It does not certify that mask files contain valid, non-corrupt binary segmentation annotations."
        ),
    }


def audit_mendeley_structure(zf: zipfile.ZipFile, infolist: list) -> dict:
    """
    Specific audit for Mendeley:
    - Identifies top-level folders and file distributions.
    - Matches numeric identifiers between 'wound_main' and 'wound_mask'.
    - Preserves original filenames and reports duplicate or unmatched identifiers.
    """
    folder_stats = defaultdict(lambda: {"count": 0, "extensions": Counter(), "samples": []})
    wound_main_id_to_files = defaultdict(list)
    wound_mask_id_to_files = defaultdict(list)

    re_main = re.compile(r"^wound_main-(\d+)", re.IGNORECASE)
    re_mask = re.compile(r"^wound_mask-(\d+)", re.IGNORECASE)

    for info in infolist:
        norm = info.filename.replace("\\", "/")
        if info.is_dir() or norm.endswith("/"):
            continue

        parts = norm.split("/")
        top_folder = parts[0] if len(parts) > 1 else "[root]"
        filename = parts[-1]

        folder_stats[top_folder]["count"] += 1
        p = Path(norm)
        folder_stats[top_folder]["extensions"][p.suffix.lower()] += 1
        if len(folder_stats[top_folder]["samples"]) < 5:
            folder_stats[top_folder]["samples"].append(filename)

        if norm.startswith("wound_main/"):
            m = re_main.match(p.stem)
            identifier = m.group(1) if m else p.stem
            wound_main_id_to_files[identifier].append(filename)
        elif norm.startswith("wound_mask/"):
            m = re_mask.match(p.stem)
            identifier = m.group(1) if m else p.stem
            wound_mask_id_to_files[identifier].append(filename)

    main_ids = set(wound_main_id_to_files.keys())
    mask_ids = set(wound_mask_id_to_files.keys())

    matched_ids = sorted(list(main_ids.intersection(mask_ids)))
    unmatched_main_ids = sorted(list(main_ids - mask_ids))
    unmatched_mask_ids = sorted(list(mask_ids - main_ids))

    duplicate_main_ids = {k: v for k, v in wound_main_id_to_files.items() if len(v) > 1}
    duplicate_mask_ids = {k: v for k, v in wound_mask_id_to_files.items() if len(v) > 1}

    sample_matched_pairs = []
    for num_id in matched_ids[:5]:
        sample_matched_pairs.append({
            "numeric_identifier": num_id,
            "wound_main_filename": wound_main_id_to_files[num_id][0],
            "wound_mask_filename": wound_mask_id_to_files[num_id][0],
        })

    return {
        "top_level_folders": {
            k: {
                "file_count": v["count"],
                "extensions": dict(v["extensions"]),
                "sample_filenames": v["samples"],
            }
            for k, v in folder_stats.items()
        },
        "total_wound_main_files": sum(len(v) for v in wound_main_id_to_files.values()),
        "total_wound_mask_files": sum(len(v) for v in wound_mask_id_to_files.values()),
        "unique_wound_main_identifiers": len(main_ids),
        "unique_wound_mask_identifiers": len(mask_ids),
        "exact_identifier_matches_count": len(matched_ids),
        "duplicate_main_identifiers": duplicate_main_ids,
        "duplicate_mask_identifiers": duplicate_mask_ids,
        "unmatched_main_identifiers_count": len(unmatched_main_ids),
        "unmatched_mask_identifiers_count": len(unmatched_mask_ids),
        "unmatched_main_samples": [wound_main_id_to_files[i] for i in unmatched_main_ids[:10]],
        "unmatched_mask_samples": [wound_mask_id_to_files[i] for i in unmatched_mask_ids[:10]],
        "sample_matched_pairs": sample_matched_pairs,
        "clinical_relevance_note": (
            "The Mendeley archive contains photographic images (wound_main), masks (wound_mask), and normal control limbs (Nomal). "
            "These represent general lower-extremity wounds and healthy controls. Clinical verification is required "
            "to differentiate specific diabetic foot ulcers (DFU) from venous, arterial, pressure, or surgical wounds."
        ),
    }


def audit_complete_hierarchy_and_inventory(zf: zipfile.ZipFile, infolist: list) -> tuple:
    """
    Computes:
    1. Complete directory hierarchy:
       - Records ALL ancestor directories for every file path.
       - Preserves explicit empty directory entries from ZIP metadata.
       - Avoids duplicate directory entries.
       - Summarizes file counts, extensions, and byte sizes for each directory.
    2. Complete file inventory:
       - Records every file's archive-relative path, basename, extension,
         parent directory, compressed size in bytes, and uncompressed size in bytes.
    """
    all_directories = set()
    dir_direct_file_counts = Counter()
    dir_extensions = defaultdict(Counter)
    dir_uncompressed_bytes = Counter()
    dir_compressed_bytes = Counter()

    file_inventory = []

    for info in infolist:
        norm = info.filename.replace("\\", "/")

        if info.is_dir() or norm.endswith("/"):
            # Explicit directory entry in the ZIP central directory
            clean_dir = norm.rstrip("/")
            if clean_dir:
                # Add this directory and all its ancestors
                parts = clean_dir.split("/")
                for i in range(1, len(parts) + 1):
                    all_directories.add("/".join(parts[:i]))
            continue

        p = Path(norm)
        parent_dir = str(p.parent).replace("\\", "/")

        # Record all ancestor directories for this file path
        if parent_dir and parent_dir != ".":
            parts = parent_dir.split("/")
            for i in range(1, len(parts) + 1):
                all_directories.add("/".join(parts[:i]))
        else:
            all_directories.add("[root]")
            parent_dir = "[root]"

        ext = p.suffix.lower() or "[no_extension]"
        base_name = p.name

        dir_direct_file_counts[parent_dir] += 1
        dir_extensions[parent_dir][ext] += 1
        dir_uncompressed_bytes[parent_dir] += info.file_size
        dir_compressed_bytes[parent_dir] += info.compress_size

        # Append to complete file inventory
        file_inventory.append({
            "relative_path": norm,
            "basename": base_name,
            "extension": ext,
            "parent_directory": parent_dir,
            "uncompressed_bytes": info.file_size,
            "compressed_bytes": info.compress_size,
        })

    directory_breakdown = {}
    for d in sorted(all_directories):
        f_count = dir_direct_file_counts[d]
        directory_breakdown[d] = {
            "is_empty_directory": (f_count == 0),
            "direct_file_count": f_count,
            "direct_extensions": dict(dir_extensions[d]),
            "uncompressed_bytes": dir_uncompressed_bytes[d],
            "uncompressed_formatted": format_size(dir_uncompressed_bytes[d]),
            "compressed_bytes": dir_compressed_bytes[d],
            "compressed_formatted": format_size(dir_compressed_bytes[d]),
        }

    hierarchy_data = {
        "directory_breakdown": directory_breakdown,
        "total_directories_recorded": len(directory_breakdown),
        "empty_directories_count": sum(1 for v in directory_breakdown.values() if v["is_empty_directory"]),
    }

    return hierarchy_data, file_inventory


def inspect_archive(config: dict, project_root: Path) -> dict:
    """
    Safely inspect an archive.
    - Checks existence before calling stat() or opening.
    - Handles bad zip files and file system errors cleanly, reporting status.
    """
    dataset_key = config["dataset_key"]
    archive_rel = config["archive_rel"]
    archive_path = (project_root / archive_rel).resolve()

    # Boundary check
    try:
        archive_path.relative_to(project_root)
    except ValueError:
        raise ValueError(f"Security error: path resolves outside project root: {archive_path}")

    report = {
        "dataset_key": dataset_key,
        "description": config["description"],
        "archive_path": str(archive_rel).replace("\\", "/"),
        "exists": False,
        "archive_size_bytes": 0,
        "archive_size_formatted": "N/A",
        "integrity_status": "ARCHIVE_NOT_FOUND",
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "general_hierarchy": {},
        "file_inventory": [],
        "specialized_audit": {},
        "error_message": None,
    }

    if not archive_path.is_file():
        return report

    report["exists"] = True

    try:
        size_bytes = archive_path.stat().st_size
        report["archive_size_bytes"] = size_bytes
        report["archive_size_formatted"] = format_size(size_bytes)
    except Exception as e:
        report["integrity_status"] = f"STAT_ERROR: {type(e).__name__}"
        report["error_message"] = str(e)
        return report

    try:
        with zipfile.ZipFile(archive_path, mode="r") as zf:
            infolist = zf.infolist()
            report["integrity_status"] = "METADATA_READ_SUCCESS"

            # 1. Complete hierarchy & file inventory
            hierarchy_data, inventory = audit_complete_hierarchy_and_inventory(zf, infolist)
            report["general_hierarchy"] = hierarchy_data
            report["file_inventory"] = inventory

            # 2. Specialized audits
            if dataset_key == "AZH":
                report["specialized_audit"] = audit_azh_structure(zf, infolist)
            elif dataset_key == "WSEG":
                report["specialized_audit"] = audit_wseg_structure(zf, infolist)
            elif dataset_key == "MENDELEY":
                report["specialized_audit"] = audit_mendeley_structure(zf, infolist)

    except zipfile.BadZipFile as e:
        report["integrity_status"] = "BAD_ZIP_FILE"
        report["error_message"] = str(e)
    except Exception as e:
        report["integrity_status"] = f"ERROR: {type(e).__name__}"
        report["error_message"] = str(e)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 1B: Detailed Dataset Structure Audit Utility (non-destructive)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_JSON_PATH,
        help=f"Destination path for structural audit JSON report (default: {DEFAULT_OUTPUT_JSON_PATH})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly permit overwriting an existing dataset_structure_audit.json.",
    )
    args = parser.parse_args()

    project_root = PROJECT_ROOT.resolve()

    # Resolve output path and enforce strict project root boundary
    if args.output.is_absolute():
        output_path = args.output.resolve()
    else:
        output_path = (project_root / args.output).resolve()

    try:
        output_path.relative_to(project_root)
    except ValueError:
        print(f"[SECURITY ERROR] Output path '{output_path}' resolves outside the project root: {project_root}", file=sys.stderr)
        return 1

    # Safeguard: do not overwrite existing output unless --force is set
    if output_path.exists() and not args.force:
        print("=" * 80, file=sys.stderr)
        print(f"[SAFEGUARD] Existing audit report detected at: {output_path}", file=sys.stderr)
        print("Operation stopped to protect existing records. Use --force or an alternative --output.", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        return 1

    print("=" * 80)
    print("Stage 1B: Detailed Dataset Structure Audit Utility")
    print(f"Executed at : {datetime.now(timezone.utc).isoformat()} UTC")
    print(f"Project Root: {project_root}")
    print("Mode        : Non-destructive, read-only ZIP metadata audit")
    print("=" * 80)

    dataset_audits = []

    for target in TARGET_ARCHIVES:
        print(f"\nAuditing [{target['dataset_key']}] {target['archive_rel'].name}...")
        audit = inspect_archive(target, project_root)
        dataset_audits.append(audit)

        if not audit["exists"]:
            print(f"  [STATUS] {audit['integrity_status']}: Archive file not found on disk.")
            continue

        if audit["integrity_status"] != "METADATA_READ_SUCCESS":
            print(f"  [ERROR] {audit['integrity_status']}: {audit['error_message']}")
            continue

        gen = audit["general_hierarchy"]
        inv = audit["file_inventory"]
        print(f"  Directories recorded: {gen['total_directories_recorded']} (Empty: {gen['empty_directories_count']})")
        print(f"  File inventory count: {len(inv)} entries cataloged")

        spec = audit["specialized_audit"]
        if target["dataset_key"] == "AZH":
            print(f"  Nested ZIP Archives   : {len(spec['nested_zip_archives'])}")
            for nz in spec["nested_zip_archives"]:
                print(f"    - Path: {nz['path']} (Compressed: {nz['compressed_size_formatted']}, Uncompressed: {nz['uncompressed_size_formatted']})")
            print(f"  Sub-cohort Directories: {list(spec['cohort_directories'].keys())}")
            for label, pinfo in spec["candidate_pairing_analysis"].items():
                print(f"    - {label}: {pinfo['total_image_files']} imgs, {pinfo['total_mask_files']} masks -> {pinfo['exact_stem_matches_count']} matches, {pinfo['unmatched_images_count']} unmatched imgs, {len(pinfo['duplicate_image_stems'])} dup stems")

        elif target["dataset_key"] == "WSEG":
            print(f"  Sample files: {spec['total_sample_files']} | Mask files: {spec['total_mask_files']}")
            print(f"  Exact relative-path matches: {spec['exact_relative_path_matches_count']} | Unmatched samples: {spec['unmatched_sample_files_count']} | Unmatched masks: {spec['unmatched_mask_files_count']}")
            print(f"  Duplicate relative paths in ZIP: {spec['duplicate_relative_paths_in_zip_count']}")
            print(f"  Duplicate basenames within sample/: {spec['duplicate_basenames_within_sample_count']}")
            print(f"  Duplicate basenames within mask/  : {spec['duplicate_basenames_within_mask_count']}")
            print(f"  Basenames appearing in both trees : {spec['basenames_appearing_in_both_sample_and_mask_count']}")
            print(f"  Basenames in multiple distinct dirs: {spec['basenames_in_multiple_distinct_directories_count']}")

        elif target["dataset_key"] == "MENDELEY":
            print("  Top-level folders:")
            for folder, fstats in spec["top_level_folders"].items():
                print(f"    - {folder}/: {fstats['file_count']} files ({fstats['extensions']})")
            print("  Pairing Pattern Analysis (wound_main vs wound_mask):")
            print(f"    - wound_main: {spec['total_wound_main_files']} | wound_mask: {spec['total_wound_mask_files']}")
            print(f"    - Exact numeric identifier matches: {spec['exact_identifier_matches_count']}")
            print(f"    - Unmatched main IDs: {spec['unmatched_main_identifiers_count']} | Unmatched mask IDs: {spec['unmatched_mask_identifiers_count']}")
            print(f"    - Duplicate IDs: Main={len(spec['duplicate_main_identifiers'])}, Mask={len(spec['duplicate_mask_identifiers'])}")

    # Build master manifest document
    master_manifest = {
        "metadata": {
            "title": "Stage 1B: Detailed Dataset Structure Audit Report",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": "scripts/audit_dataset_structure.py",
            "project_root": str(project_root).replace("\\", "/"),
            "audit_mode": "read_only_zip_metadata_and_inventory",
            "scientific_disclaimer": (
                "Candidate filename/stem pairings reflect structural filename alignment only. "
                "Matching filenames do not prove valid segmentation masks or accurate ground truth. "
                "Clinical disease etiology (e.g. DFU vs general wounds vs normal controls in Mendeley) "
                "remains unverified pending expert clinical annotation."
            ),
        },
        "datasets": dataset_audits,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(master_manifest, f, indent=2)

    rel_out_str = str(output_path.relative_to(project_root)).replace("\\", "/")
    print("\n" + "=" * 80)
    print(f"Structural Audit Manifest successfully written to: {rel_out_str}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
