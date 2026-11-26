"""
Safe Dataset ZIP Archive Inspection Utility
===========================================

This utility performs non-destructive, read-only inspection of dataset ZIP archives
located in:
  - data/raw/AZH/archives/
  - data/raw/WSEG/archives/
  - data/raw/MENDELEY/archives/

Adheres strictly to research verification guidelines:
  1. Inspects headers and tables of contents without extracting, moving, modifying,
     renaming, or deleting any files.
  2. Uses only standard-library Python modules (zipfile, os, sys, json, pathlib,
     collections, datetime, argparse, re).
  3. All paths are strictly anchored to the project root determined relative to
     this script file; any path resolving outside the project root is rejected.
  4. Audits path safety (path traversal '..', absolute paths, encryption flags).
  5. Identifies candidate image/mask directories using tokenized boundary matching,
     while explicitly avoiding premature claims of verified clinical image counts or
     paired masks.
  6. Performs metadata and central-directory inspection only without full CRC scans.
  7. Protects existing inspection manifests: halts if the target output file exists,
     unless an explicit approved alternative path or overwrite flag is specified.
  8. Outputs a structured summary to stdout and writes a machine-readable JSON
     audit report to data/manifests/archive_inspection.json (or specified path).
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import zipfile


# Determine project root from the absolute location of this script
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent

# Target datasets, designated storage subdirectories, and expected filenames
DATASETS = [
    {
        "dataset_key": "AZH",
        "archive_dir_rel": Path("data/raw/AZH/archives"),
        "expected_filename": "wound-segmentation-master.zip",
        "description": "AZH repository wound segmentation archive",
    },
    {
        "dataset_key": "WSEG",
        "archive_dir_rel": Path("data/raw/WSEG/archives"),
        "expected_filename": "wseg_dataset.zip",
        "description": "WSEG dataset archive",
    },
    {
        "dataset_key": "MENDELEY",
        "archive_dir_rel": Path("data/raw/MENDELEY/archives"),
        "expected_filename": "Wound Image Dataset.zip",
        "description": "Mendeley lower-limb wound collection archive",
    },
]

DEFAULT_OUTPUT_JSON_PATH = PROJECT_ROOT / "data" / "manifests" / "archive_inspection.json"

# Candidate keyword patterns for heuristic categorization (non-definitive)
# Substring keywords (safe for directory parts or stems)
MASK_SUBSTRING_KEYWORDS = ("mask", "label", "annotation", "groundtruth", "seg")
IMAGE_SUBSTRING_KEYWORDS = ("image", "img", "photo", "original", "input")

# Regex to match tokenized keywords or exact short abbreviations like "gt"
# Ensures "gt" is matched only as an isolated word or delimited by [_.-] (e.g., "sample_gt.png", "gt/001.png")
# rather than as an arbitrary substring in words like "length", "weighted", etc.
MASK_TOKEN_REGEX = re.compile(r"(?:^|[\W_])(?:mask|label|annotation|groundtruth|gt|seg)(?:$|[\W_])", re.IGNORECASE)
IMAGE_TOKEN_REGEX = re.compile(r"(?:^|[\W_])(?:image|img|photo|original|input)(?:$|[\W_])", re.IGNORECASE)


def format_size(size_bytes: int) -> str:
    """Format bytes into human-readable MB and GB strings."""
    mb = size_bytes / (1024 * 1024)
    gb = size_bytes / (1024 * 1024 * 1024)
    if gb >= 1.0:
        return f"{mb:.2f} MB ({gb:.3f} GB)"
    return f"{mb:.2f} MB"


def is_candidate_mask(p: Path) -> bool:
    """
    Check if a path component or filename represents a candidate mask.
    Matches distinct tokens or specific directory substrings, avoiding arbitrary substring false positives.
    """
    stem_lower = p.stem.lower()
    if MASK_TOKEN_REGEX.search(stem_lower):
        return True
    for part in p.parent.parts:
        part_lower = part.lower()
        if any(kw in part_lower for kw in MASK_SUBSTRING_KEYWORDS) or MASK_TOKEN_REGEX.search(part_lower):
            return True
    return False


def is_candidate_image(p: Path) -> bool:
    """
    Check if a path component or filename represents a candidate original image.
    Matches distinct tokens or specific directory substrings.
    """
    stem_lower = p.stem.lower()
    if IMAGE_TOKEN_REGEX.search(stem_lower):
        return True
    for part in p.parent.parts:
        part_lower = part.lower()
        if any(kw in part_lower for kw in IMAGE_SUBSTRING_KEYWORDS) or IMAGE_TOKEN_REGEX.search(part_lower):
            return True
    return False


def inspect_single_archive(target_config: dict, project_root: Path) -> dict:
    """
    Safely inspect a single archive entry without extraction or full CRC decompression.
    Reads central directory metadata only. All paths are anchored to project_root.
    """
    dataset_key = target_config["dataset_key"]
    archive_dir_rel = target_config["archive_dir_rel"]
    expected_filename = target_config["expected_filename"]
    
    archive_dir = (project_root / archive_dir_rel).resolve()
    archive_path = (archive_dir / expected_filename).resolve()

    # Enforce project root boundary check
    if not str(archive_path).startswith(str(project_root)):
        raise ValueError(f"Security error: archive path '{archive_path}' resolves outside project root.")

    rel_archive_dir_str = str(archive_dir_rel).replace("\\", "/")
    rel_archive_path_str = f"{rel_archive_dir_str}/{expected_filename}"

    report = {
        "dataset_key": dataset_key,
        "description": target_config["description"],
        "archive_directory": rel_archive_dir_str,
        "expected_filename": expected_filename,
        "archive_path": rel_archive_path_str,
        "exists": False,
        "file_size_bytes": 0,
        "file_size_formatted": "N/A",
        "integrity_status": "NOT_FOUND",
        "crc_verification_performed": False,
        "total_archive_entries": 0,
        "total_files": 0,
        "total_directories": 0,
        "extension_counts": {},
        "first_30_file_paths": [],
        "candidate_mask_directories": [],
        "candidate_image_directories": [],
        "duplicate_filenames_within_archive": {},
        "security_flags": {
            "path_traversal_entries": [],
            "absolute_path_entries": [],
            "encrypted_entries": [],
        },
        "verification_notes": (
            "Metadata-only inspection of archive table of contents. "
            "Full CRC verification has not been performed at this initial stage. "
            "Candidate directories are heuristic indications only; no clinical "
            "image counts, patient counts, or paired ground-truth masks are verified."
        ),
    }

    if not archive_path.is_file():
        existing_files = list(archive_dir.glob("*.zip")) if archive_dir.is_dir() else []
        if existing_files:
            report["integrity_status"] = (
                f"EXPECTED_FILE_MISSING (found other zip files: {[f.name for f in existing_files]})"
            )
        else:
            report["integrity_status"] = "ARCHIVE_FILE_NOT_FOUND"
        return report

    report["exists"] = True
    size_bytes = archive_path.stat().st_size
    report["file_size_bytes"] = size_bytes
    report["file_size_formatted"] = format_size(size_bytes)

    # Perform non-destructive read-only inspection of metadata only
    try:
        with zipfile.ZipFile(archive_path, mode="r") as zf:
            # Metadata read successful; explicitly do not run full CRC scan (testzip)
            report["integrity_status"] = "METADATA_READ_SUCCESS_NOT_FULLY_CRC_VERIFIED"
            report["crc_verification_performed"] = False

            infolist = zf.infolist()
            report["total_archive_entries"] = len(infolist)

            ext_counter = Counter()
            filename_counter = Counter()
            sample_paths = []
            cand_mask_dirs = set()
            cand_img_dirs = set()

            files_count = 0
            dirs_count = 0

            for info in infolist:
                raw_filename = info.filename
                normalized_path = raw_filename.replace("\\", "/")

                # Security checks
                if ".." in normalized_path.split("/"):
                    report["security_flags"]["path_traversal_entries"].append(raw_filename)
                if normalized_path.startswith("/") or (len(raw_filename) > 1 and raw_filename[1] == ":"):
                    report["security_flags"]["absolute_path_entries"].append(raw_filename)
                if info.flag_bits & 0x1:
                    report["security_flags"]["encrypted_entries"].append(raw_filename)

                # Differentiate directories from files
                is_dir = info.is_dir() or raw_filename.endswith("/")
                if is_dir:
                    dirs_count += 1
                    continue

                files_count += 1

                # Sample up to first 30 file paths
                if len(sample_paths) < 30:
                    sample_paths.append(normalized_path)

                # Extension analysis
                p = Path(normalized_path)
                ext = p.suffix.lower()
                ext_key = ext if ext else "[no_extension]"
                ext_counter[ext_key] += 1

                # Track basename for duplicate filename analysis
                base_name = p.name
                filename_counter[base_name] += 1

                # Candidate categorization heuristics using token/boundary matching
                if is_candidate_mask(p):
                    cand_mask_dirs.add(str(p.parent).replace("\\", "/"))
                elif is_candidate_image(p):
                    cand_img_dirs.add(str(p.parent).replace("\\", "/"))

            report["total_files"] = files_count
            report["total_directories"] = dirs_count
            report["extension_counts"] = dict(ext_counter.most_common())
            report["first_30_file_paths"] = sample_paths
            report["candidate_mask_directories"] = sorted(list(cand_mask_dirs))[:20]
            report["candidate_image_directories"] = sorted(list(cand_img_dirs))[:20]

            # Duplicate basenames reporting
            duplicates = {name: count for name, count in filename_counter.items() if count > 1}
            report["duplicate_filenames_within_archive"] = {
                "total_duplicate_basenames": len(duplicates),
                "samples": dict(list(duplicates.items())[:15]),
            }

    except zipfile.BadZipFile as e:
        report["integrity_status"] = f"BAD_ZIP_FILE: {str(e)}"
    except Exception as e:
        report["integrity_status"] = f"ERROR: {type(e).__name__}: {str(e)}"

    return report


def main() -> int:
    """Run inspection across all defined archives, display report, and safely save JSON."""
    parser = argparse.ArgumentParser(
        description="Inspect dataset ZIP archives metadata without full CRC scan or extraction."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_JSON_PATH,
        help="Destination path for inspection JSON report (must resolve inside project root).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly permit overwriting the existing output JSON report.",
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
        print("=" * 80, file=sys.stderr)
        print(f"[SECURITY ERROR] Output path '{output_path}' resolves outside the project root: {project_root}", file=sys.stderr)
        print("Operation rejected to maintain project isolation.", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        return 1

    # Protection: check whether destination report already exists
    if output_path.exists() and not args.force:
        print("=" * 80, file=sys.stderr)
        print(f"[SAFEGUARD] Existing inspection report detected at: {output_path}", file=sys.stderr)
        print("To protect verified manifests from unintentional overwrite, the operation has been stopped.", file=sys.stderr)
        print("Provide an alternative output path using --output <new_path.json> or pass --force if overwrite is explicitly approved.", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        return 1

    print("=" * 80)
    print("DFU Cross-Domain Research: Dataset ZIP Archive Inspection Utility")
    print(f"Executed at : {datetime.now(timezone.utc).isoformat()} UTC")
    print(f"Project Root: {project_root}")
    print("Mode        : Read-only, metadata & central-directory inspection only")
    print("Integrity   : Full CRC verification NOT performed at this initial stage")
    print("=" * 80)

    summary_results = []

    for target in DATASETS:
        print(f"\n[{target['dataset_key']}] Inspecting {target['expected_filename']}...")
        result = inspect_single_archive(target, project_root)
        summary_results.append(result)

        print(f"  Storage Path      : {result['archive_path']}")
        print(f"  Exists            : {result['exists']}")
        print(f"  Size              : {result['file_size_formatted']}")
        print(f"  Integrity Status  : {result['integrity_status']}")
        print(f"  CRC Scan Verified : {result['crc_verification_performed']} (full CRC check omitted for speed and safety)")

        if not result["exists"]:
            print("  Note              : Archive file not found at expected path.")
            continue

        print(f"  Entries Total     : {result['total_archive_entries']} ({result['total_files']} files, {result['total_directories']} directories)")
        print(f"  Extensions        : {result['extension_counts']}")

        # Security checks
        sec = result["security_flags"]
        has_security_issue = any(len(v) > 0 for v in sec.values())
        if has_security_issue:
            print(f"  [WARNING] Security Flags Detected: {sec}")
        else:
            print("  Security Checks   : Clean (no path traversal, absolute paths, or encryption detected)")

        # Duplicate basenames
        dup_info = result["duplicate_filenames_within_archive"]
        print(f"  Duplicate Basenames: {dup_info['total_duplicate_basenames']} unique names appear multiple times")
        if dup_info["total_duplicate_basenames"] > 0:
            print(f"    Sample Duplicates : {dup_info['samples']}")

        # Candidate directories
        print(f"  Candidate Image Dirs : {result['candidate_image_directories'][:5]} (candidate only, unverified)")
        print(f"  Candidate Mask Dirs  : {result['candidate_mask_directories'][:5]} (candidate only, unverified)")

        # First file samples
        print("  First 5 Entries:")
        for sample in result["first_30_file_paths"][:5]:
            print(f"    - {sample}")
        if len(result["first_30_file_paths"]) > 5:
            print(f"    ... (and {len(result['first_30_file_paths']) - 5} more in JSON manifest)")

    # Construct complete JSON document
    manifest_doc = {
        "metadata": {
            "title": "DFU Cross-Domain Research Archive Inspection Report",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": "scripts/inspect_archives.py",
            "audit_mode": "metadata_only_central_directory_inspection",
            "crc_verification_performed": False,
            "scientific_disclaimer": (
                "This report records raw ZIP archive contents and structural metadata only. "
                "Full CRC-32 verification has not been performed at this initial stage. "
                "No clinical image counts, patient counts, or paired ground-truth masks are certified. "
                "Candidate directory labels are heuristic keywords and require downstream verification."
            ),
        },
        "datasets": summary_results,
    }

    # Save safely to the designated destination
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest_doc, f, indent=2)

    rel_out_str = str(output_path.relative_to(project_root)).replace("\\", "/")
    print("\n" + "=" * 80)
    print(f"JSON Inspection Manifest successfully written to: {rel_out_str}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
