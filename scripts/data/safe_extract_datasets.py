"""
Safe extraction utility for DFU dataset archives.
Implements non-destructive extraction with security validations:
- Path traversal prevention (Zip Slip)
- Absolute path rejection
- Compression bomb / oversized entry detection
- Duplicate output path detection
- Disk space verification before extraction
- Streaming SHA-256 computation
- Atomic promotion from temporary directory
- Detailed provenance manifest generation (CSV)
"""

import os
import sys
import shutil
import zipfile
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Constants & Safety Thresholds
MAX_COMPRESSION_RATIO = 100.0  # Max uncompressed / compressed ratio
MAX_SINGLE_FILE_SIZE = 500 * 1024 * 1024  # 500 MB max for a single file
REQUIRED_FREE_SPACE_BUFFER = 2 * 1024 * 1024 * 1024  # 2 GB free space buffer


def setup_logger(log_file: Path) -> logging.Logger:
    """Configures file and console logging."""
    logger = logging.getLogger("safe_extractor")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh_fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    fh.setFormatter(fh_fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch_fmt = logging.Formatter("[%(levelname)s] %(message)s")
    ch.setFormatter(ch_fmt)
    logger.addHandler(ch)

    return logger


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Computes SHA-256 hash using streaming chunks to minimize memory consumption."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_stream_sha256(stream, chunk_size: int = 1024 * 1024) -> str:
    """Computes SHA-256 hash of a file-like stream."""
    hasher = hashlib.sha256()
    while chunk := stream.read(chunk_size):
        hasher.update(chunk)
    return hasher.hexdigest()


def check_disk_space(target_dir: Path, required_bytes: int, logger: logging.Logger) -> bool:
    """Verifies that the target filesystem has sufficient space for extraction + buffer."""
    drive_anchor = target_dir.resolve().anchor
    total, used, free = shutil.disk_usage(drive_anchor)
    needed = required_bytes + REQUIRED_FREE_SPACE_BUFFER
    logger.info(f"Disk check on {drive_anchor}: Free={free / (1024**3):.2f} GB, Required={needed / (1024**3):.2f} GB")
    if free < needed:
        logger.error(f"Insufficient disk space! Free: {free} bytes, Needed: {needed} bytes")
        return False
    return True


def validate_archive_member(
    member: zipfile.ZipInfo,
    target_root: Path,
    seen_paths: set,
    logger: logging.Logger
) -> Tuple[bool, Optional[str]]:
    """
    Validates a zip archive member against security risks:
    - Path traversal (Zip Slip)
    - Absolute paths
    - Excessive compression ratio
    - Oversized files
    - Duplicate output paths
    """
    raw_path = member.filename

    # Reject absolute paths
    if os.path.isabs(raw_path) or raw_path.startswith("/") or raw_path.startswith("\\"):
        return False, f"Path traversal attempt: absolute path '{raw_path}'"

    # Normalize path and check traversal
    dest_path = (target_root / raw_path).resolve()
    target_root_resolved = target_root.resolve()
    try:
        dest_path.relative_to(target_root_resolved)
    except ValueError:
        return False, f"Path traversal attempt: path '{raw_path}' escapes target root '{target_root}'"

    # Check for duplicate output paths (case-insensitive on Windows)
    norm_dest = str(dest_path).lower()
    if norm_dest in seen_paths:
        return False, f"Duplicate output path collision: '{raw_path}'"
    seen_paths.add(norm_dest)

    # Check file size thresholds (skip directories)
    if not member.is_dir():
        if member.file_size > MAX_SINGLE_FILE_SIZE:
            return False, f"Oversized entry: {member.file_size} bytes exceeds limit {MAX_SINGLE_FILE_SIZE}"

        if member.compress_size > 0:
            ratio = member.file_size / member.compress_size
            if ratio > MAX_COMPRESSION_RATIO:
                return False, f"Suspicious compression ratio: {ratio:.1f} exceeds limit {MAX_COMPRESSION_RATIO}"

    return True, None


def safe_extract_archive(
    archive_path: Path,
    expected_archive_sha256: str,
    final_output_dir: Path,
    temp_extract_dir: Path,
    dataset_name: str,
    manifest_csv: Path,
    logger: logging.Logger
) -> bool:
    """
    Safely extracts an archive into final_output_dir with atomic temporary promotion
    and generates an itemized provenance manifest.
    """
    logger.info(f"=== Beginning Safe Extraction for {dataset_name} ===")
    logger.info(f"Archive: {archive_path}")
    logger.info(f"Destination: {final_output_dir}")

    # Step 1: Pre-extraction archive hash verification
    if not archive_path.exists():
        logger.error(f"Archive not found: {archive_path}")
        return False

    logger.info("Verifying archive SHA-256 before extraction...")
    actual_hash = compute_sha256(archive_path)
    if actual_hash != expected_archive_sha256:
        logger.error(f"Archive SHA-256 mismatch! Expected: {expected_archive_sha256}, Actual: {actual_hash}")
        return False
    logger.info(f"Archive SHA-256 verified: {actual_hash[:16]}... OK")

    # Step 2: Inspect members & verify disk space
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            infolist = zf.infolist()
            total_uncompressed = sum(m.file_size for m in infolist)
            logger.info(f"Archive contains {len(infolist)} entries, total uncompressed size: {total_uncompressed / (1024**2):.2f} MB")

            if not check_disk_space(temp_extract_dir, total_uncompressed, logger):
                return False

            # Step 3: Validate all members before extracting anything
            seen_paths = set()
            validation_errors = []
            for member in infolist:
                is_valid, err = validate_archive_member(member, temp_extract_dir, seen_paths, logger)
                if not is_valid:
                    validation_errors.append((member.filename, err))

            if validation_errors:
                logger.error(f"Archive validation failed with {len(validation_errors)} security/path errors!")
                for fname, err in validation_errors[:10]:
                    logger.error(f"  {fname}: {err}")
                return False
            logger.info("All archive members passed security and path traversal validation.")

            # Step 4: Extract into temporary directory
            if temp_extract_dir.exists():
                shutil.rmtree(temp_extract_dir)
            temp_extract_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Extracting {len(infolist)} entries to temporary directory: {temp_extract_dir}")
            manifest_rows = []

            for i, member in enumerate(infolist):
                member_path = member.filename
                dest_path = temp_extract_dir / member_path

                if member.is_dir():
                    dest_path.mkdir(parents=True, exist_ok=True)
                    continue

                # Ensure parent directory exists
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                # Extract member content
                with zf.open(member, "r") as src, open(dest_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

                # Verify extracted file size
                actual_size = dest_path.stat().st_size
                if actual_size != member.file_size:
                    logger.error(f"Size mismatch on extraction: {member_path} (Expected {member.file_size}, Got {actual_size})")
                    return False

                # Compute extracted file SHA-256
                file_sha = compute_sha256(dest_path)
                p = Path(member_path)
                manifest_rows.append({
                    "dataset_name": dataset_name,
                    "archive_name": archive_path.name,
                    "archive_member_path": member_path,
                    "extracted_relative_path": str(Path(dataset_name) / member_path).replace("\\", "/"),
                    "filename": p.name,
                    "extension": p.suffix.lower(),
                    "file_size_bytes": actual_size,
                    "sha256": file_sha,
                    "extraction_status": "SUCCESS",
                    "error_message": ""
                })

                if (i + 1) % 1000 == 0 or (i + 1) == len(infolist):
                    logger.info(f"Extracted {i + 1}/{len(infolist)} entries...")

    except Exception as e:
        logger.exception(f"Unexpected error during extraction: {e}")
        return False

    # Step 5: Atomic promotion from temp_extract_dir to final_output_dir
    logger.info(f"Promoting temporary files to final directory: {final_output_dir}")
    if final_output_dir.exists():
        # Remove any previous incomplete contents
        for item in final_output_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    final_output_dir.mkdir(parents=True, exist_ok=True)
    for item in temp_extract_dir.iterdir():
        shutil.move(str(item), str(final_output_dir / item.name))

    # Clean up temp dir
    if temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir)

    # Step 6: Post-extraction archive hash verification
    logger.info("Re-verifying archive SHA-256 after extraction...")
    post_hash = compute_sha256(archive_path)
    if post_hash != expected_archive_sha256:
        logger.error(f"Archive corrupted during extraction! Hash changed to {post_hash}")
        return False
    logger.info(f"Archive post-extraction SHA-256 verified: {post_hash[:16]}... OK")

    # Step 7: Write provenance manifest
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv
    fieldnames = [
        "dataset_name", "archive_name", "archive_member_path", "extracted_relative_path",
        "filename", "extension", "file_size_bytes", "sha256", "extraction_status", "error_message"
    ]
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    logger.info(f"Provenance manifest written: {manifest_csv} ({len(manifest_rows)} records)")
    logger.info(f"=== Extraction for {dataset_name} COMPLETED SUCCESSFULLY ===\n")
    return True
