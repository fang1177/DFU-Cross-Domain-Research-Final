"""
Stage 4A boundary-field generation and persistence (v1).

Generates and persists the approved two-sided tolerance-bounded distance fields
(``phi_fp`` and ``phi_fn``) for eligible Stage 3.2B canonical masks.

Authoritative source: ``stage_3_2B_manifest_repair_v2/canonical_manifest_corrected_v2.csv``.
The approved field mathematics lives in ``src.data.transforms`` and is reused
verbatim; this module only orchestrates eligibility, provenance, persistence,
and deterministic/idempotent validation.

Design guarantees
-----------------
* Deterministic: no randomness, no timestamps in the field manifest.
* Self-describing content hash: dtype, shape, memory order and endianness are
  hashed together with the raw float32 payload.
* Idempotent: a rerun recomputes the expected fields in memory, validates the
  existing artifacts, and rewrites nothing when they match.
* Conflict-safe: existing artifacts that differ from the expected content are
  reported as conflicts and are never overwritten unless ``regenerate=True``.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

from src.data.transforms import compute_tolerance_boundary_fields

# ---------------------------------------------------------------------------
# Approved constants
# ---------------------------------------------------------------------------
FIELD_VERSION = "stage4a_boundary_fields_v1"
FIELD_ALGORITHM = "compute_tolerance_boundary_fields"
TAU = 2.0
MAX_CAP = 20.0
OUTPUT_DTYPE = "float32"
HASH_CONVENTION = "dfu-canonical-field-sha256-v1"

ELIGIBLE_MASK_STATUSES = frozenset(
    {
        "strict_binary",
        "canonical_binarized_jpeg",
        "authentic_negative_empty",
        "authentic_negative_healthy",
    }
)
EXCLUDED_COHORTS = frozenset({"mismatched_target"})
SEGMENTATION_TRAINING_COHORTS = frozenset(
    {"source_training", "internal_validation", "native_target"}
)

DEFAULT_MANIFEST = (
    "data/processed/stage_3_2B_manifest_repair_v2/"
    "canonical_manifest_corrected_v2.csv"
)
DEFAULT_MASKS_DIR = "data/processed/stage_3_2B_canonical/masks"
DEFAULT_OUTPUT_DIR = "data/processed/stage_4A_boundary_fields_v1"

_RECORD_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

MANIFEST_COLUMNS = [
    "record_id",
    "pair_id",
    "cohort",
    "annotation_role",
    "cluster_id",
    "unique_content_id",
    "unit_role",
    "is_independent_training_unit",
    "source_mask_path",
    "source_mask_sha256_manifest",
    "source_mask_sha256_actual",
    "source_mask_hash_verified",
    "field_algorithm",
    "field_version",
    "hash_convention",
    "tau",
    "max_cap",
    "output_dtype",
    "output_shape",
    "phi_fp_filename",
    "phi_fn_filename",
    "phi_fp_content_hash",
    "phi_fn_content_hash",
    "phi_fp_file_sha256",
    "phi_fn_file_sha256",
    "is_empty_mask",
    "foreground_pixel_count",
    "interior_field_all_zero",
    "is_diagnostic",
    "generation_status",
]


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------
def canonical_field_header(arr: np.ndarray) -> bytes:
    """Return the self-describing header for the canonical field hash."""
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(f"canonical_field_header expects a 2-D array, got {a.shape}")
    return (
        f"DFU-BOUNDARY-FIELD|hash={HASH_CONVENTION}|dtype=float32|"
        f"endian=little|shape={a.shape[0]}x{a.shape[1]}|order=C|"
    ).encode("ascii")


def canonical_field_hash(arr: np.ndarray) -> str:
    """
    Compute the self-describing canonical SHA-256 of a float32 (H, W) field.

    The hashed byte stream is::

        b"DFU-BOUNDARY-FIELD|hash=dfu-canonical-field-sha256-v1|"
        b"dtype=float32|endian=little|shape=<H>x<W>|order=C|"
        + np.ascontiguousarray(arr, dtype="<f4").tobytes(order="C")

    This avoids the ambiguity of a raw ``arr.tobytes()`` digest: dtype, shape,
    memory order, and endianness are part of the hashed material.
    """
    a = np.asarray(arr)
    if a.dtype != np.dtype(np.float32):
        raise ValueError(f"canonical_field_hash expects float32, got {a.dtype}")
    contiguous = np.ascontiguousarray(a, dtype="<f4")
    digest = hashlib.sha256()
    digest.update(canonical_field_header(contiguous))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def load_field_array(path: Path) -> np.ndarray:
    """Load a persisted field and validate its dtype, rank, and C-contiguity."""
    arr = np.load(path, allow_pickle=False)
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"{path} does not contain a single ndarray")
    if arr.dtype != np.dtype(np.float32):
        raise ValueError(f"{path} has dtype {arr.dtype}, expected float32")
    if arr.ndim != 2:
        raise ValueError(f"{path} has shape {arr.shape}, expected (H, W)")
    return np.ascontiguousarray(arr, dtype="<f4")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.save(handle, arr, allow_pickle=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Manifest eligibility
# ---------------------------------------------------------------------------
def load_manifest_rows(manifest_path: Path) -> List[Dict[str, str]]:
    with open(manifest_path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def is_eligible_record(row: Dict[str, str]) -> bool:
    """Eligibility derived from the corrected v2 manifest fields."""
    if row.get("cohort", "") in EXCLUDED_COHORTS:
        return False
    if row.get("is_diagnostic", "").strip().lower() == "true":
        return False
    if row.get("mask_status", "") not in ELIGIBLE_MASK_STATUSES:
        return False
    if not row.get("proc_mask_filename", "").strip():
        return False
    return True


def record_id_for(row: Dict[str, str]) -> str:
    return f"{row['cohort']}_{row['pair_id']}"


def unit_role_for(row: Dict[str, str]) -> str:
    if row.get("cohort") == "healthy_controls":
        return "healthy_control_negative"
    if row.get("is_primary_reference", "").strip().lower() == "true":
        return "primary_unique_image"
    return "alternate_annotation"


def is_independent_training_unit(row: Dict[str, str]) -> bool:
    """Alternates and healthy controls are never independent training units."""
    return (
        row.get("cohort") in SEGMENTATION_TRAINING_COHORTS
        and row.get("is_primary_reference", "").strip().lower() == "true"
        and row.get("is_diagnostic", "").strip().lower() != "true"
    )


def eligible_records(
    manifest_path: Path,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Return ``(eligible_rows, excluded_diagnostic_rows)`` from the manifest.

    Raises ``ValueError`` if two eligible rows would map to the same record id.
    """
    all_rows = load_manifest_rows(manifest_path)
    eligible: List[Dict[str, str]] = []
    excluded: List[Dict[str, str]] = []
    seen: Dict[str, Dict[str, str]] = {}

    for row in all_rows:
        if not is_eligible_record(row):
            if row.get("is_diagnostic", "").strip().lower() == "true":
                excluded.append(row)
            continue
        rid = record_id_for(row)
        if not _RECORD_ID_RE.match(rid):
            raise ValueError(f"Unsafe record id derived from manifest: {rid!r}")
        if rid in seen:
            raise ValueError(
                "Record-id collision for eligible rows: "
                f"{rid!r} ({seen[rid].get('cohort')}/{seen[rid].get('pair_id')}) "
                f"and ({row.get('cohort')}/{row.get('pair_id')})"
            )
        seen[rid] = row
        eligible.append(row)
    return eligible, excluded


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _read_manifest_index(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return {row["record_id"]: row for row in csv.DictReader(handle)}


def _mask_to_binary(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        arr = np.array(image.convert("L"))
    if arr.ndim != 2:
        raise ValueError(f"Mask {path} did not decode to a 2-D array")
    return arr > 0


def _new_manifest_row(
    row: Dict[str, str],
    rec_id: str,
    source_mask_path: Path,
    mask_sha_manifest: str,
    mask_sha_actual: str,
    mask_hash_ok: bool,
    fp_name: str,
    fn_name: str,
    fp_hash: str,
    fn_hash: str,
    fp_file_sha: str,
    fn_file_sha: str,
    empty_mask: bool,
    fg_count: int,
    interior_zero: bool,
    status: str,
    shape: Optional[Tuple[int, int]],
) -> Dict[str, Any]:
    return {
        "record_id": rec_id,
        "pair_id": row["pair_id"],
        "cohort": row["cohort"],
        "annotation_role": (
            "primary"
            if row.get("is_primary_reference", "").strip().lower() == "true"
            else "alternate"
        ),
        "cluster_id": row.get("cluster_id", ""),
        "unique_content_id": row.get("unique_content_id", ""),
        "unit_role": unit_role_for(row),
        "is_independent_training_unit": str(is_independent_training_unit(row)),
        "source_mask_path": source_mask_path.as_posix(),
        "source_mask_sha256_manifest": mask_sha_manifest,
        "source_mask_sha256_actual": mask_sha_actual,
        "source_mask_hash_verified": str(mask_hash_ok),
        "field_algorithm": FIELD_ALGORITHM,
        "field_version": FIELD_VERSION,
        "hash_convention": HASH_CONVENTION,
        "tau": str(TAU),
        "max_cap": str(MAX_CAP),
        "output_dtype": OUTPUT_DTYPE,
        "output_shape": f"{shape[0]}x{shape[1]}" if shape else "",
        "phi_fp_filename": fp_name,
        "phi_fn_filename": fn_name,
        "phi_fp_content_hash": fp_hash,
        "phi_fn_content_hash": fn_hash,
        "phi_fp_file_sha256": fp_file_sha,
        "phi_fn_file_sha256": fn_file_sha,
        "is_empty_mask": str(empty_mask),
        "foreground_pixel_count": str(fg_count),
        "interior_field_all_zero": str(interior_zero),
        "is_diagnostic": "False",
        "generation_status": status,
    }


def _write_manifest(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda item: item["record_id"])
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for item in ordered:
            writer.writerow(item)
    os.replace(tmp, path)


def _write_summary(path: Path, summary: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Stage 4A Boundary-Field Generation Summary (v1)")
    lines.append("")
    lines.append(f"- Field version: `{FIELD_VERSION}`")
    lines.append(f"- Algorithm: `{FIELD_ALGORITHM}`")
    lines.append(f"- Hash convention: `{HASH_CONVENTION}`")
    lines.append(f"- tau: `{TAU}`")
    lines.append(f"- max_cap: `{MAX_CAP}`")
    lines.append(f"- Output dtype: `{OUTPUT_DTYPE}`")
    lines.append(f"- Generated (UTC): `{summary['generated_at_utc']}`")
    lines.append(f"- Regenerate mode: `{summary['regenerate']}`")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append(f"- Manifest rows read: {summary['manifest_rows']}")
    lines.append(f"- Eligible field records: {summary['eligible_records']}")
    lines.append(f"- Manifest rows written: {summary['manifest_records_written']}")
    valid_total = (
        summary["status_counts"].get("generated", 0)
        + summary["status_counts"].get("skipped_valid", 0)
        + summary["status_counts"].get("regenerated", 0)
    )
    lines.append(f"- Artifacts valid at end of run: {valid_total}")
    lines.append(f"- Generated this run: {summary['status_counts'].get('generated', 0)}")
    lines.append(
        f"- Validated and skipped: {summary['status_counts'].get('skipped_valid', 0)}"
    )
    lines.append(
        f"- Explicitly regenerated: {summary['status_counts'].get('regenerated', 0)}"
    )
    lines.append(f"- Conflicts: {summary['status_counts'].get('conflict', 0)}")
    lines.append(f"- Failed records: {summary['failed_records']}")
    lines.append(f"- Empty masks: {summary['empty_mask_records']}")
    lines.append(
        f"- Non-empty masks with all-zero interior field (thin lesions): "
        f"{summary['interior_all_zero_records']}"
    )
    lines.append(f"- Stale manifest records (no longer eligible): {summary['stale_manifest_records']}")
    lines.append(
        f"- Excluded diagnostic manifest rows: {summary['excluded_diagnostic_records']}"
    )
    lines.append("")
    lines.append("## Status breakdown")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("| --- | ---: |")
    for status in sorted(summary["status_counts"]):
        lines.append(f"| {status} | {summary['status_counts'][status]} |")
    lines.append("")
    lines.append("## Cohort breakdown")
    lines.append("")
    lines.append("| Cohort | Eligible | Empty | Thin (interior all zero) | Primaries | Alternates |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for cohort in sorted(summary["cohorts"]):
        entry = summary["cohorts"][cohort]
        lines.append(
            f"| {cohort} | {entry['eligible']} | {entry['empty']} | "
            f"{entry['thin']} | {entry['primary']} | {entry['alternate']} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Healthy controls are eligible negative records and are retained, but "
        "`is_independent_training_unit=False`; they remain separate from segmentation training."
    )
    lines.append(
        "- Alternate annotations are preserved for provenance but are not independent "
        "training units and do not increase sampling probability."
    )
    lines.append(
        "- Empty masks produce two all-zero float32 fields; thin lesions are retained "
        "and flagged via `interior_field_all_zero` rather than altering the approved algorithm."
    )
    if summary["conflict_ids"]:
        lines.append(
            "- Conflicting record ids (not overwritten): "
            + ", ".join(summary["conflict_ids"])
        )
    if summary["failed_ids"]:
        lines.append(
            "- Failed record ids: " + ", ".join(summary["failed_ids"])
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_boundary_fields(
    manifest_path: Path,
    masks_dir: Path,
    output_dir: Path,
    *,
    tau: float = TAU,
    max_cap: float = MAX_CAP,
    regenerate: bool = False,
    limit: Optional[int] = None,
    phi_fp_dirname: str = "phi_fp",
    phi_fn_dirname: str = "phi_fn",
) -> Dict[str, Any]:
    """
    Generate/validate persisted boundary fields for every eligible manifest row.

    Parameters
    ----------
    manifest_path:
        Authoritative corrected v2 manifest CSV.
    masks_dir:
        Directory containing canonical masks (read-only).
    output_dir:
        Versioned output directory. ``phi_fp/``, ``phi_fn/``, the manifest and
        the summary are written here.
    tau, max_cap:
        Passed through unchanged to ``compute_tolerance_boundary_fields``.
    regenerate:
        Explicit opt-in that allows replacing conflicting/valid artifacts.
    limit:
        Optional debugging limit applied after eligibility (deterministic order).

    Returns
    -------
    dict
        Machine-readable summary (counts, statuses, cohort breakdown, ids).
    """
    manifest_path = Path(manifest_path)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)
    fp_dir = output_dir / phi_fp_dirname
    fn_dir = output_dir / phi_fn_dirname
    out_manifest = output_dir / "boundary_field_manifest.csv"
    out_summary = output_dir / "generation_summary.md"

    fp_dir.mkdir(parents=True, exist_ok=True)
    fn_dir.mkdir(parents=True, exist_ok=True)

    all_rows = load_manifest_rows(manifest_path)
    eligible, excluded_diag = eligible_records(manifest_path)
    if limit is not None:
        eligible = eligible[:limit]

    existing = _read_manifest_index(out_manifest)
    current_ids = {record_id_for(row) for row in eligible}
    stale_ids = sorted(set(existing) - current_ids)

    manifest_rows_out: List[Dict[str, Any]] = []
    status_counts: Dict[str, int] = {}
    cohort_stats: Dict[str, Dict[str, int]] = {}
    conflict_ids: List[str] = []
    failed_ids: List[str] = []
    empty_count = 0
    thin_count = 0

    for row in eligible:
        rec_id = record_id_for(row)
        cohort = row["cohort"]
        stats = cohort_stats.setdefault(
            cohort,
            {"eligible": 0, "empty": 0, "thin": 0, "primary": 0, "alternate": 0},
        )
        stats["eligible"] += 1
        if row.get("is_primary_reference", "").strip().lower() == "true":
            stats["primary"] += 1
        else:
            stats["alternate"] += 1

        fp_name = f"{rec_id}.npy"
        fn_name = f"{rec_id}.npy"
        fp_path = fp_dir / fp_name
        fn_path = fn_dir / fn_name
        mask_path = masks_dir / row["proc_mask_filename"]
        mask_sha_manifest = row.get("proc_mask_sha256", "")

        def _fail(status: str, actual_sha: str = "") -> None:
            status_counts[status] = status_counts.get(status, 0) + 1
            failed_ids.append(rec_id)
            manifest_rows_out.append(
                _new_manifest_row(
                    row,
                    rec_id,
                    mask_path,
                    mask_sha_manifest,
                    actual_sha,
                    False,
                    fp_name,
                    fn_name,
                    "",
                    "",
                    "",
                    "",
                    False,
                    int(row.get("proc_fg_count") or 0),
                    False,
                    status,
                    None,
                )
            )

        if not mask_path.exists():
            _fail("failed_missing_mask")
            continue

        actual_sha = sha256_file(mask_path)
        if actual_sha != mask_sha_manifest:
            _fail("failed_source_hash_mismatch", actual_sha)
            continue

        try:
            mask = _mask_to_binary(mask_path)
        except Exception:
            _fail("failed_read_error", actual_sha)
            continue

        phi_fp, phi_fn = compute_tolerance_boundary_fields(
            mask, tau=tau, max_cap=max_cap
        )
        if phi_fp.dtype != np.float32 or phi_fn.dtype != np.float32:
            raise TypeError("compute_tolerance_boundary_fields must return float32")

        expected_fp_hash = canonical_field_hash(phi_fp)
        expected_fn_hash = canonical_field_hash(phi_fn)
        fg_count = int(mask.sum())
        empty_mask = fg_count == 0
        interior_zero = (not empty_mask) and not bool(np.any(phi_fn))
        if empty_mask:
            empty_count += 1
            stats["empty"] += 1
        if interior_zero:
            thin_count += 1
            stats["thin"] += 1

        fp_exists = fp_path.exists()
        fn_exists = fn_path.exists()
        actual_fp_hash = ""
        actual_fn_hash = ""
        actual_fp_file_sha = ""
        actual_fn_file_sha = ""
        artifacts_valid = False

        if fp_exists and fn_exists:
            try:
                loaded_fp = load_field_array(fp_path)
                loaded_fn = load_field_array(fn_path)
                if loaded_fp.shape != phi_fp.shape or loaded_fn.shape != phi_fn.shape:
                    raise ValueError("persisted field shape does not match generated field")
                actual_fp_hash = canonical_field_hash(loaded_fp)
                actual_fn_hash = canonical_field_hash(loaded_fn)
                actual_fp_file_sha = sha256_file(fp_path)
                actual_fn_file_sha = sha256_file(fn_path)
                artifacts_valid = True
            except Exception:
                artifacts_valid = False

        if not fp_exists and not fn_exists:
            action = "generated"
        elif regenerate:
            action = "regenerated"
        elif artifacts_valid and (
            actual_fp_hash == expected_fp_hash
            and actual_fn_hash == expected_fn_hash
            and existing.get(rec_id, {}).get("source_mask_sha256_manifest", "")
            in ("", mask_sha_manifest)
        ):
            action = "skipped_valid"
        else:
            action = "conflict"

        if action in ("generated", "regenerated"):
            _atomic_save_npy(fp_path, phi_fp)
            _atomic_save_npy(fn_path, phi_fn)
            fp_hash = expected_fp_hash
            fn_hash = expected_fn_hash
            fp_file_sha = sha256_file(fp_path)
            fn_file_sha = sha256_file(fn_path)
        elif action == "skipped_valid":
            fp_hash = expected_fp_hash
            fn_hash = expected_fn_hash
            fp_file_sha = actual_fp_file_sha
            fn_file_sha = actual_fn_file_sha
        else:  # conflict — never overwrite
            fp_hash = actual_fp_hash
            fn_hash = actual_fn_hash
            fp_file_sha = actual_fp_file_sha
            fn_file_sha = actual_fn_file_sha
            conflict_ids.append(rec_id)

        status_counts[action] = status_counts.get(action, 0) + 1
        manifest_rows_out.append(
            _new_manifest_row(
                row,
                rec_id,
                mask_path,
                mask_sha_manifest,
                actual_sha,
                True,
                fp_name,
                fn_name,
                fp_hash,
                fn_hash,
                fp_file_sha,
                fn_file_sha,
                empty_mask,
                fg_count,
                interior_zero,
                action,
                phi_fp.shape,
            )
        )

    _write_manifest(out_manifest, manifest_rows_out)

    summary: Dict[str, Any] = {
        "field_version": FIELD_VERSION,
        "field_algorithm": FIELD_ALGORITHM,
        "hash_convention": HASH_CONVENTION,
        "tau": tau,
        "max_cap": max_cap,
        "regenerate": regenerate,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": manifest_path.as_posix(),
        "masks_dir": masks_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "manifest_rows": len(all_rows),
        "eligible_records": len(eligible),
        "manifest_records_written": len(manifest_rows_out),
        "excluded_diagnostic_records": len(excluded_diag),
        "stale_manifest_records": len(stale_ids),
        "status_counts": status_counts,
        "failed_records": len(failed_ids),
        "failed_ids": failed_ids,
        "conflict_ids": conflict_ids,
        "empty_mask_records": empty_count,
        "interior_all_zero_records": thin_count,
        "cohorts": cohort_stats,
        "has_failures": bool(failed_ids),
        "has_conflicts": bool(conflict_ids),
    }
    _write_summary(out_summary, summary)
    return summary