"""
Focused tests for Stage 4A Task 3B boundary-field generation and persistence.

All tests use temporary directories. The real dataset is never used as an
output location.
"""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.boundary_fields import (
    HASH_CONVENTION,
    canonical_field_hash,
    canonical_field_header,
    generate_boundary_fields,
    load_field_array,
)
from src.data.transforms import compute_tolerance_boundary_fields

MANIFEST_COLUMNS = [
    "cohort",
    "pair_id",
    "cluster_id",
    "unique_content_id",
    "is_primary_reference",
    "is_diagnostic",
    "mask_status",
    "proc_mask_filename",
    "proc_mask_sha256",
    "proc_fg_count",
]


def _save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path, format="PNG")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _make_row(
    *,
    cohort: str,
    pair_id: str,
    mask: np.ndarray,
    mask_status: str = "strict_binary",
    is_primary: bool = True,
    is_diagnostic: bool = False,
    cluster_id: str | None = None,
    masks_dir: Path,
    mask_filename: str | None = None,
    sha_override: str | None = None,
) -> dict:
    filename = mask_filename or f"{cohort}_{pair_id}_canonical_mask.png"
    path = masks_dir / filename
    _save_mask(path, mask)
    return {
        "cohort": cohort,
        "pair_id": pair_id,
        "cluster_id": cluster_id or f"{cohort}_cluster_{pair_id}",
        "unique_content_id": f"{cohort}_img_{pair_id}",
        "is_primary_reference": str(is_primary),
        "is_diagnostic": str(is_diagnostic),
        "mask_status": mask_status,
        "proc_mask_filename": filename,
        "proc_mask_sha256": sha_override if sha_override is not None else _sha256(path),
        "proc_fg_count": str(int(mask.sum())),
    }


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _read_manifest(path: Path) -> dict[str, dict]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return {row["record_id"]: row for row in csv.DictReader(handle)}


def _block_mask(size: int = 32, start: int = 12, stop: int = 20) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[start:stop, start:stop] = True
    return mask


def _setup(tmp_path: Path, rows: list[dict]) -> tuple[Path, Path, Path]:
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, rows)
    out = tmp_path / "output"
    return manifest, masks_dir, out


# ---------------------------------------------------------------------------
# 1. Empty mask -> exactly zero fields
# ---------------------------------------------------------------------------
def test_empty_mask_produces_zero_fields(tmp_path: Path):
    masks = tmp_path / "masks"
    masks.mkdir()
    row = _make_row(
        cohort="healthy_controls",
        pair_id="0001",
        mask=np.zeros((16, 16), dtype=bool),
        mask_status="authentic_negative_healthy",
        masks_dir=masks,
    )
    manifest, _, out = _setup(tmp_path, [row])
    summary = generate_boundary_fields(manifest, masks, out)

    rec = _read_manifest(out / "boundary_field_manifest.csv")["healthy_controls_0001"]
    assert rec["generation_status"] == "generated"
    assert rec["is_empty_mask"] == "True"
    assert rec["interior_field_all_zero"] == "False"
    assert summary["empty_mask_records"] == 1

    fp = load_field_array(out / "phi_fp" / "healthy_controls_0001.npy")
    fn = load_field_array(out / "phi_fn" / "healthy_controls_0001.npy")
    assert fp.dtype == np.float32 and fn.dtype == np.float32
    assert np.array_equal(fp, np.zeros((16, 16), dtype=np.float32))
    assert np.array_equal(fn, np.zeros((16, 16), dtype=np.float32))


# ---------------------------------------------------------------------------
# 2. Non-empty mask -> float32 shape and expected signs
# ---------------------------------------------------------------------------
def test_non_empty_mask_shape_and_signs(tmp_path: Path):
    masks = tmp_path / "masks"
    masks.mkdir()
    mask = _block_mask(32, 12, 20)
    row = _make_row(
        cohort="source_training", pair_id="0002", mask=mask, masks_dir=masks
    )
    manifest, _, out = _setup(tmp_path, [row])
    generate_boundary_fields(manifest, masks, out)

    fp = load_field_array(out / "phi_fp" / "source_training_0002.npy")
    fn = load_field_array(out / "phi_fn" / "source_training_0002.npy")
    assert fp.shape == (32, 32) and fn.shape == (32, 32)
    assert fp.dtype == np.float32 and fn.dtype == np.float32

    # Interior center is in the interior (false-negative) zone only.
    assert fn[16, 16] > 0
    assert fp[16, 16] == 0
    # Far corner is exterior (false-positive) zone only.
    assert fp[0, 0] > 0
    assert fn[0, 0] == 0
    # Signed field convention.
    signed = fp - fn
    assert signed[16, 16] < 0 and signed[0, 0] > 0


# ---------------------------------------------------------------------------
# 3. Thin lesion -> no crash and zero-interior status recorded
# ---------------------------------------------------------------------------
def test_thin_lesion_records_zero_interior(tmp_path: Path):
    masks = tmp_path / "masks"
    masks.mkdir()
    mask = np.zeros((24, 24), dtype=bool)
    mask[11, 4:20] = True  # one-pixel-wide lesion
    row = _make_row(
        cohort="source_training", pair_id="0003", mask=mask, masks_dir=masks
    )
    manifest, _, out = _setup(tmp_path, [row])
    summary = generate_boundary_fields(manifest, masks, out)

    fn = load_field_array(out / "phi_fn" / "source_training_0003.npy")
    assert not np.any(fn)
    rec = _read_manifest(out / "boundary_field_manifest.csv")["source_training_0003"]
    assert rec["interior_field_all_zero"] == "True"
    assert rec["is_empty_mask"] == "False"
    assert summary["interior_all_zero_records"] == 1
    assert not summary["has_failures"]


# ---------------------------------------------------------------------------
# 4. tau and max_cap are applied correctly
# ---------------------------------------------------------------------------
def test_tau_and_max_cap_applied(tmp_path: Path):
    mask = np.zeros((64, 64), dtype=bool)
    mask[30:34, 30:34] = True

    phi_fp, phi_fn = compute_tolerance_boundary_fields(mask, tau=2.0, max_cap=20.0)
    assert float(phi_fp.max()) == pytest.approx(20.0)
    assert float(phi_fp.max()) <= 20.0

    # Explicit independent recomputation of the approved formula.
    from scipy.ndimage import binary_erosion, distance_transform_edt

    eroded = binary_erosion(mask, structure=np.ones((3, 3)))
    edt = distance_transform_edt((mask ^ eroded) == 0)
    expected_fp = np.zeros(mask.shape, dtype=np.float32)
    zone = (~mask) & (edt > 2.0)
    expected_fp[zone] = np.minimum(edt[zone] - 2.0, 20.0)
    assert np.array_equal(phi_fp, expected_fp)

    # Larger tolerance suppresses the near-boundary band.
    phi_fp_tau5, _ = compute_tolerance_boundary_fields(mask, tau=5.0, max_cap=20.0)
    assert not np.any(phi_fp_tau5[edt <= 5.0])
    assert int(np.count_nonzero(phi_fp_tau5)) < int(np.count_nonzero(phi_fp))

    # Persisted provenance records the parameters.
    masks = tmp_path / "masks"
    masks.mkdir()
    row = _make_row(
        cohort="native_target",
        pair_id="0004",
        mask=mask,
        mask_status="canonical_binarized_jpeg",
        masks_dir=masks,
    )
    manifest, _, out = _setup(tmp_path, [row])
    generate_boundary_fields(manifest, masks, out, tau=2.0, max_cap=20.0)
    rec = _read_manifest(out / "boundary_field_manifest.csv")["native_target_0004"]
    assert rec["tau"] == "2.0"
    assert rec["max_cap"] == "20.0"


# ---------------------------------------------------------------------------
# 5. Deterministic output across repeated generation
# ---------------------------------------------------------------------------
def test_repeated_generation_is_deterministic(tmp_path: Path):
    masks = tmp_path / "masks"
    masks.mkdir()
    row = _make_row(
        cohort="source_training", pair_id="0005", mask=_block_mask(), masks_dir=masks
    )
    manifest, _, out = _setup(tmp_path, [row])

    first = generate_boundary_fields(manifest, masks, out)
    fp_bytes_1 = (out / "phi_fp" / "source_training_0005.npy").read_bytes()
    fn_bytes_1 = (out / "phi_fn" / "source_training_0005.npy").read_bytes()

    second = generate_boundary_fields(manifest, masks, out)
    fp_bytes_2 = (out / "phi_fp" / "source_training_0005.npy").read_bytes()
    fn_bytes_2 = (out / "phi_fn" / "source_training_0005.npy").read_bytes()

    assert first["status_counts"] == {"generated": 1}
    assert second["status_counts"] == {"skipped_valid": 1}
    assert fp_bytes_1 == fp_bytes_2
    assert fn_bytes_1 == fn_bytes_2


# ---------------------------------------------------------------------------
# 6. Canonical field hash independently recomputable
# ---------------------------------------------------------------------------
def test_canonical_hash_is_self_describing_and_recomputable(tmp_path: Path):
    masks = tmp_path / "masks"
    masks.mkdir()
    row = _make_row(
        cohort="source_training", pair_id="0006", mask=_block_mask(), masks_dir=masks
    )
    manifest, _, out = _setup(tmp_path, [row])
    generate_boundary_fields(manifest, masks, out)

    field_path = out / "phi_fp" / "source_training_0006.npy"
    arr = np.load(field_path, allow_pickle=False)
    rec = _read_manifest(out / "boundary_field_manifest.csv")["source_training_0006"]

    # Rebuild the canonical byte stream from scratch.
    header = (
        b"DFU-BOUNDARY-FIELD|hash=dfu-canonical-field-sha256-v1|"
        b"dtype=float32|endian=little|shape=32x32|order=C|"
    )
    independent = hashlib.sha256(
        header + np.ascontiguousarray(arr, dtype="<f4").tobytes(order="C")
    ).hexdigest()
    assert independent == rec["phi_fp_content_hash"]
    assert independent == canonical_field_hash(arr)
    assert canonical_field_header(arr) == header
    assert rec["hash_convention"] == HASH_CONVENTION

    # The canonical hash differs from the old raw-bytes hash.
    raw = hashlib.sha256(arr.tobytes()).hexdigest()
    assert raw != independent


# ---------------------------------------------------------------------------
# 7. Source-mask hash mismatch is detected and blocks generation
# ---------------------------------------------------------------------------
def test_source_hash_mismatch_blocks_generation(tmp_path: Path):
    masks = tmp_path / "masks"
    masks.mkdir()
    row = _make_row(
        cohort="source_training",
        pair_id="0007",
        mask=_block_mask(),
        masks_dir=masks,
        sha_override="0" * 64,
    )
    manifest, _, out = _setup(tmp_path, [row])
    summary = generate_boundary_fields(manifest, masks, out)

    assert summary["has_failures"]
    assert summary["status_counts"] == {"failed_source_hash_mismatch": 1}
    assert not (out / "phi_fp" / "source_training_0007.npy").exists()
    assert not (out / "phi_fn" / "source_training_0007.npy").exists()
    rec = _read_manifest(out / "boundary_field_manifest.csv")["source_training_0007"]
    assert rec["generation_status"] == "failed_source_hash_mismatch"
    assert rec["source_mask_hash_verified"] == "False"


# ---------------------------------------------------------------------------
# 8. Existing matching artifacts are not rewritten
# ---------------------------------------------------------------------------
def test_matching_artifacts_are_not_rewritten(tmp_path: Path):
    masks = tmp_path / "masks"
    masks.mkdir()
    row = _make_row(
        cohort="source_training", pair_id="0008", mask=_block_mask(), masks_dir=masks
    )
    manifest, _, out = _setup(tmp_path, [row])
    generate_boundary_fields(manifest, masks, out)

    fp_path = out / "phi_fp" / "source_training_0008.npy"
    fn_path = out / "phi_fn" / "source_training_0008.npy"
    before = (fp_path.stat().st_mtime_ns, fn_path.stat().st_mtime_ns)

    summary = generate_boundary_fields(manifest, masks, out)
    after = (fp_path.stat().st_mtime_ns, fn_path.stat().st_mtime_ns)

    assert summary["status_counts"] == {"skipped_valid": 1}
    assert before == after


# ---------------------------------------------------------------------------
# 9. Conflicting/stale artifacts are detected, not silently overwritten
# ---------------------------------------------------------------------------
def test_conflicting_artifact_is_detected_and_requires_explicit_regeneration(
    tmp_path: Path,
):
    masks = tmp_path / "masks"
    masks.mkdir()
    row = _make_row(
        cohort="source_training", pair_id="0009", mask=_block_mask(), masks_dir=masks
    )
    manifest, _, out = _setup(tmp_path, [row])
    generate_boundary_fields(manifest, masks, out)

    fp_path = out / "phi_fp" / "source_training_0009.npy"
    expected = load_field_array(fp_path).copy()
    tampered = np.zeros_like(expected)
    np.save(fp_path, tampered, allow_pickle=False)
    tampered_hash = canonical_field_hash(tampered)

    summary = generate_boundary_fields(manifest, masks, out)
    assert summary["has_conflicts"]
    assert summary["status_counts"] == {"conflict": 1}
    # Not overwritten.
    still_tampered = load_field_array(fp_path)
    assert np.array_equal(still_tampered, tampered)
    rec = _read_manifest(out / "boundary_field_manifest.csv")["source_training_0009"]
    assert rec["generation_status"] == "conflict"
    assert rec["phi_fp_content_hash"] == tampered_hash

    # Explicit regeneration is required to repair the conflict.
    repaired = generate_boundary_fields(manifest, masks, out, regenerate=True)
    assert repaired["status_counts"] == {"regenerated": 1}
    assert np.array_equal(load_field_array(fp_path), expected)


def test_partial_artifact_is_a_conflict(tmp_path: Path):
    masks = tmp_path / "masks"
    masks.mkdir()
    row = _make_row(
        cohort="source_training", pair_id="0010", mask=_block_mask(), masks_dir=masks
    )
    manifest, _, out = _setup(tmp_path, [row])
    generate_boundary_fields(manifest, masks, out)

    (out / "phi_fn" / "source_training_0010.npy").unlink()
    summary = generate_boundary_fields(manifest, masks, out)
    assert summary["has_conflicts"]
    # The remaining artifact is left untouched.
    assert (out / "phi_fp" / "source_training_0010.npy").exists()
    assert not (out / "phi_fn" / "source_training_0010.npy").exists()


def test_stale_manifest_records_are_detected_and_left_untouched(tmp_path: Path):
    masks = tmp_path / "masks"
    masks.mkdir()
    rows = [
        _make_row(
            cohort="source_training", pair_id="0011", mask=_block_mask(), masks_dir=masks
        ),
        _make_row(
            cohort="source_training", pair_id="0012", mask=_block_mask(), masks_dir=masks
        ),
    ]
    manifest, _, out = _setup(tmp_path, rows)
    generate_boundary_fields(manifest, masks, out)
    stale_file = out / "phi_fp" / "source_training_0012.npy"
    assert stale_file.exists()

    # Remove the second record from the authoritative manifest.
    with open(manifest, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerow(rows[0])

    summary = generate_boundary_fields(manifest, masks, out)
    assert summary["stale_manifest_records"] == 1
    assert summary["eligible_records"] == 1
    # Stale artifacts are reported and never deleted or overwritten.
    assert stale_file.exists()
    recs = _read_manifest(out / "boundary_field_manifest.csv")
    assert set(recs) == {"source_training_0011"}


# ---------------------------------------------------------------------------
# 10. Diagnostic samples are excluded
# ---------------------------------------------------------------------------
def test_diagnostic_samples_are_excluded(tmp_path: Path):
    masks = tmp_path / "masks"
    masks.mkdir()
    rows = [
        _make_row(
            cohort="mismatched_target",
            pair_id="0001",
            mask=_block_mask(),
            mask_status="diagnostic_continuous_grayscale",
            is_diagnostic=True,
            masks_dir=masks,
        ),
        _make_row(
            cohort="internal_validation",
            pair_id="0233",
            mask=_block_mask(),
            mask_status="diagnostic_multiclass",
            is_diagnostic=True,
            masks_dir=masks,
        ),
        _make_row(
            cohort="source_training",
            pair_id="0001",
            mask=_block_mask(),
            masks_dir=masks,
        ),
    ]
    manifest, _, out = _setup(tmp_path, rows)
    summary = generate_boundary_fields(manifest, masks, out)

    recs = _read_manifest(out / "boundary_field_manifest.csv")
    assert set(recs) == {"source_training_0001"}
    assert summary["eligible_records"] == 1
    assert summary["excluded_diagnostic_records"] == 2
    assert not (out / "phi_fp" / "mismatched_target_0001.npy").exists()


# ---------------------------------------------------------------------------
# 11. Alternates preserved but not independent training units
# ---------------------------------------------------------------------------
def test_alternates_preserved_but_not_independent(tmp_path: Path):
    masks = tmp_path / "masks"
    masks.mkdir()
    rows = [
        _make_row(
            cohort="source_training",
            pair_id="0001",
            mask=_block_mask(),
            is_primary=True,
            cluster_id="source_cluster_0001",
            masks_dir=masks,
        ),
        _make_row(
            cohort="source_training",
            pair_id="0002",
            mask=_block_mask(32, 4, 10),
            is_primary=False,
            cluster_id="source_cluster_0001",
            masks_dir=masks,
        ),
        _make_row(
            cohort="healthy_controls",
            pair_id="0003",
            mask=np.zeros((16, 16), dtype=bool),
            mask_status="authentic_negative_healthy",
            is_primary=True,
            masks_dir=masks,
        ),
    ]
    manifest, _, out = _setup(tmp_path, rows)
    summary = generate_boundary_fields(manifest, masks, out)
    recs = _read_manifest(out / "boundary_field_manifest.csv")

    assert summary["eligible_records"] == 3
    alt = recs["source_training_0002"]
    assert alt["annotation_role"] == "alternate"
    assert alt["unit_role"] == "alternate_annotation"
    assert alt["is_independent_training_unit"] == "False"
    assert alt["generation_status"] == "generated"

    primary = recs["source_training_0001"]
    assert primary["annotation_role"] == "primary"
    assert primary["is_independent_training_unit"] == "True"

    healthy = recs["healthy_controls_0003"]
    assert healthy["unit_role"] == "healthy_control_negative"
    assert healthy["is_independent_training_unit"] == "False"


# ---------------------------------------------------------------------------
# 12. Exact eligible-record counts and generation summary
# ---------------------------------------------------------------------------
def test_counts_and_summary(tmp_path: Path):
    masks = tmp_path / "masks"
    masks.mkdir()
    rows = [
        _make_row(
            cohort="source_training", pair_id="0001", mask=_block_mask(), masks_dir=masks
        ),
        _make_row(
            cohort="source_training", pair_id="0002", mask=_block_mask(), masks_dir=masks
        ),
        _make_row(
            cohort="native_target",
            pair_id="0001",
            mask=_block_mask(),
            mask_status="canonical_binarized_jpeg",
            masks_dir=masks,
        ),
        _make_row(
            cohort="healthy_controls",
            pair_id="0001",
            mask=np.zeros((16, 16), dtype=bool),
            mask_status="authentic_negative_healthy",
            masks_dir=masks,
        ),
        _make_row(
            cohort="source_training",
            pair_id="9999",
            mask=_block_mask(),
            mask_status="diagnostic_continuous_grayscale",
            is_diagnostic=True,
            masks_dir=masks,
        ),
    ]
    manifest, _, out = _setup(tmp_path, rows)
    summary = generate_boundary_fields(manifest, masks, out)

    assert summary["manifest_rows"] == 5
    assert summary["eligible_records"] == 4
    assert summary["manifest_records_written"] == 4
    assert summary["status_counts"] == {"generated": 4}
    assert summary["empty_mask_records"] == 1
    assert summary["excluded_diagnostic_records"] == 1

    assert summary["cohorts"]["source_training"]["eligible"] == 2
    assert summary["cohorts"]["native_target"]["eligible"] == 1
    assert summary["cohorts"]["healthy_controls"]["eligible"] == 1

    manifest_file = out / "boundary_field_manifest.csv"
    summary_file = out / "generation_summary.md"
    assert manifest_file.exists() and summary_file.exists()
    text = summary_file.read_text(encoding="utf-8")
    assert "Eligible field records: 4" in text
    assert "Empty masks: 1" in text

    # Every generated field is loadable with the required dtype/shape.
    for rec_id in ("source_training_0001", "native_target_0001"):
        arr = load_field_array(out / "phi_fp" / f"{rec_id}.npy")
        assert arr.dtype == np.float32 and arr.shape == (32, 32)


# ---------------------------------------------------------------------------
# Extra: record-id collision must fail loudly
# ---------------------------------------------------------------------------
def test_record_id_collision_is_rejected(tmp_path: Path):
    masks = tmp_path / "masks"
    masks.mkdir()
    rows = [
        _make_row(
            cohort="source_training", pair_id="0001", mask=_block_mask(), masks_dir=masks
        ),
        _make_row(
            cohort="source_training", pair_id="0001", mask=_block_mask(), masks_dir=masks
        ),
    ]
    manifest, _, out = _setup(tmp_path, rows)
    with pytest.raises(ValueError, match="collision"):
        generate_boundary_fields(manifest, masks, out)