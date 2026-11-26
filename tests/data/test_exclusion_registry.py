"""Focused fail-closed tests for the D-S1-03 exclusion registry.

All data is synthetic. No real research image, manifest mutation or target
artifact is used.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from src.data.exclusion_registry import (
    APPROVED_ANALYSIS_POPULATION,
    APPROVED_EXCLUSION_IDS,
    ExclusionRegistryError,
    load_exclusion_registry,
)
from tests.support_exclusion_registry import (
    PopulationFixture,
    copy_registry_files,
    write_registry_variant,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(
    fixture: PopulationFixture,
    *,
    registry: Path | None = None,
    sidecar: Path | None = None,
    manifest: Path | None = None,
    review: Path | None = None,
    canonical: Path | None = None,
    image_root: Path | None = None,
    expected_registry_csv_sha256: str | None = None,
    expected_sidecar_sha256: str | None = None,
    verify_files: bool = True,
):
    return load_exclusion_registry(
        registry_path=registry or fixture.registry,
        sidecar_path=sidecar or fixture.sidecar,
        manifest_path=manifest or fixture.manifest,
        project_root=fixture.root,
        canonical_dir=canonical or fixture.canonical,
        source_image_root=image_root or fixture.image_root,
        review_path=review or fixture.review,
        expected_sidecar_sha256=expected_sidecar_sha256,
        expected_registry_csv_sha256=expected_registry_csv_sha256,
        verify_files=verify_files,
    )


# ---------------------------------------------------------------------------
# Happy path and exact membership
# ---------------------------------------------------------------------------
def test_valid_registry_loads_with_exact_membership(exclusion_population):
    fixture = exclusion_population
    registry = _load(
        fixture,
        expected_registry_csv_sha256=fixture.registry_sha256,
        expected_sidecar_sha256=fixture.sidecar_sha256,
    )

    assert registry.registry_csv_sha256 == fixture.registry_sha256
    assert registry.sidecar_sha256 == fixture.sidecar_sha256
    assert registry.manifest_sha256 == fixture.manifest_sha256
    assert registry.review_sha256 == fixture.review_sha256
    assert len(registry.records) == 23
    assert dict(registry.expected_population) == {
        "source_training": APPROVED_ANALYSIS_POPULATION["source_training"],
        "internal_validation": APPROVED_ANALYSIS_POPULATION["internal_validation"],
    }

    for cohort, approved in APPROVED_EXCLUSION_IDS.items():
        assert registry.for_cohort(cohort) == approved
        registry.assert_matches_membership(cohort, approved)

    assert registry.excluded_record_ids == frozenset(
        f"{cohort}_{pair_id}"
        for cohort, pair_ids in APPROVED_EXCLUSION_IDS.items()
        for pair_id in pair_ids
    )
    with pytest.raises(ExclusionRegistryError, match="not covered"):
        registry.for_cohort("native_target")


def test_registry_does_not_modify_manifest(exclusion_population):
    fixture = exclusion_population
    before = _sha256(fixture.manifest)
    _load(fixture)
    assert _sha256(fixture.manifest) == before == fixture.manifest_sha256


# ---------------------------------------------------------------------------
# Pin / source hash mismatches
# ---------------------------------------------------------------------------
def test_registry_csv_hash_mismatch_rejected(exclusion_population, tmp_path):
    fixture = exclusion_population
    corrupted = tmp_path / "registry.csv"
    shutil.copy2(fixture.registry, corrupted)
    corrupted.write_bytes(corrupted.read_bytes() + b"\n")

    with pytest.raises(ExclusionRegistryError, match="CSV SHA-256 mismatch"):
        _load(
            fixture,
            registry=corrupted,
            expected_registry_csv_sha256=fixture.registry_sha256,
        )


def test_sidecar_hash_mismatch_rejected(exclusion_population, tmp_path):
    fixture = exclusion_population
    corrupted = tmp_path / "sidecar.json"
    shutil.copy2(fixture.sidecar, corrupted)
    corrupted.write_text(corrupted.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ExclusionRegistryError, match="sidecar SHA-256 mismatch"):
        _load(
            fixture,
            sidecar=corrupted,
            expected_sidecar_sha256=fixture.sidecar_sha256,
        )


def test_manifest_hash_mismatch_rejected(exclusion_population, tmp_path):
    fixture = exclusion_population
    manifest = tmp_path / "manifest.csv"
    shutil.copy2(fixture.manifest, manifest)
    manifest.write_bytes(manifest.read_bytes() + b"\n")

    with pytest.raises(ExclusionRegistryError, match="manifest.*SHA-256"):
        _load(fixture, manifest=manifest, expected_sidecar_sha256=None)


def test_review_hash_mismatch_rejected(exclusion_population, tmp_path):
    fixture = exclusion_population
    review = tmp_path / "review.csv"
    shutil.copy2(fixture.review, review)
    review.write_bytes(review.read_bytes() + b"\n")

    with pytest.raises(ExclusionRegistryError, match="review CSV.*SHA-256"):
        _load(fixture, review=review, expected_sidecar_sha256=None)


# ---------------------------------------------------------------------------
# Structural and membership validation
# ---------------------------------------------------------------------------
def test_missing_registry_entry_rejected(exclusion_population, tmp_path):
    fixture = exclusion_population
    registry_path, sidecar_path = write_registry_variant(
        fixture,
        tmp_path,
        rows=fixture.registry_rows[:-1],
        sidecar_overrides={
            "total_row_count": 23,
            "source_training_exclusions": 18,
            "internal_validation_exclusions": 5,
        },
    )
    with pytest.raises(ExclusionRegistryError, match="has 22 rows"):
        _load(fixture, registry=registry_path, sidecar=sidecar_path)


def test_duplicate_registry_entry_rejected(exclusion_population, tmp_path):
    fixture = exclusion_population
    rows = list(fixture.registry_rows)
    rows[-1] = dict(rows[0])
    registry_path, sidecar_path = write_registry_variant(
        fixture,
        tmp_path,
        rows=rows,
        sidecar_overrides={
            "total_row_count": 23,
            "source_training_exclusions": 18,
            "internal_validation_exclusions": 5,
        },
    )
    with pytest.raises(ExclusionRegistryError, match="Duplicate registry composite key"):
        _load(fixture, registry=registry_path, sidecar=sidecar_path)


def test_extra_non_approved_record_rejected(exclusion_population, tmp_path):
    fixture = exclusion_population
    rows = [dict(row) for row in fixture.registry_rows]
    rows[0]["pair_id"] = "9999"
    rows[0]["record_id"] = f"{rows[0]['cohort']}_9999"
    registry_path, sidecar_path = write_registry_variant(
        fixture, tmp_path, rows=rows
    )
    with pytest.raises(ExclusionRegistryError, match="approved D-S1-03 set"):
        _load(fixture, registry=registry_path, sidecar=sidecar_path)


def test_wrong_cohort_record_rejected(exclusion_population, tmp_path):
    fixture = exclusion_population
    rows = [dict(row) for row in fixture.registry_rows]
    rows[0]["cohort"] = "mismatched_target"
    rows[0]["record_id"] = f"mismatched_target_{rows[0]['pair_id']}"
    registry_path, sidecar_path = write_registry_variant(
        fixture,
        tmp_path,
        rows=rows,
        sidecar_overrides={
            "total_row_count": 23,
            "source_training_exclusions": 18,
            "internal_validation_exclusions": 5,
        },
    )
    with pytest.raises(ExclusionRegistryError, match="unsupported cohort"):
        _load(fixture, registry=registry_path, sidecar=sidecar_path)


def test_missing_required_column_rejected(exclusion_population, tmp_path):
    fixture = exclusion_population
    rows = [{key: value for key, value in row.items() if key != "policy_status"}
            for row in fixture.registry_rows]
    registry_path, sidecar_path = write_registry_variant(
        fixture, tmp_path, rows=rows
    )
    with pytest.raises(ExclusionRegistryError, match="missing columns"):
        _load(fixture, registry=registry_path, sidecar=sidecar_path)


# ---------------------------------------------------------------------------
# Actual file hash verification
# ---------------------------------------------------------------------------
def test_original_image_hash_mismatch_rejected(
    exclusion_population, tmp_path
):
    fixture = exclusion_population
    canonical, image_root = copy_registry_files(fixture, tmp_path / "files")
    row = fixture.registry_rows[0]
    manifest_row = next(
        item
        for item in fixture.manifest_rows
        if item["cohort"] == row["cohort"] and item["pair_id"] == row["pair_id"]
    )
    (image_root / manifest_row["orig_img_path"]).write_bytes(b"tampered")

    with pytest.raises(ExclusionRegistryError, match="original image SHA-256 mismatch"):
        _load(fixture, canonical=canonical, image_root=image_root)


def test_canonical_image_hash_mismatch_rejected(
    exclusion_population, tmp_path
):
    fixture = exclusion_population
    canonical, image_root = copy_registry_files(fixture, tmp_path / "files")
    row = fixture.registry_rows[0]
    (canonical / "images" / row["canonical_image_filename"]).write_bytes(b"tampered")

    with pytest.raises(
        ExclusionRegistryError, match="canonical bilinear image SHA-256 mismatch"
    ):
        _load(fixture, canonical=canonical, image_root=image_root)


def test_canonical_mask_hash_mismatch_rejected(exclusion_population, tmp_path):
    fixture = exclusion_population
    canonical, image_root = copy_registry_files(fixture, tmp_path / "files")
    row = fixture.registry_rows[0]
    (canonical / "masks" / row["canonical_mask_filename"]).write_bytes(b"tampered")

    with pytest.raises(ExclusionRegistryError, match="canonical mask SHA-256 mismatch"):
        _load(fixture, canonical=canonical, image_root=image_root)


def test_original_mask_hash_mismatch_rejected(exclusion_population, tmp_path):
    fixture = exclusion_population
    canonical, image_root = copy_registry_files(fixture, tmp_path / "files")
    row = fixture.registry_rows[0]
    manifest_row = next(
        item
        for item in fixture.manifest_rows
        if item["cohort"] == row["cohort"] and item["pair_id"] == row["pair_id"]
    )
    (image_root / manifest_row["orig_mask_path"]).write_bytes(b"tampered")

    with pytest.raises(ExclusionRegistryError, match="original mask SHA-256 mismatch"):
        _load(fixture, canonical=canonical, image_root=image_root)