"""Fail-closed loader for the approved D-S1-03 exclusion registry.

The registry is a separate CSV artifact that records exactly which primary
records are excluded from the approved analysis population. It never mutates
the authoritative corrected-v2 manifest, images or masks.

Two independent things are verified here:

1. **Artifact integrity.** The sidecar pins the SHA-256 of the registry CSV, the
   corrected-v2 manifest, and the Stage 1E review CSV. Every pin is re-checked
   against the actual bytes on disk.
2. **Exact membership.** The registry must contain exactly the approved 18
   ``source_training`` and 5 ``internal_validation`` composite keys, and that
   set must equal the primary, non-diagnostic, zero-foreground records in the
   manifest. Missing, duplicate, extra, conflicting or hash-mismatched entries
   abort construction.

The native target cohort is never selected, hashed, or used to derive any
population here.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Tuple

DEFAULT_REGISTRY_PATH = (
    "data/processed/analysis_population_v1/exclusion_registry_v1.csv"
)
DEFAULT_SIDECAR_PATH = (
    "data/processed/analysis_population_v1/exclusion_registry_v1.manifest.json"
)
DEFAULT_SOURCE_IMAGE_ROOT = "data/interim/stage_3_1/extracted/AZH"
DEFAULT_REVIEW_PATH = "results/temporary/stage_1e/wp1_review_classification.csv"

#: G2-recorded SHA-256 of the approved sidecar and registry CSV.
PINNED_SIDECAR_SHA256 = (
    "e8d9f32031a7e442c44cc87824a19235b8cad3864c11a51b637a85452d8d284d"
)
PINNED_REGISTRY_CSV_SHA256 = (
    "78f02d672afd5a477d2fae6d5082fcf86a86d7459f160a6f4de8e8a2e5d1b5c7"
)

DECISION_ID = "D-S1-03 / S1-Option-E"
REGISTRY_VERSION = "v1"

#: Cohorts this registry is allowed to describe.
REGISTRY_COHORTS: Tuple[str, ...] = ("source_training", "internal_validation")

#: Exact approved exclusion membership (D-S1-03 / S1 Option E).
APPROVED_EXCLUSION_IDS: Mapping[str, FrozenSet[str]] = {
    "source_training": frozenset(
        {
            "0092", "0131", "0223", "0281", "0361", "0457", "0502", "0564",
            "0600", "0701", "0725", "0736", "0778", "0832", "0899", "0912",
            "0931", "0950",
        }
    ),
    "internal_validation": frozenset({"0128", "0417", "0483", "0533", "0869"}),
}

#: Frozen dataset cohort sizes (corrected-v2 manifest / Stage 1E audit).
FROZEN_PRIMARY_COUNTS: Mapping[str, int] = {
    "source_training": 945,
    "internal_validation": 191,
}

#: Approved analysis population after exclusions.
APPROVED_ANALYSIS_POPULATION: Mapping[str, int] = {
    "source_training": 927,
    "internal_validation": 186,
}

REQUIRED_REGISTRY_COLUMNS: FrozenSet[str] = frozenset(
    {
        "registry_version",
        "record_id",
        "cohort",
        "pair_id",
        "canonical_image_filename",
        "canonical_mask_filename",
        "orig_img_sha256",
        "proc_img_bilinear_sha256",
        "orig_mask_sha256",
        "proc_mask_sha256",
        "manifest_mask_status",
        "is_primary_reference",
        "is_diagnostic",
        "orig_fg_count",
        "proc_fg_count",
        "classification_code",
        "classification_label",
        "confidence",
        "policy_status",
        "sensitivity_role",
        "selection_allowed",
        "calibration_allowed",
        "decision_id",
        "source_reference",
    }
)

REQUIRED_SIDECAR_KEYS: FrozenSet[str] = frozenset(
    {
        "registry_version",
        "decision_id",
        "source_manifest_path",
        "source_manifest_sha256",
        "source_review_path",
        "source_review_sha256",
        "registry_csv_path",
        "registry_csv_sha256",
        "total_row_count",
        "source_training_exclusions",
        "internal_validation_exclusions",
        "expected_training_population",
        "expected_validation_population",
    }
)


class ExclusionRegistryError(RuntimeError):
    """Raised when the registry cannot be trusted, so loading must abort."""


@dataclass(frozen=True)
class ExclusionRegistry:
    """A fully validated D-S1-03 exclusion registry."""

    registry_version: str
    registry_csv_path: Path
    registry_csv_sha256: str
    sidecar_path: Path
    sidecar_sha256: str
    manifest_path: Path
    manifest_sha256: str
    review_path: Path
    review_sha256: str
    excluded_pair_ids: Mapping[str, FrozenSet[str]]
    expected_population: Mapping[str, int]
    records: Tuple[Mapping[str, str], ...]

    @property
    def cohorts(self) -> Tuple[str, ...]:
        return tuple(sorted(self.excluded_pair_ids))

    @property
    def excluded_record_ids(self) -> FrozenSet[str]:
        return frozenset(
            f"{cohort}_{pair_id}"
            for cohort, pair_ids in self.excluded_pair_ids.items()
            for pair_id in pair_ids
        )

    def for_cohort(self, cohort: str) -> FrozenSet[str]:
        """Return the excluded Pair IDs for ``cohort``.

        A cohort not covered by the registry is a hard error: silently
        returning an empty set would let an unregistered cohort through.
        """
        if cohort not in self.excluded_pair_ids:
            raise ExclusionRegistryError(
                f"Cohort {cohort!r} is not covered by the exclusion registry "
                f"({sorted(self.excluded_pair_ids)})."
            )
        return self.excluded_pair_ids[cohort]

    def assert_matches_membership(
        self, cohort: str, pair_ids: Iterable[str]
    ) -> None:
        expected = self.for_cohort(cohort)
        actual = {str(pair_id) for pair_id in pair_ids}
        missing = expected - actual
        extra = actual - expected
        if missing or extra:
            raise ExclusionRegistryError(
                f"Registry membership mismatch for cohort {cohort!r}: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------
def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _fail(message: str) -> None:
    raise ExclusionRegistryError(message)


def _is_hex64(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_csv(path: Path) -> Tuple[list[str], list[Dict[str, str]]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            _fail(f"CSV {path} has no header")
        rows = list(reader)
        return list(reader.fieldnames), rows


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------
def load_exclusion_registry(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    sidecar_path: str | Path = DEFAULT_SIDECAR_PATH,
    manifest_path: str | Path,
    project_root: str | Path,
    canonical_dir: str | Path,
    source_image_root: str | Path = DEFAULT_SOURCE_IMAGE_ROOT,
    review_path: str | Path = DEFAULT_REVIEW_PATH,
    expected_sidecar_sha256: Optional[str] = PINNED_SIDECAR_SHA256,
    expected_registry_csv_sha256: Optional[str] = PINNED_REGISTRY_CSV_SHA256,
    verify_files: bool = True,
) -> ExclusionRegistry:
    """Load and validate the registry, or raise :class:`ExclusionRegistryError`.

    ``expected_sidecar_sha256`` and ``expected_registry_csv_sha256`` default to
    the pinned v1 values. Synthetic callers may pass the fixture hashes (or
    ``None`` to trust the sidecar for that single check). Everything else is
    always enforced.
    """
    root = Path(project_root)
    registry_file = _resolve(root, registry_path)
    sidecar_file = _resolve(root, sidecar_path)
    manifest_file = _resolve(root, manifest_path)
    canonical_root = _resolve(root, canonical_dir)
    image_root = _resolve(root, source_image_root)
    review_file = _resolve(root, review_path)

    for label, path in (
        ("registry CSV", registry_file),
        ("registry sidecar", sidecar_file),
        ("corrected-v2 manifest", manifest_file),
        ("Stage 1E review CSV", review_file),
    ):
        if not path.is_file():
            _fail(f"Required {label} not found: {path}")

    # --- Sidecar integrity ------------------------------------------------
    sidecar_sha = sha256_file(sidecar_file)
    if expected_sidecar_sha256 is not None and sidecar_sha != expected_sidecar_sha256:
        _fail(
            "Exclusion-registry sidecar SHA-256 mismatch: expected "
            f"{expected_sidecar_sha256}, got {sidecar_sha} ({sidecar_file})"
        )
    try:
        sidecar: Dict[str, Any] = json.loads(
            sidecar_file.read_text(encoding="utf-8")
        )
    except Exception as exc:  # pragma: no cover - defensive
        raise ExclusionRegistryError(
            f"Cannot parse exclusion-registry sidecar {sidecar_file}: {exc}"
        ) from exc

    if not isinstance(sidecar, dict):
        _fail("Exclusion-registry sidecar must contain a JSON object")
    missing = REQUIRED_SIDECAR_KEYS - set(sidecar)
    if missing:
        _fail(f"Exclusion-registry sidecar is missing keys: {sorted(missing)}")

    if sidecar["decision_id"] != DECISION_ID:
        _fail(
            f"Sidecar decision_id {sidecar['decision_id']!r} != {DECISION_ID!r}"
        )
    if sidecar["registry_version"] != REGISTRY_VERSION:
        _fail(
            f"Sidecar registry_version {sidecar['registry_version']!r} is not "
            f"the approved version {REGISTRY_VERSION!r}"
        )

    # --- Pinned source hashes --------------------------------------------
    registry_sha = sha256_file(registry_file)
    if (
        expected_registry_csv_sha256 is not None
        and registry_sha != expected_registry_csv_sha256
    ):
        _fail(
            "Exclusion-registry CSV SHA-256 mismatch: expected "
            f"{expected_registry_csv_sha256}, got {registry_sha}"
        )
    if sidecar["registry_csv_sha256"] != registry_sha:
        _fail(
            "Registry CSV does not match the sidecar pin: sidecar "
            f"{sidecar['registry_csv_sha256']}, actual {registry_sha}"
        )

    manifest_sha = sha256_file(manifest_file)
    if sidecar["source_manifest_sha256"] != manifest_sha:
        _fail(
            "Corrected-v2 manifest SHA-256 does not match the sidecar pin: "
            f"sidecar {sidecar['source_manifest_sha256']}, actual {manifest_sha}"
        )

    review_sha = sha256_file(review_file)
    if sidecar["source_review_sha256"] != review_sha:
        _fail(
            "Stage 1E review CSV SHA-256 does not match the sidecar pin: "
            f"sidecar {sidecar['source_review_sha256']}, actual {review_sha}"
        )

    # --- Expected counts --------------------------------------------------
    expected_counts = {
        "source_training": int(sidecar["source_training_exclusions"]),
        "internal_validation": int(sidecar["internal_validation_exclusions"]),
    }
    if expected_counts["source_training"] != len(
        APPROVED_EXCLUSION_IDS["source_training"]
    ):
        _fail("Sidecar source-training exclusion count is not the approved 18")
    if expected_counts["internal_validation"] != len(
        APPROVED_EXCLUSION_IDS["internal_validation"]
    ):
        _fail("Sidecar internal-validation exclusion count is not the approved 5")
    if int(sidecar["total_row_count"]) != sum(expected_counts.values()):
        _fail("Sidecar total_row_count does not equal 18 + 5")
    if int(sidecar["expected_training_population"]) != APPROVED_ANALYSIS_POPULATION[
        "source_training"
    ]:
        _fail("Sidecar expected training population is not 927")
    if int(sidecar["expected_validation_population"]) != APPROVED_ANALYSIS_POPULATION[
        "internal_validation"
    ]:
        _fail("Sidecar expected validation population is not 186")

    # --- Registry CSV structure and policy fields ------------------------
    fieldnames, rows = _read_csv(registry_file)
    missing = REQUIRED_REGISTRY_COLUMNS - set(fieldnames)
    if missing:
        _fail(f"Registry CSV is missing columns: {sorted(missing)}")
    if any(None in row for row in rows):
        _fail("Registry CSV contains a row with more values than its header")
    if len(rows) != int(sidecar["total_row_count"]):
        _fail(
            f"Registry CSV has {len(rows)} rows; sidecar requires "
            f"{sidecar['total_row_count']}"
        )

    seen: Dict[Tuple[str, str], int] = {}
    for index, row in enumerate(rows, start=2):
        cohort = row["cohort"]
        pair_id = row["pair_id"]
        if cohort not in REGISTRY_COHORTS:
            _fail(f"Registry row {index} has unsupported cohort {cohort!r}")
        if not pair_id:
            _fail(f"Registry row {index} has an empty pair_id")
        key = (cohort, pair_id)
        if key in seen:
            _fail(
                f"Duplicate registry composite key {key} at rows "
                f"{seen[key]} and {index}"
            )
        seen[key] = index
        if row["record_id"] != f"{cohort}_{pair_id}":
            _fail(
                f"Registry row {index} record_id {row['record_id']!r} != "
                f"{cohort}_{pair_id}"
            )
        if row["registry_version"] != sidecar["registry_version"]:
            _fail(f"Registry row {index} version disagrees with sidecar")
        if row["decision_id"] != DECISION_ID:
            _fail(f"Registry row {index} has an unapproved decision_id")
        if row["policy_status"] != "exclude_primary":
            _fail(f"Registry row {index} policy_status is not exclude_primary")
        if row["sensitivity_role"] != "assumed_negative_sensitivity":
            _fail(f"Registry row {index} has an unapproved sensitivity_role")
        if row["selection_allowed"].lower() != "false":
            _fail(f"Registry row {index} allows primary selection")
        if row["calibration_allowed"].lower() != "false":
            _fail(f"Registry row {index} allows calibration")
        if row["manifest_mask_status"] != "authentic_negative_empty":
            _fail(f"Registry row {index} is not an authentic empty mask")
        if row["is_primary_reference"] != "True":
            _fail(f"Registry row {index} is not a primary reference")
        if row["is_diagnostic"] != "False":
            _fail(f"Registry row {index} is diagnostic")
        if row["orig_fg_count"] != "0" or row["proc_fg_count"] != "0":
            _fail(f"Registry row {index} has non-zero foreground")
        if not row["classification_code"] or not row["classification_label"]:
            _fail(f"Registry row {index} is missing its review classification")
        if not row["confidence"]:
            _fail(f"Registry row {index} is missing review confidence")
        if not row["source_reference"]:
            _fail(f"Registry row {index} is missing source_reference")
        for field in (
            "orig_img_sha256",
            "proc_img_bilinear_sha256",
            "orig_mask_sha256",
            "proc_mask_sha256",
        ):
            if not _is_hex64(row[field]):
                _fail(f"Registry row {index} has malformed {field}")

    # --- Exact approved membership ---------------------------------------
    actual_keys = set(seen)
    approved_keys = {
        (cohort, pair_id)
        for cohort, pair_ids in APPROVED_EXCLUSION_IDS.items()
        for pair_id in pair_ids
    }
    if actual_keys != approved_keys:
        _fail(
            "Registry membership differs from the approved D-S1-03 set: "
            f"missing={sorted(approved_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - approved_keys)}"
        )
    for cohort in REGISTRY_COHORTS:
        count = sum(1 for key in actual_keys if key[0] == cohort)
        if count != expected_counts[cohort]:
            _fail(
                f"Registry cohort {cohort!r} has {count} rows; expected "
                f"{expected_counts[cohort]}"
            )

    # --- Manifest identity ------------------------------------------------
    manifest_fieldnames, manifest_rows = _read_csv(manifest_file)
    required_manifest = {
        "cohort",
        "pair_id",
        "is_primary_reference",
        "is_diagnostic",
        "mask_status",
        "orig_fg_count",
        "proc_fg_count",
        "orig_img_path",
        "orig_mask_path",
        "orig_img_sha256",
        "proc_img_bilinear_sha256",
        "orig_mask_sha256",
        "proc_mask_filename",
        "proc_mask_sha256",
    }
    missing = required_manifest - set(manifest_fieldnames)
    if missing:
        _fail(f"Corrected-v2 manifest is missing columns: {sorted(missing)}")

    manifest_index: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in manifest_rows:
        cohort = row.get("cohort", "")
        if cohort not in REGISTRY_COHORTS:
            # Native-target rows are never used to derive an analysis population.
            continue
        key = (cohort, row.get("pair_id", ""))
        if key in manifest_index:
            _fail(f"Duplicate manifest composite key {key}")
        manifest_index[key] = row

    primary_records: Dict[str, set[str]] = {c: set() for c in REGISTRY_COHORTS}
    empty_primary_records: Dict[str, set[str]] = {c: set() for c in REGISTRY_COHORTS}
    for key, manifest_row in manifest_index.items():
        cohort, pair_id = key
        if manifest_row["is_primary_reference"] != "True":
            continue
        if manifest_row["is_diagnostic"] != "False":
            continue
        primary_records[cohort].add(pair_id)
        if (
            manifest_row["orig_fg_count"] == "0"
            or manifest_row["proc_fg_count"] == "0"
            or manifest_row["mask_status"] == "authentic_negative_empty"
        ):
            empty_primary_records[cohort].add(pair_id)

    for cohort in REGISTRY_COHORTS:
        if len(primary_records[cohort]) != FROZEN_PRIMARY_COUNTS[cohort]:
            _fail(
                f"Manifest cohort {cohort!r} has {len(primary_records[cohort])} "
                f"primary records; frozen definition requires "
                f"{FROZEN_PRIMARY_COUNTS[cohort]}"
            )
        registered = {
            pair_id for (c, pair_id) in actual_keys if c == cohort
        }
        if empty_primary_records[cohort] != registered:
            _fail(
                f"Fail-closed membership check failed for {cohort!r}: "
                f"unregistered empties="
                f"{sorted(empty_primary_records[cohort] - registered)}, "
                f"registered non-empties={sorted(registered - empty_primary_records[cohort])}"
            )

    for row in rows:
        cohort = row["cohort"]
        pair_id = row["pair_id"]
        key = (cohort, pair_id)
        manifest_row = manifest_index.get(key)
        if manifest_row is None:
            _fail(f"Registry record {cohort}/{pair_id} is absent from the manifest")
        if manifest_row["is_primary_reference"] != "True":
            _fail(f"Manifest record {cohort}/{pair_id} is not primary")
        if manifest_row["is_diagnostic"] != "False":
            _fail(f"Manifest record {cohort}/{pair_id} is diagnostic")
        if manifest_row["mask_status"] != "authentic_negative_empty":
            _fail(f"Manifest record {cohort}/{pair_id} is not an authentic empty")
        if (
            manifest_row["orig_fg_count"] != "0"
            or manifest_row["proc_fg_count"] != "0"
        ):
            _fail(f"Manifest record {cohort}/{pair_id} has non-zero foreground")
        if row["manifest_mask_status"] != manifest_row["mask_status"]:
            _fail(f"Mask status conflict for {cohort}/{pair_id}")
        if row["is_primary_reference"] != manifest_row["is_primary_reference"]:
            _fail(f"Primary-role conflict for {cohort}/{pair_id}")
        if row["is_diagnostic"] != manifest_row["is_diagnostic"]:
            _fail(f"Diagnostic-role conflict for {cohort}/{pair_id}")
        for field in {
            "orig_img_sha256",
            "proc_img_bilinear_sha256",
            "orig_mask_sha256",
            "proc_mask_sha256",
        }:
            if row[field] != manifest_row[field]:
                _fail(
                    f"Registry/manifest {field} conflict for {cohort}/{pair_id}: "
                    f"{row[field]!r} != {manifest_row[field]!r}"
                )
        expected_image = f"{cohort}_{pair_id}_bilinear.png"
        if row["canonical_image_filename"] != expected_image:
            _fail(
                f"Registry canonical image filename for {cohort}/{pair_id} is "
                f"{row['canonical_image_filename']!r}, expected {expected_image!r}"
            )
        if row["canonical_mask_filename"] != manifest_row["proc_mask_filename"]:
            _fail(f"Registry canonical mask filename conflict for {cohort}/{pair_id}")

    # --- Actual file hashes ----------------------------------------------
    if verify_files:
        for row in rows:
            cohort = row["cohort"]
            pair_id = row["pair_id"]
            manifest_row = manifest_index[(cohort, pair_id)]
            candidates = (
                (
                    "original image",
                    image_root / manifest_row["orig_img_path"],
                    row["orig_img_sha256"],
                ),
                (
                    "original mask",
                    image_root / manifest_row["orig_mask_path"],
                    row["orig_mask_sha256"],
                ),
                (
                    "canonical bilinear image",
                    canonical_root / "images" / row["canonical_image_filename"],
                    row["proc_img_bilinear_sha256"],
                ),
                (
                    "canonical mask",
                    canonical_root / "masks" / row["canonical_mask_filename"],
                    row["proc_mask_sha256"],
                ),
            )
            for label, path, expected in candidates:
                if not path.is_file():
                    _fail(
                        f"Missing {label} for {cohort}/{pair_id}: {path}"
                    )
                actual = sha256_file(path)
                if actual != expected:
                    _fail(
                        f"{label} SHA-256 mismatch for {cohort}/{pair_id} "
                        f"({path}): expected {expected}, got {actual}"
                    )

    excluded_by_cohort = {
        cohort: frozenset(
            pair_id for (c, pair_id) in actual_keys if c == cohort
        )
        for cohort in REGISTRY_COHORTS
    }
    return ExclusionRegistry(
        registry_version=sidecar["registry_version"],
        registry_csv_path=registry_file,
        registry_csv_sha256=registry_sha,
        sidecar_path=sidecar_file,
        sidecar_sha256=sidecar_sha,
        manifest_path=manifest_file,
        manifest_sha256=manifest_sha,
        review_path=review_file,
        review_sha256=review_sha,
        excluded_pair_ids=excluded_by_cohort,
        expected_population={
            "source_training": int(sidecar["expected_training_population"]),
            "internal_validation": int(sidecar["expected_validation_population"]),
        },
        records=tuple(dict(row) for row in rows),
    )