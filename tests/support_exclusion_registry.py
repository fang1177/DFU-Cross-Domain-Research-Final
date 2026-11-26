"""Synthetic fixtures for the D-S1-03 exclusion-registry tests.

Nothing here reads the real dataset. The fixture builds a small, self-contained
project whose manifest still has the frozen 945 / 191 primary cohorts, so the
production population and membership checks can run end to end without the
research images.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from src.data.boundary_fields import (
    FIELD_VERSION,
    HASH_CONVENTION,
    MAX_CAP,
    TAU,
    canonical_field_hash,
)
from src.data.exclusion_registry import (
    APPROVED_EXCLUSION_IDS,
    DECISION_ID,
    REGISTRY_VERSION,
)

SIZE = 8

_EXTRA_SOURCE = 945 - len(APPROVED_EXCLUSION_IDS["source_training"])
_EXTRA_VALIDATION = 191 - len(APPROVED_EXCLUSION_IDS["internal_validation"])


@dataclass
class PopulationFixture:
    root: Path
    manifest: Path
    registry: Path
    sidecar: Path
    review: Path
    canonical: Path
    image_root: Path
    fields: Path
    registry_rows: list[dict]
    manifest_rows: list[dict]
    registry_sha256: str
    sidecar_sha256: str
    manifest_sha256: str
    review_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _paired_ids(cohort: str, count: int, approved: Iterable[str]) -> list[str]:
    ids = list(approved)
    prefix = "x" if cohort == "source_training" else "v"
    index = 0
    while len(ids) < count:
        candidate = f"{prefix}{index:04d}"
        if candidate not in ids:
            ids.append(candidate)
        index += 1
    return ids


def _build_registry_row(record: dict) -> dict:
    return {
        "registry_version": REGISTRY_VERSION,
        "record_id": record["record_id"],
        "cohort": record["cohort"],
        "pair_id": record["pair_id"],
        "canonical_image_filename": record["canonical_image_filename"],
        "canonical_mask_filename": record["canonical_mask_filename"],
        "orig_img_sha256": record["orig_img_sha256"],
        "proc_img_bilinear_sha256": record["proc_img_bilinear_sha256"],
        "orig_mask_sha256": record["orig_mask_sha256"],
        "proc_mask_sha256": record["proc_mask_sha256"],
        "manifest_mask_status": "authentic_negative_empty",
        "is_primary_reference": "True",
        "is_diagnostic": "False",
        "orig_fg_count": "0",
        "proc_fg_count": "0",
        "classification_code": "2",
        "classification_label": "Possible annotation omission",
        "confidence": "high",
        "policy_status": "exclude_primary",
        "sensitivity_role": "assumed_negative_sensitivity",
        "selection_allowed": "false",
        "calibration_allowed": "false",
        "decision_id": DECISION_ID,
        "source_reference": "wp1_review_classification.csv:2",
    }


def write_registry_variant(
    fixture: PopulationFixture,
    destination: Path,
    *,
    rows: Sequence[Mapping[str, str]],
    manifest: Path | None = None,
    review: Path | None = None,
    sidecar_overrides: Mapping[str, object] | None = None,
) -> tuple[Path, Path]:
    """Write a registry CSV plus a sidecar pinned to it and the chosen sources.

    Used to simulate internally consistent but structurally invalid registries.
    """
    destination.mkdir(parents=True, exist_ok=True)
    registry_path = destination / "exclusion_registry_v1.csv"
    sidecar_path = destination / "exclusion_registry_v1.manifest.json"
    manifest_path = manifest or fixture.manifest
    review_path = review or fixture.review

    _write_csv(registry_path, rows)
    registry_sha = _sha256(registry_path)
    sidecar = {
        "registry_version": REGISTRY_VERSION,
        "decision_id": DECISION_ID,
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_review_path": str(review_path),
        "source_review_sha256": _sha256(review_path),
        "registry_csv_path": str(registry_path),
        "registry_csv_sha256": registry_sha,
        "total_row_count": len(rows),
        "source_training_exclusions": sum(
            1 for row in rows if row["cohort"] == "source_training"
        ),
        "internal_validation_exclusions": sum(
            1 for row in rows if row["cohort"] == "internal_validation"
        ),
        "expected_training_population": 927,
        "expected_validation_population": 186,
    }
    if sidecar_overrides:
        sidecar.update(sidecar_overrides)
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2) + "\n", encoding="utf-8"
    )
    return registry_path, sidecar_path


def copy_registry_files(
    fixture: PopulationFixture, destination: Path
) -> tuple[Path, Path]:
    """Copy only the 23 excluded records' files to an isolated tree."""
    import shutil

    canonical = destination / "canonical"
    image_root = destination / "AZH"
    (canonical / "images").mkdir(parents=True, exist_ok=True)
    (canonical / "masks").mkdir(parents=True, exist_ok=True)
    (image_root / "orig").mkdir(parents=True, exist_ok=True)

    for row in fixture.registry_rows:
        cohort = row["cohort"]
        pair_id = row["pair_id"]
        record_id = f"{cohort}_{pair_id}"
        manifest_row = next(
            item
            for item in fixture.manifest_rows
            if item["cohort"] == cohort and item["pair_id"] == pair_id
        )
        shutil.copy2(
            fixture.canonical / "images" / row["canonical_image_filename"],
            canonical / "images" / row["canonical_image_filename"],
        )
        shutil.copy2(
            fixture.canonical / "masks" / row["canonical_mask_filename"],
            canonical / "masks" / row["canonical_mask_filename"],
        )
        shutil.copy2(
            fixture.image_root / manifest_row["orig_img_path"],
            image_root / manifest_row["orig_img_path"],
        )
        shutil.copy2(
            fixture.image_root / manifest_row["orig_mask_path"],
            image_root / manifest_row["orig_mask_path"],
        )
    return canonical, image_root


def build_population(root: Path) -> PopulationFixture:
    """Create a complete synthetic project with 945 / 191 primaries."""
    root = Path(root)
    canonical = root / "canonical"
    image_root = root / "AZH"
    fields = root / "fields"
    (canonical / "images").mkdir(parents=True, exist_ok=True)
    (canonical / "masks").mkdir(parents=True, exist_ok=True)
    (image_root / "orig").mkdir(parents=True, exist_ok=True)
    (fields / "phi_fp").mkdir(parents=True, exist_ok=True)
    (fields / "phi_fn").mkdir(parents=True, exist_ok=True)

    # One shared mask and one shared zero field pair for non-excluded records.
    shared_mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    shared_mask[2, 2] = 255
    shared_mask_path = canonical / "masks" / "shared_mask.png"
    Image.fromarray(shared_mask, mode="L").save(shared_mask_path, format="PNG")
    shared_mask_sha = _sha256(shared_mask_path)

    zero_field = np.zeros((SIZE, SIZE), dtype=np.float32)
    shared_field_path = fields / "phi_fp" / "shared.npy"
    np.save(shared_field_path, zero_field, allow_pickle=False)
    np.save(fields / "phi_fn" / "shared.npy", zero_field, allow_pickle=False)
    shared_field_hash = canonical_field_hash(zero_field)

    image = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    image[..., 0] = np.arange(SIZE, dtype=np.uint8).reshape(1, SIZE)
    image[0, 0] = (255, 255, 255)

    manifest_rows: list[dict] = []
    field_rows: list[dict] = []
    registry_rows: list[dict] = []
    excluded_records: list[dict] = []

    cohorts = {
        "source_training": _paired_ids(
            "source_training",
            _EXTRA_SOURCE + len(APPROVED_EXCLUSION_IDS["source_training"]),
            APPROVED_EXCLUSION_IDS["source_training"],
        ),
        "internal_validation": _paired_ids(
            "internal_validation",
            _EXTRA_VALIDATION
            + len(APPROVED_EXCLUSION_IDS["internal_validation"]),
            APPROVED_EXCLUSION_IDS["internal_validation"],
        ),
    }

    for cohort, pair_ids in cohorts.items():
        approved = APPROVED_EXCLUSION_IDS[cohort]
        for pair_id in pair_ids:
            record_id = f"{cohort}_{pair_id}"
            excluded = pair_id in approved
            image_name = f"{cohort}_{pair_id}_bilinear.png"
            image_path = canonical / "images" / image_name
            Image.fromarray(image, mode="RGB").save(image_path, format="PNG")
            image_sha = _sha256(image_path)

            if excluded:
                mask_name = f"{cohort}_{pair_id}_empty_mask.png"
                mask_path = canonical / "masks" / mask_name
                Image.fromarray(
                    np.zeros((SIZE, SIZE), dtype=np.uint8), mode="L"
                ).save(mask_path, format="PNG")
                mask_sha = _sha256(mask_path)
                orig_image = image_root / "orig" / f"{record_id}_orig.png"
                orig_mask = image_root / "orig" / f"{record_id}_orig_mask.png"
                orig_image.write_bytes(b"original-image:" + record_id.encode())
                orig_mask.write_bytes(b"original-mask:" + record_id.encode())
                orig_image_sha = _sha256(orig_image)
                orig_mask_sha = _sha256(orig_mask)
                orig_fg = 0
                proc_fg = 0
                mask_status = "authentic_negative_empty"
            else:
                mask_name = "shared_mask.png"
                mask_sha = shared_mask_sha
                orig_image_sha = hashlib.sha256(
                    f"orig:{record_id}".encode()
                ).hexdigest()
                orig_mask_sha = hashlib.sha256(
                    f"orig-mask:{record_id}".encode()
                ).hexdigest()
                orig_fg = 1
                proc_fg = 1
                mask_status = "strict_binary"

            manifest_row = {
                "cohort": cohort,
                "pair_id": pair_id,
                "cluster_id": f"{cohort}_cluster_{pair_id}",
                "unique_content_id": f"{cohort}_content_{pair_id}",
                "is_primary_reference": "True",
                "is_diagnostic": "False",
                "orig_img_path": f"orig/{record_id}_orig.png",
                "orig_mask_path": f"orig/{record_id}_orig_mask.png",
                "orig_img_sha256": orig_image_sha,
                "proc_img_bilinear_sha256": image_sha,
                "orig_mask_sha256": orig_mask_sha,
                "proc_mask_filename": mask_name,
                "proc_mask_sha256": mask_sha,
                "mask_status": mask_status,
                "orig_fg_count": str(orig_fg),
                "proc_fg_count": str(proc_fg),
            }
            manifest_rows.append(manifest_row)

            field_rows.append(
                {
                    "record_id": record_id,
                    "pair_id": pair_id,
                    "cohort": cohort,
                    "phi_fp_filename": "shared.npy",
                    "phi_fn_filename": "shared.npy",
                    "phi_fp_content_hash": shared_field_hash,
                    "phi_fn_content_hash": shared_field_hash,
                    "source_mask_sha256_manifest": mask_sha,
                    "source_mask_hash_verified": "True",
                    "field_version": FIELD_VERSION,
                    "hash_convention": HASH_CONVENTION,
                    "tau": str(TAU),
                    "max_cap": str(MAX_CAP),
                    "generation_status": "generated",
                }
            )
            if excluded:
                excluded_records.append(manifest_row)
                registry_rows.append(
                    _build_registry_row(
                        {
                            **manifest_row,
                            "record_id": record_id,
                            "canonical_image_filename": image_name,
                            "canonical_mask_filename": mask_name,
                        }
                    )
                )

    manifest_path = root / "canonical_manifest.csv"
    field_manifest_path = fields / "boundary_field_manifest.csv"
    _write_csv(manifest_path, manifest_rows)
    _write_csv(field_manifest_path, field_rows)

    review_path = root / "wp1_review_classification.csv"
    review_rows = [
        {"cohort": row["cohort"], "pair_id": row["pair_id"]}
        for row in excluded_records
    ]
    _write_csv(review_path, review_rows)

    registry_path = root / "exclusion_registry_v1.csv"
    _write_csv(registry_path, registry_rows)
    registry_sha = _sha256(registry_path)
    manifest_sha = _sha256(manifest_path)
    review_sha = _sha256(review_path)

    sidecar_path = root / "exclusion_registry_v1.manifest.json"
    sidecar = {
        "registry_version": REGISTRY_VERSION,
        "decision_id": DECISION_ID,
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": manifest_sha,
        "source_review_path": str(review_path),
        "source_review_sha256": review_sha,
        "registry_csv_path": str(registry_path),
        "registry_csv_sha256": registry_sha,
        "total_row_count": len(registry_rows),
        "source_training_exclusions": len(APPROVED_EXCLUSION_IDS["source_training"]),
        "internal_validation_exclusions": len(
            APPROVED_EXCLUSION_IDS["internal_validation"]
        ),
        "expected_training_population": 927,
        "expected_validation_population": 186,
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")

    return PopulationFixture(
        root=root,
        manifest=manifest_path,
        registry=registry_path,
        sidecar=sidecar_path,
        review=review_path,
        canonical=canonical,
        image_root=image_root,
        fields=fields,
        registry_rows=registry_rows,
        manifest_rows=manifest_rows,
        registry_sha256=registry_sha,
        sidecar_sha256=_sha256(sidecar_path),
        manifest_sha256=manifest_sha,
        review_sha256=review_sha,
    )