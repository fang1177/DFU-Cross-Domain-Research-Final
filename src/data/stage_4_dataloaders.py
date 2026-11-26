"""
Production Stage 4 segmentation Dataset and DataLoader integration.

Authoritative sources
---------------------
* ``stage_3_2B_manifest_repair_v2/canonical_manifest_corrected_v2.csv``
* ``stage_3_2B_canonical/images/`` and ``stage_3_2B_canonical/masks/``
* ``stage_4A_boundary_fields_v1/phi_fp/``, ``phi_fn/`` and
  ``boundary_field_manifest.csv``

Approved annotation policy
--------------------------
Each segmentation dataset contains exactly one primary record per
``orig_img_sha256`` image-content cluster, selected by the lowest alphanumeric
Pair ID. The manifest's ``is_primary_reference`` flag is cross-checked against
that derivation; a disagreement is a hard error. Alternates, diagnostics,
healthy controls, and mismatched-target records are never sample units for the
segmentation loaders.

Tensor contract
---------------
``image``   float32 ``[3, H, W]``  raw RGB in ``[0, 1]`` (not normalized)
``mask``    float32 ``[1, H, W]``  binary ``{0, 1}``
``phi_fp``  float32 ``[1, H, W]``  persisted Task 3B field
``phi_fn``  float32 ``[1, H, W]``  persisted Task 3B field
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.boundary_fields import (
    FIELD_VERSION,
    HASH_CONVENTION,
    MAX_CAP,
    TAU,
    canonical_field_hash,
)
from src.data.config import Stage4DataLoaderConfig
from src.data.exclusion_registry import ExclusionRegistry
from src.data.samplers import EpochShuffleSampler
from src.data.transforms import (
    augment_sample,
    compute_tolerance_boundary_fields,
    derive_augmentation_seed,
)

import os
DEFAULT_PROJECT_ROOT = os.environ.get("DFU_PROJECT_ROOT", "E:/DFU_Cross_Domain_Research")
DEFAULT_MANIFEST = (
    "data/processed/stage_3_2B_manifest_repair_v2/"
    "canonical_manifest_corrected_v2.csv"
)
DEFAULT_CANONICAL_DIR = "data/processed/stage_3_2B_canonical"
DEFAULT_BOUNDARY_FIELD_DIR = "data/processed/stage_4A_boundary_fields_v1"

SEGMENTATION_COHORT_ROLES: Dict[str, str] = {
    "source_training": "train",
    "internal_validation": "validation",
    "native_target": "target_evaluation",
}
EXPECTED_PRIMARY_COUNTS: Dict[str, int] = {
    "source_training": 945,
    "internal_validation": 191,
    "native_target": 2187,
}
VALID_FIELD_STATUSES = frozenset({"generated", "skipped_valid", "regenerated"})

REQUIRED_FIELD_MANIFEST_COLUMNS = frozenset(
    {
        "record_id",
        "pair_id",
        "cohort",
        "phi_fp_filename",
        "phi_fn_filename",
        "phi_fp_content_hash",
        "phi_fn_content_hash",
        "source_mask_sha256_manifest",
        "source_mask_hash_verified",
        "field_version",
        "hash_convention",
        "tau",
        "max_cap",
        "generation_status",
    }
)


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _image_filename(cohort: str, pair_id: str) -> str:
    return f"{cohort}_{pair_id}_bilinear.png"


def _load_field_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    rows = _read_csv(path)
    if not rows:
        raise ValueError(f"Boundary-field manifest is empty: {path}")

    missing_columns = REQUIRED_FIELD_MANIFEST_COLUMNS - set(rows[0].keys())
    if missing_columns:
        raise ValueError(
            f"Boundary-field manifest {path} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    mapping: Dict[str, Dict[str, str]] = {}
    for row in rows:
        record_id = row["record_id"]
        if not record_id:
            raise ValueError(f"Boundary-field manifest contains an empty record_id in {path}")
        if record_id in mapping:
            raise ValueError(
                f"Duplicate boundary-field record_id {record_id!r} in {path}"
            )
        mapping[record_id] = row
    return mapping


def _safe_basename(filename: str, record_id: str) -> str:
    if not filename or Path(filename).name != filename:
        raise ValueError(
            f"Unsafe boundary-field filename {filename!r} for record {record_id!r}"
        )
    return filename


def _select_primary_records(
    manifest_rows: Sequence[Dict[str, str]],
    cohort: str,
) -> tuple[List[Dict[str, str]], int]:
    """
    Select exactly one primary record per ``orig_img_sha256`` cluster using the
    lowest alphanumeric Pair ID, cross-checking the manifest role flag.
    """
    cohort_rows = [row for row in manifest_rows if row["cohort"] == cohort]
    eligible_rows: List[Dict[str, str]] = []
    excluded_diagnostics = 0

    for row in cohort_rows:
        if row.get("is_diagnostic", "").strip().lower() == "true":
            excluded_diagnostics += 1
            continue
        eligible_rows.append(row)

    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in eligible_rows:
        image_sha = row.get("orig_img_sha256", "")
        if not image_sha:
            raise ValueError(
                f"Missing orig_img_sha256 for {cohort} pair {row.get('pair_id')}"
            )
        groups[image_sha].append(row)

    selected: List[Dict[str, str]] = []
    for image_sha, group in groups.items():
        lowest_pair_id = min(str(row["pair_id"]) for row in group)
        primaries = [
            row
            for row in group
            if row.get("is_primary_reference", "").strip().lower() == "true"
        ]
        if len(primaries) != 1:
            raise ValueError(
                f"Cohort {cohort} image {image_sha} has {len(primaries)} primary "
                "records; exactly one is required"
            )
        primary = primaries[0]
        if str(primary["pair_id"]) != lowest_pair_id:
            raise ValueError(
                f"Cohort {cohort} image {image_sha} primary Pair ID "
                f"{primary['pair_id']!r} does not match the lowest alphanumeric "
                f"Pair ID {lowest_pair_id!r}"
            )

        cluster_ids = {row.get("cluster_id", "") for row in group}
        if len(cluster_ids) != 1:
            raise ValueError(
                f"Cohort {cohort} image {image_sha} spans multiple cluster ids: "
                f"{sorted(cluster_ids)}"
            )
        selected.append(primary)

    selected.sort(key=lambda row: str(row["pair_id"]))
    return selected, excluded_diagnostics


class DFUCanonicalDataset(Dataset):
    """
    PyTorch Dataset for one approved segmentation cohort.

    Parameters
    ----------
    cohort:
        One of ``source_training``, ``internal_validation``, ``native_target``.
        Healthy controls and diagnostic cohorts are rejected.
    expected_count:
        Optional exact-count guard. The production factory passes the approved
        counts (945 / 191 / 2187) so silent population drift fails loudly.
    verify_field_hashes:
        If True (default), every loaded field must match the canonical content
        hash recorded by Task 3B.
    exclusion_registry:
        Optional, fully validated :class:`ExclusionRegistry`. When supplied,
        the approved excluded Pair IDs for this cohort are removed after primary
        selection. The registry is never used to mutate the manifest, and a
        cohort not covered by the registry is rejected.
    """

    def __init__(
        self,
        manifest_path: str | Path = DEFAULT_MANIFEST,
        cohort: str = "source_training",
        split_role: Optional[str] = None,
        project_root: str | Path = DEFAULT_PROJECT_ROOT,
        canonical_dir: str | Path = DEFAULT_CANONICAL_DIR,
        boundary_field_dir: str | Path = DEFAULT_BOUNDARY_FIELD_DIR,
        field_manifest_path: str | Path | None = None,
        expected_count: Optional[int] = None,
        verify_field_hashes: bool = True,
        exclusion_registry: Optional[ExclusionRegistry] = None,
        training_seed: int = 42,
    ):
        if cohort not in SEGMENTATION_COHORT_ROLES:
            raise ValueError(
                f"Cohort {cohort!r} is not a segmentation cohort. Approved "
                f"segmentation cohorts are {sorted(SEGMENTATION_COHORT_ROLES)}; "
                "healthy controls, mismatched targets, and diagnostics are "
                "intentionally excluded."
            )
        expected_role = SEGMENTATION_COHORT_ROLES[cohort]
        if split_role is not None and split_role != expected_role:
            raise ValueError(
                f"split_role {split_role!r} does not match cohort {cohort!r} "
                f"role {expected_role!r}"
            )

        self.project_root = Path(project_root)
        self.manifest_path = _resolve(self.project_root, manifest_path)
        self.canonical_dir = _resolve(self.project_root, canonical_dir)
        self.boundary_field_dir = _resolve(self.project_root, boundary_field_dir)
        self.field_manifest_path = (
            _resolve(self.project_root, field_manifest_path)
            if field_manifest_path is not None
            else self.boundary_field_dir / "boundary_field_manifest.csv"
        )
        self.cohort = cohort
        self.split_role = expected_role
        self.verify_field_hashes = bool(verify_field_hashes)
        self.exclusion_registry = exclusion_registry
        self.training_seed = int(training_seed)
        self.current_epoch = 0

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Authoritative manifest not found: {self.manifest_path}")
        if not self.field_manifest_path.exists():
            raise FileNotFoundError(
                f"Boundary-field manifest not found: {self.field_manifest_path}"
            )

        manifest_rows = _read_csv(self.manifest_path)
        selected, self.excluded_diagnostic_count = _select_primary_records(
            manifest_rows, cohort
        )

        # Apply the approved D-S1-03 exclusions by exact composite identity.
        # ``for_cohort`` fails closed for any cohort the registry does not cover.
        self.excluded_pair_ids: frozenset[str] = frozenset()
        if exclusion_registry is not None:
            excluded_pair_ids = exclusion_registry.for_cohort(cohort)
            selected_pair_ids = {str(row["pair_id"]) for row in selected}
            missing = excluded_pair_ids - selected_pair_ids
            if missing:
                raise ValueError(
                    f"Exclusion registry lists records that are not selected "
                    f"primaries for cohort {cohort!r}: {sorted(missing)}"
                )
            self.excluded_pair_ids = frozenset(excluded_pair_ids)
            selected = [
                row for row in selected if str(row["pair_id"]) not in excluded_pair_ids
            ]

        if expected_count is not None and len(selected) != int(expected_count):
            raise ValueError(
                f"Cohort {cohort} has {len(selected)} eligible primary records; "
                f"expected exactly {int(expected_count)}"
            )

        field_manifest = _load_field_manifest(self.field_manifest_path)
        self.records: List[Dict[str, Any]] = []
        missing_artifacts: List[str] = []
        seen_record_ids: set[str] = set()

        for row in selected:
            record_id = f"{cohort}_{row['pair_id']}"
            if record_id in seen_record_ids:
                raise ValueError(f"Duplicate dataset record_id: {record_id!r}")
            seen_record_ids.add(record_id)

            if not row.get("proc_mask_filename"):
                raise ValueError(
                    f"Eligible record {record_id!r} has no canonical mask filename"
                )
            image_path = (
                self.canonical_dir / "images" / _image_filename(cohort, row["pair_id"])
            )
            mask_path = self.canonical_dir / "masks" / row["proc_mask_filename"]

            field_row = field_manifest.get(record_id)
            if field_row is None:
                raise ValueError(
                    f"No boundary-field record for {record_id!r}; regenerate Task 3B "
                    "fields or verify the authoritative manifest"
                )
            if field_row.get("pair_id") != row["pair_id"] or field_row.get("cohort") != cohort:
                raise ValueError(
                    f"Boundary-field identity mismatch for {record_id!r}: "
                    f"field record is {field_row.get('cohort')}/{field_row.get('pair_id')}"
                )
            if field_row.get("source_mask_sha256_manifest") != row.get("proc_mask_sha256"):
                raise ValueError(
                    f"Stale boundary-field record {record_id!r}: source mask hash "
                    f"{field_row.get('source_mask_sha256_manifest')!r} does not match "
                    f"authoritative manifest {row.get('proc_mask_sha256')!r}"
                )
            if field_row.get("source_mask_hash_verified") != "True":
                raise ValueError(
                    f"Boundary-field source mask was not verified for {record_id!r}"
                )
            if field_row.get("generation_status") not in VALID_FIELD_STATUSES:
                raise ValueError(
                    f"Boundary-field record {record_id!r} has unusable status "
                    f"{field_row.get('generation_status')!r}"
                )
            if field_row.get("field_version") != FIELD_VERSION:
                raise ValueError(
                    f"Boundary-field version mismatch for {record_id!r}: "
                    f"{field_row.get('field_version')!r} != {FIELD_VERSION!r}"
                )
            if field_row.get("hash_convention") != HASH_CONVENTION:
                raise ValueError(
                    f"Boundary-field hash convention mismatch for {record_id!r}"
                )
            if float(field_row.get("tau", "nan")) != TAU:
                raise ValueError(f"Boundary-field tau mismatch for {record_id!r}")
            if float(field_row.get("max_cap", "nan")) != MAX_CAP:
                raise ValueError(f"Boundary-field max_cap mismatch for {record_id!r}")

            fp_name = _safe_basename(field_row["phi_fp_filename"], record_id)
            fn_name = _safe_basename(field_row["phi_fn_filename"], record_id)
            phi_fp_path = self.boundary_field_dir / "phi_fp" / fp_name
            phi_fn_path = self.boundary_field_dir / "phi_fn" / fn_name

            for label, path in (
                ("image", image_path),
                ("mask", mask_path),
                ("phi_fp", phi_fp_path),
                ("phi_fn", phi_fn_path),
            ):
                if not path.exists():
                    missing_artifacts.append(f"{record_id}: {label} -> {path}")

            record = dict(row)
            record.update(
                {
                    "record_id": record_id,
                    "split_role": expected_role,
                    "annotation_role": "primary",
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "phi_fp_path": phi_fp_path,
                    "phi_fn_path": phi_fn_path,
                    "phi_fp_content_hash": field_row["phi_fp_content_hash"],
                    "phi_fn_content_hash": field_row["phi_fn_content_hash"],
                }
            )
            self.records.append(record)

        if missing_artifacts:
            display = "\n".join(missing_artifacts[:20])
            remainder = len(missing_artifacts) - 20
            suffix = f"\n... and {remainder} more" if remainder > 0 else ""
            raise FileNotFoundError(
                f"Missing Stage 4 artifacts for cohort {cohort}:\n{display}{suffix}"
            )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        """Set the current source-training epoch for deterministic augmentation."""
        self.current_epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        record_id = record["record_id"]

        # Load image, mask, and precomputed boundary fields
        image = self._load_image(record)
        mask = self._load_mask(record, image.shape[-2:])
        phi_fp = self._load_field(
            record["phi_fp_path"],
            record["phi_fp_content_hash"],
            record_id,
            "phi_fp",
            mask.shape[-2:],
        )
        phi_fn = self._load_field(
            record["phi_fn_path"],
            record["phi_fn_content_hash"],
            record_id,
            "phi_fn",
            mask.shape[-2:],
        )

        # D3-j: Apply augmentation and recompute phi_fp/phi_fn only for
        # the source-training cohort. Validation and target cohorts remain
        # unaugmented, preserving the original precomputed boundary fields.
        if self.cohort == "source_training":
            # Convert tensors to numpy for the augmentation pipeline
            image_np = image.permute(1, 2, 0).numpy() if image.ndim == 3 else image.numpy()
            mask_np = mask.numpy().squeeze() if mask.ndim > 2 else mask.numpy()

            # D3 seed policy (approved amendment): derive from the configured
            # training seed, stable record ID, and current epoch. Do not use the
            # shuffled positional index as the augmentation seed.
            augmentation_seed = derive_augmentation_seed(
                self.training_seed, str(record["record_id"]), int(self.current_epoch)
            )
            augmented_image, augmented_mask = augment_sample(
                image_np, mask_np, seed=augmentation_seed,
            )

            # Convert augmented image back to torch tensor [3, H, W]
            if augmented_image.ndim == 3:
                image = torch.from_numpy(augmented_image).permute(2, 0, 1).contiguous()
            else:
                image = torch.from_numpy(augmented_image)

            # Convert augmented mask back to torch tensor [1, H, W]
            mask = torch.from_numpy(augmented_mask).unsqueeze(0).contiguous()

            # D3-j: Recompute phi_fp and phi_fn from the geometrically augmented mask
            mask_np_for_fields = augmented_mask.astype(np.int32)
            if mask_np_for_fields.ndim == 3:
                mask_np_for_fields = mask_np_for_fields.squeeze()
            new_phi_fp, new_phi_fn = compute_tolerance_boundary_fields(
                mask_np_for_fields, tau=TAU, max_cap=MAX_CAP
            )
            phi_fp = torch.from_numpy(new_phi_fp).unsqueeze(0).contiguous()
            phi_fn = torch.from_numpy(new_phi_fn).unsqueeze(0).contiguous()
        else:
            # For validation/target cohorts, ensure tensors are contiguous
            image = image.contiguous()
            mask = mask.contiguous()
            phi_fp = phi_fp.contiguous()
            phi_fn = phi_fn.contiguous()

        return {
            "image": image,
            "mask": mask,
            "phi_fp": phi_fp,
            "phi_fn": phi_fn,
            "pair_id": record["pair_id"],
            "record_id": record_id,
            "cohort": record["cohort"],
            "cluster_id": record["cluster_id"],
            "unique_content_id": record["unique_content_id"],
            "orig_img_sha256": record["orig_img_sha256"],
            "annotation_role": record["annotation_role"],
            "is_primary_reference": True,
            "is_diagnostic": False,
            "split_role": record["split_role"],
            "mask_status": record["mask_status"],
            "proc_mask_filename": record["proc_mask_filename"],
            "source_mask_path": record["mask_path"].as_posix(),
            "source_mask_sha256": record["proc_mask_sha256"],
            "image_path": record["image_path"].as_posix(),
            "phi_fp_content_hash": record["phi_fp_content_hash"],
            "phi_fn_content_hash": record["phi_fn_content_hash"],
            "orig_fg_count": int(record["orig_fg_count"]),
            "proc_fg_count": int(record["proc_fg_count"]),
        }

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def _load_image(self, record: Mapping[str, Any]) -> torch.Tensor:
        path = record["image_path"]
        try:
            with Image.open(path) as image:
                rgb = np.array(image.convert("RGB"), dtype=np.float32, copy=True) / 255.0
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load RGB image for {record['record_id']} from {path}: {exc}"
            ) from exc
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(
                f"Image {path} has shape {rgb.shape}; expected (H, W, 3)"
            )
        if not np.isfinite(rgb).all():
            raise ValueError(f"Image {path} contains non-finite values")
        return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()

    def _load_mask(
        self, record: Mapping[str, Any], expected_hw: Sequence[int]
    ) -> torch.Tensor:
        path = record["mask_path"]
        try:
            with Image.open(path) as image:
                raw = np.array(image.convert("L"), dtype=np.uint8, copy=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load mask for {record['record_id']} from {path}: {exc}"
            ) from exc
        if raw.ndim != 2:
            raise ValueError(f"Mask {path} did not decode to a 2-D array")
        if tuple(raw.shape) != tuple(expected_hw):
            raise ValueError(
                f"Mask {path} shape {raw.shape} does not match image shape "
                f"{tuple(expected_hw)}"
            )
        unique_values = set(np.unique(raw).tolist())
        if not unique_values.issubset({0, 255}):
            raise ValueError(
                f"Mask {path} is not binary 0/255; found values "
                f"{sorted(unique_values)[:10]}"
            )
        binary = (raw > 0).astype(np.float32)
        return torch.from_numpy(binary).unsqueeze(0)

    def _load_field(
        self,
        path: Path,
        expected_hash: str,
        record_id: str,
        field_name: str,
        expected_hw: Sequence[int],
    ) -> torch.Tensor:
        try:
            array = np.load(path, allow_pickle=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load {field_name} for {record_id} from {path}: {exc}"
            ) from exc
        if not isinstance(array, np.ndarray):
            raise ValueError(f"{field_name} file {path} does not contain an ndarray")
        if array.dtype != np.dtype(np.float32):
            raise ValueError(
                f"{field_name} for {record_id} has dtype {array.dtype}; expected float32"
            )
        if array.ndim != 2:
            raise ValueError(
                f"{field_name} for {record_id} has rank {array.ndim}; expected 2"
            )
        if tuple(array.shape) != tuple(expected_hw):
            raise ValueError(
                f"{field_name} for {record_id} has shape {array.shape}; expected "
                f"{tuple(expected_hw)}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{field_name} for {record_id} contains non-finite values")
        if self.verify_field_hashes:
            actual_hash = canonical_field_hash(array)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Canonical {field_name} hash mismatch for {record_id} "
                    f"({path}): expected {expected_hash!r}, got {actual_hash!r}"
                )
        contiguous = np.ascontiguousarray(array, dtype=np.float32)
        return torch.from_numpy(contiguous).unsqueeze(0)


def build_stage_4_dataloaders(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    project_root: str | Path = DEFAULT_PROJECT_ROOT,
    canonical_dir: str | Path = DEFAULT_CANONICAL_DIR,
    boundary_field_dir: str | Path = DEFAULT_BOUNDARY_FIELD_DIR,
    seed: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Build the three approved segmentation datasets and real PyTorch DataLoaders.

    No controls, mismatched-target, or diagnostic loader is created. Training
    is shuffled with a deterministic epoch-aware sampler; validation and native
    target evaluation are deterministic and unshuffled.

    Training loops must call ``loaders["train_sampler"].set_epoch(epoch)`` at
    the start of each epoch to obtain the documented reproducible permutation.
    """
    config = Stage4DataLoaderConfig()
    train_seed = config.train_seed if seed is None else int(seed)
    batch_size = config.batch_size if batch_size is None else int(batch_size)
    num_workers = config.num_workers if num_workers is None else int(num_workers)
    pin_memory = config.pin_memory if pin_memory is None else bool(pin_memory)

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    train_ds = DFUCanonicalDataset(
        manifest_path=manifest_path,
        cohort="source_training",
        split_role="train",
        project_root=project_root,
        canonical_dir=canonical_dir,
        boundary_field_dir=boundary_field_dir,
        expected_count=EXPECTED_PRIMARY_COUNTS["source_training"],
        training_seed=int(train_seed),
    )
    val_ds = DFUCanonicalDataset(
        manifest_path=manifest_path,
        cohort="internal_validation",
        split_role="validation",
        project_root=project_root,
        canonical_dir=canonical_dir,
        boundary_field_dir=boundary_field_dir,
        expected_count=EXPECTED_PRIMARY_COUNTS["internal_validation"],
    )
    target_ds = DFUCanonicalDataset(
        manifest_path=manifest_path,
        cohort="native_target",
        split_role="target_evaluation",
        project_root=project_root,
        canonical_dir=canonical_dir,
        boundary_field_dir=boundary_field_dir,
        expected_count=EXPECTED_PRIMARY_COUNTS["native_target"],
    )

    train_sampler = EpochShuffleSampler(len(train_ds), seed=train_seed)
    persistent = bool(config.persistent_workers and num_workers > 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    target_loader = DataLoader(
        target_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )

    return {
        "train_dataset": train_ds,
        "train_sampler": train_sampler,
        "train_loader": train_loader,
        "val_dataset": val_ds,
        "val_loader": val_loader,
        "target_native_dataset": target_ds,
        "target_native_loader": target_loader,
        "config": {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "train_seed": train_seed,
        },
    }