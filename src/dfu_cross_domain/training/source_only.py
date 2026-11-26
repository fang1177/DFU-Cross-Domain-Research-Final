"""Source-only DataLoader construction for baseline training.

Why this module exists
---------------------
``src.data.stage_4_dataloaders.build_stage_4_dataloaders`` always constructs
the ``native_target`` dataset and loader. Reusing that factory inside a
training process would import target artifacts into the training pipeline,
which the project forbids. This module provides a separate, source-only
construction path that builds exactly two datasets and two loaders:

* ``source_training`` -> training loader (epoch-shuffled)
* ``internal_validation`` -> validation loader (sequential)

The native external target cohort is never referenced, imported, constructed,
or iterated here.

Analysis population (D-S1-03)
-----------------------------
The frozen corrected-v2 cohorts contain 945 training and 191 validation
primaries. The approved S1 Option E analysis population excludes exactly 18
source-training and 5 internal-validation label-uncertain primaries.

The exclusion is supplied by the separate, hash-verified registry in
``src.data.exclusion_registry``. This factory always:

1. loads and validates the registry and its sidecar;
2. verifies the pinned manifest and review CSV hashes;
3. verifies all 23 composite keys, their cohorts, and their image/mask hashes;
4. removes exactly those Pair IDs from primary selection;
5. requires the resulting sizes to be exactly 927 / 186.

There is no expected-count override and no loader-factory override. Callers
cannot construct the frozen 945 / 191 or native-target populations through
this path.
"""

from __future__ import annotations

from typing import Any, Dict

from torch.utils.data import DataLoader

from src.data.exclusion_registry import (
    APPROVED_ANALYSIS_POPULATION as _APPROVED_ANALYSIS_POPULATION,
    DEFAULT_REGISTRY_PATH,
    DEFAULT_REVIEW_PATH,
    DEFAULT_SIDECAR_PATH,
    DEFAULT_SOURCE_IMAGE_ROOT,
    FROZEN_PRIMARY_COUNTS as _FROZEN_PRIMARY_COUNTS,
    PINNED_REGISTRY_CSV_SHA256,
    PINNED_SIDECAR_SHA256,
    ExclusionRegistry,
    ExclusionRegistryError,
    load_exclusion_registry,
)
from src.data.samplers import EpochShuffleSampler
from src.data.stage_4_dataloaders import (
    DEFAULT_BOUNDARY_FIELD_DIR,
    DEFAULT_CANONICAL_DIR,
    DEFAULT_MANIFEST,
    DEFAULT_PROJECT_ROOT,
    DFUCanonicalDataset,
)

#: The only cohorts this module is permitted to construct.
SOURCE_ONLY_COHORTS = ("source_training", "internal_validation")

#: Frozen cohort sizes from the corrected-v2 manifest / Stage 1E audit.
FROZEN_PRIMARY_COUNTS: Dict[str, int] = dict(_FROZEN_PRIMARY_COUNTS)

#: Approved primary analysis population (S1 Option E, 2026-11-14).
APPROVED_PRIMARY_POPULATION: Dict[str, int] = {
    "training": _APPROVED_ANALYSIS_POPULATION["source_training"],
    "validation": _APPROVED_ANALYSIS_POPULATION["internal_validation"],
}

_EXPECTED_ANALYSIS_COUNTS: Dict[str, int] = dict(_APPROVED_ANALYSIS_POPULATION)


class PendingAnalysisPopulationError(RuntimeError):
    """Raised when loaders are not the approved 927 / 186 analysis population."""


class SourceOnlyLoaderError(RuntimeError):
    """Raised when source-only loaders are not safely constructed."""


def assert_approved_primary_population(
    train_size: int, validation_size: int
) -> None:
    """Fail closed unless the loaders match the approved 927 / 186 population."""
    expected_train = APPROVED_PRIMARY_POPULATION["training"]
    expected_validation = APPROVED_PRIMARY_POPULATION["validation"]

    if int(train_size) == expected_train and int(validation_size) == expected_validation:
        return

    raise PendingAnalysisPopulationError(
        "Approved primary analysis population is not available: got "
        f"{int(train_size)} training / {int(validation_size)} validation, "
        f"expected {expected_train} / {expected_validation}. The frozen "
        "cohort definitions are 945 / 191 and contain 18 source-training and "
        "5 internal-validation label-uncertain primaries. These must be "
        "excluded through the hash-verified D-S1-03 exclusion registry; no "
        "other exclusion mechanism and no count override is permitted."
    )


def validate_source_only_loaders(loaders: Dict[str, Any]) -> ExclusionRegistry:
    """Re-verify that source-only loaders carry the approved registry policy.

    This is a defence-in-depth check for the training entry point. It confirms
    that both datasets were built with a validated exclusion registry, that
    their cohort roles are the two permitted source cohorts, that the excluded
    identities match the registry exactly, and that no excluded record leaked
    into a dataset.
    """
    required_keys = {"train_dataset", "train_loader", "val_dataset", "val_loader"}
    missing = required_keys - set(loaders)
    if missing:
        raise SourceOnlyLoaderError(
            f"Source-only loader output is missing keys: {sorted(missing)}"
        )
    forbidden_keys = {
        key for key in loaders if key.startswith("target") or "native" in key
    }
    if forbidden_keys:
        raise SourceOnlyLoaderError(
            f"Source-only loaders must not expose target artifacts: "
            f"{sorted(forbidden_keys)}"
        )

    train_ds = loaders["train_dataset"]
    val_ds = loaders["val_dataset"]
    expected_roles = (("source_training", train_ds), ("internal_validation", val_ds))

    registry: ExclusionRegistry | None = None
    for cohort, dataset in expected_roles:
        if getattr(dataset, "cohort", None) != cohort:
            raise SourceOnlyLoaderError(
                f"Dataset role mismatch: expected cohort {cohort!r}, got "
                f"{getattr(dataset, 'cohort', None)!r}"
            )
        dataset_registry = getattr(dataset, "exclusion_registry", None)
        if not isinstance(dataset_registry, ExclusionRegistry):
            raise SourceOnlyLoaderError(
                f"Dataset {cohort!r} was not built with a validated D-S1-03 "
                "exclusion registry"
            )
        if registry is None:
            registry = dataset_registry
        elif (
            dataset_registry.registry_csv_sha256 != registry.registry_csv_sha256
            or dataset_registry.manifest_sha256 != registry.manifest_sha256
        ):
            raise SourceOnlyLoaderError(
                "Training and validation datasets use different registries"
            )

        expected_exclusions = registry.for_cohort(cohort)
        dataset_exclusions = frozenset(
            str(value) for value in getattr(dataset, "excluded_pair_ids", ())
        )
        if dataset_exclusions != expected_exclusions:
            raise SourceOnlyLoaderError(
                f"Dataset {cohort!r} exclusion set does not match the registry: "
                f"missing={sorted(expected_exclusions - dataset_exclusions)}, "
                f"extra={sorted(dataset_exclusions - expected_exclusions)}"
            )
        records = getattr(dataset, "records", None)
        if records is not None:
            leaked = {
                str(record.get("pair_id"))
                for record in records
                if str(record.get("pair_id")) in expected_exclusions
            }
            if leaked:
                raise SourceOnlyLoaderError(
                    f"Excluded records leaked into {cohort!r}: {sorted(leaked)}"
                )

    assert registry is not None
    assert_approved_primary_population(len(train_ds), len(val_ds))
    return registry


def build_source_only_dataloaders(
    *,
    manifest_path: str = DEFAULT_MANIFEST,
    project_root: str = DEFAULT_PROJECT_ROOT,
    canonical_dir: str = DEFAULT_CANONICAL_DIR,
    boundary_field_dir: str = DEFAULT_BOUNDARY_FIELD_DIR,
    registry_path: str = DEFAULT_REGISTRY_PATH,
    registry_sidecar_path: str = DEFAULT_SIDECAR_PATH,
    source_image_root: str = DEFAULT_SOURCE_IMAGE_ROOT,
    review_path: str = DEFAULT_REVIEW_PATH,
    expected_registry_csv_sha256: str | None = PINNED_REGISTRY_CSV_SHA256,
    expected_sidecar_sha256: str | None = PINNED_SIDECAR_SHA256,
    verify_registry_files: bool = True,
    seed: int = 42,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Dict[str, Any]:
    """Build validated 927 / 186 source-only datasets and loaders.

    The registry is validated before either dataset is constructed. There is
    deliberately no ``train_expected_count`` / ``val_expected_count`` override:
    the approved population is enforced unconditionally.
    """
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(num_workers) < 0:
        raise ValueError("num_workers must be non-negative")

    registry = load_exclusion_registry(
        registry_path=registry_path,
        sidecar_path=registry_sidecar_path,
        manifest_path=manifest_path,
        project_root=project_root,
        canonical_dir=canonical_dir,
        source_image_root=source_image_root,
        review_path=review_path,
        expected_sidecar_sha256=expected_sidecar_sha256,
        expected_registry_csv_sha256=expected_registry_csv_sha256,
        verify_files=bool(verify_registry_files),
    )

    if registry.expected_population != _EXPECTED_ANALYSIS_COUNTS:
        raise ExclusionRegistryError(
            "Registry sidecar does not describe the approved 927 / 186 "
            f"population: {registry.expected_population}"
        )

    # Exactly two datasets, both source cohorts. No target construction.
    train_ds = DFUCanonicalDataset(
        manifest_path=manifest_path,
        cohort="source_training",
        split_role="train",
        project_root=project_root,
        canonical_dir=canonical_dir,
        boundary_field_dir=boundary_field_dir,
        expected_count=_EXPECTED_ANALYSIS_COUNTS["source_training"],
        exclusion_registry=registry,
        training_seed=int(seed),
    )
    val_ds = DFUCanonicalDataset(
        manifest_path=manifest_path,
        cohort="internal_validation",
        split_role="validation",
        project_root=project_root,
        canonical_dir=canonical_dir,
        boundary_field_dir=boundary_field_dir,
        expected_count=_EXPECTED_ANALYSIS_COUNTS["internal_validation"],
        exclusion_registry=registry,
    )

    constructed = {train_ds.cohort, val_ds.cohort}
    if constructed != set(SOURCE_ONLY_COHORTS):
        raise RuntimeError(
            "Source-only factory constructed unexpected cohorts: "
            f"{sorted(constructed)}"
        )

    train_sampler = EpochShuffleSampler(len(train_ds), seed=int(seed))
    persistent = bool(num_workers > 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        sampler=train_sampler,
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=persistent,
    )

    loaders = {
        "train_dataset": train_ds,
        "train_sampler": train_sampler,
        "train_loader": train_loader,
        "val_dataset": val_ds,
        "val_loader": val_loader,
        "exclusion_registry": registry,
        "config": {
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "pin_memory": bool(pin_memory),
            "seed": int(seed),
            "source_only": True,
            "cohorts": list(SOURCE_ONLY_COHORTS),
            "registry_version": registry.registry_version,
            "registry_csv_sha256": registry.registry_csv_sha256,
            "registry_sidecar_sha256": registry.sidecar_sha256,
            "manifest_sha256": registry.manifest_sha256,
            "expected_training_population": _EXPECTED_ANALYSIS_COUNTS["source_training"],
            "expected_validation_population": _EXPECTED_ANALYSIS_COUNTS[
                "internal_validation"
            ],
        },
    }
    validate_source_only_loaders(loaders)
    return loaders