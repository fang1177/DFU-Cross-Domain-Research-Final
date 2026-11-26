"""Focused source-only integration tests for D-S1-03 exclusions.

The session fixture is synthetic (945 / 191 manifest primaries, tiny images),
so these tests exercise the real dataset, registry and trainer code paths
without touching research data or the native target cohort.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from src.data.exclusion_registry import (
    APPROVED_EXCLUSION_IDS,
    ExclusionRegistryError,
    load_exclusion_registry,
)
from src.data.stage_4_dataloaders import DFUCanonicalDataset
from src.dfu_cross_domain.training.config import TrainingConfig
from src.dfu_cross_domain.training.source_only import (
    SourceOnlyLoaderError,
    build_source_only_dataloaders,
    validate_source_only_loaders,
)
from src.dfu_cross_domain.training.trainer import run_source_only_training


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seg = torch.nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.boundary = torch.nn.Conv2d(3, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        return {
            "seg_logits": self.seg(x),
            "boundary_logits": self.boundary(x),
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _factory_kwargs(fixture) -> dict:
    return dict(
        project_root=str(fixture.root),
        manifest_path=str(fixture.manifest),
        canonical_dir=str(fixture.canonical),
        boundary_field_dir=str(fixture.fields),
        registry_path=str(fixture.registry),
        registry_sidecar_path=str(fixture.sidecar),
        source_image_root=str(fixture.image_root),
        review_path=str(fixture.review),
        expected_registry_csv_sha256=fixture.registry_sha256,
        expected_sidecar_sha256=fixture.sidecar_sha256,
    )


def _config(tmp_path: Path, **overrides) -> TrainingConfig:
    values = dict(
        epochs=1,
        batch_size=2,
        warmup_epochs=0,
        early_stopping_patience=0,
        device="cpu",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
        max_train_batches=1,
        max_val_batches=1,
    )
    values.update(overrides)
    return TrainingConfig(**values)


def _load_registry(fixture):
    return load_exclusion_registry(
        registry_path=fixture.registry,
        sidecar_path=fixture.sidecar,
        manifest_path=fixture.manifest,
        project_root=fixture.root,
        canonical_dir=fixture.canonical,
        source_image_root=fixture.image_root,
        review_path=fixture.review,
        expected_sidecar_sha256=fixture.sidecar_sha256,
        expected_registry_csv_sha256=fixture.registry_sha256,
    )


# ---------------------------------------------------------------------------
# Population construction
# ---------------------------------------------------------------------------
def test_factory_builds_exact_927_186(exclusion_population):
    fixture = exclusion_population
    loaders = build_source_only_dataloaders(
        **_factory_kwargs(fixture), batch_size=2, num_workers=0
    )

    train_ds = loaders["train_dataset"]
    val_ds = loaders["val_dataset"]
    assert len(train_ds) == 927
    assert len(val_ds) == 186
    assert len(train_ds.records) == 927
    assert len(val_ds.records) == 186

    train_ids = {record["pair_id"] for record in train_ds.records}
    val_ids = {record["pair_id"] for record in val_ds.records}
    assert train_ids.isdisjoint(APPROVED_EXCLUSION_IDS["source_training"])
    assert val_ids.isdisjoint(APPROVED_EXCLUSION_IDS["internal_validation"])
    assert train_ds.excluded_pair_ids == APPROVED_EXCLUSION_IDS["source_training"]
    assert val_ds.excluded_pair_ids == APPROVED_EXCLUSION_IDS["internal_validation"]

    assert "target_native_loader" not in loaders
    assert "target_native_dataset" not in loaders
    assert loaders["config"]["expected_training_population"] == 927
    assert loaders["config"]["expected_validation_population"] == 186
    assert loaders["exclusion_registry"] is train_ds.exclusion_registry
    assert loaders["exclusion_registry"] is val_ds.exclusion_registry


def test_excluded_records_absent_from_both_cohorts(exclusion_population):
    fixture = exclusion_population
    loaders = build_source_only_dataloaders(**_factory_kwargs(fixture))

    excluded_ids = loaders["exclusion_registry"].excluded_record_ids
    train_ids = {
        f"{record['cohort']}_{record['pair_id']}"
        for record in loaders["train_dataset"].records
    }
    val_ids = {
        f"{record['cohort']}_{record['pair_id']}"
        for record in loaders["val_dataset"].records
    }
    assert train_ids.isdisjoint(excluded_ids)
    assert val_ids.isdisjoint(excluded_ids)


def test_factory_preserves_manifest(exclusion_population):
    fixture = exclusion_population
    before = _sha256(fixture.manifest)
    build_source_only_dataloaders(**_factory_kwargs(fixture))
    assert _sha256(fixture.manifest) == before == fixture.manifest_sha256


# ---------------------------------------------------------------------------
# Target isolation
# ---------------------------------------------------------------------------
def test_target_cohort_rejected_by_registry(exclusion_population):
    fixture = exclusion_population
    registry = _load_registry(fixture)
    with pytest.raises(ExclusionRegistryError, match="not covered"):
        registry.for_cohort("native_target")

    with pytest.raises(ExclusionRegistryError, match="not covered"):
        DFUCanonicalDataset(
            manifest_path=fixture.manifest,
            cohort="native_target",
            project_root=fixture.root,
            canonical_dir=fixture.canonical,
            boundary_field_dir=fixture.fields,
            exclusion_registry=registry,
        )


def test_factory_never_calls_combined_stage4_factory(
    exclusion_population, monkeypatch
):
    fixture = exclusion_population

    import src.data.stage_4_dataloaders as stage4

    def _forbidden(*args, **kwargs):
        raise AssertionError("Combined Stage 4 factory must not be used")

    monkeypatch.setattr(stage4, "build_stage_4_dataloaders", _forbidden)
    loaders = build_source_only_dataloaders(**_factory_kwargs(fixture))
    assert set(loaders) >= {"train_loader", "val_loader"}


# ---------------------------------------------------------------------------
# Entry-point bypass rejection
# ---------------------------------------------------------------------------
def test_custom_loader_factory_rejected(tmp_path, exclusion_population):
    fixture = exclusion_population

    def fake_factory(**kwargs):
        raise AssertionError("custom factory must never run")

    with pytest.raises(SourceOnlyLoaderError, match="loader_factory"):
        run_source_only_training(
            model=TinyNet(),
            config=_config(tmp_path),
            loader_factory=fake_factory,
            **_factory_kwargs(fixture),
        )


def test_disabled_population_check_rejected(tmp_path, exclusion_population):
    fixture = exclusion_population
    with pytest.raises(SourceOnlyLoaderError, match="require_approved_population"):
        run_source_only_training(
            model=TinyNet(),
            config=_config(tmp_path),
            require_approved_population=False,
            **_factory_kwargs(fixture),
        )


def test_expected_count_override_rejected(tmp_path, exclusion_population):
    fixture = exclusion_population
    with pytest.raises(SourceOnlyLoaderError, match="train_expected_count"):
        run_source_only_training(
            model=TinyNet(),
            config=_config(tmp_path),
            train_expected_count=945,
            **_factory_kwargs(fixture),
        )


def test_tampered_registry_blocks_before_training(tmp_path, exclusion_population):
    fixture = exclusion_population
    registry = tmp_path / "tampered_registry.csv"
    registry.write_bytes(fixture.registry.read_bytes() + b"\n")

    config = _config(tmp_path)
    with pytest.raises(ExclusionRegistryError, match="CSV SHA-256 mismatch"):
        run_source_only_training(
            model=TinyNet(),
            config=config,
            **{
                **_factory_kwargs(fixture),
                "registry_path": str(registry),
            },
        )
    assert not (tmp_path / "checkpoints" / "last.pt").exists()


def test_validate_loaders_rejects_dataset_without_registry():
    class FakeDataset:
        cohort = "source_training"

        def __len__(self):
            return 927

    with pytest.raises(SourceOnlyLoaderError, match="validated D-S1-03"):
        validate_source_only_loaders(
            {
                "train_dataset": FakeDataset(),
                "train_loader": object(),
                "val_dataset": FakeDataset(),
                "val_loader": object(),
            }
        )


# ---------------------------------------------------------------------------
# End-to-end synthetic training on the approved population
# ---------------------------------------------------------------------------
def test_entry_point_runs_on_927_186(tmp_path, exclusion_population):
    fixture = exclusion_population
    summary = run_source_only_training(
        model=TinyNet(),
        config=_config(tmp_path),
        **_factory_kwargs(fixture),
    )
    assert summary["epochs_run"] == 1
    assert (tmp_path / "checkpoints" / "last.pt").exists()