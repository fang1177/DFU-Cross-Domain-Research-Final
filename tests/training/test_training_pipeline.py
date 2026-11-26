"""Focused synthetic-data tests for the source-only baseline training pipeline.

No research dataset, checkpoint, evaluation or external-target artifact is
used. Every fixture in this module is generated in memory or under
``tmp_path``.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.dfu_cross_domain.training.checkpoint import (
    CheckpointConfigError,
    load_checkpoint,
    resolve_selection_state,
    save_checkpoint,
)
from src.dfu_cross_domain.training.config import TrainingConfig
from src.dfu_cross_domain.training.engine import train_one_epoch, validate_epoch
from src.dfu_cross_domain.training.scheduler import build_warmup_cosine_scheduler
from src.dfu_cross_domain.training.seeding import seed_everything
from src.dfu_cross_domain.training.trainer import (
    build_optimizer,
    fit,
    run_source_only_training,
)
from src.dfu_cross_domain.training import source_only as source_only_module
from src.dfu_cross_domain.training import trainer as trainer_module
from src.dfu_cross_domain.training.source_only import (
    SourceOnlyLoaderError,
    build_source_only_dataloaders,
)
from src.models.mobile_dfu_net.losses import MobileDFUCompoundLoss

SIZE = 16


class SyntheticSegDataset(Dataset):
    """Cheap deterministic segmentation fixture with the production key set."""

    def __init__(self, n: int, size: int = SIZE, seed: int = 0):
        self.n = int(n)
        self.size = int(size)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int):
        generator = torch.Generator().manual_seed(self.seed * 100003 + int(index))
        image = torch.rand(3, self.size, self.size, generator=generator)
        mask = (torch.rand(1, self.size, self.size, generator=generator) > 0.7).float()
        zeros = torch.zeros(1, self.size, self.size)
        return {
            "image": image,
            "mask": mask,
            "phi_fp": zeros.clone(),
            "phi_fn": zeros.clone(),
            "pair_id": f"{int(index):04d}",
        }


class TinyNet(torch.nn.Module):
    """Minimal model implementing the MobileDFUNet forward contract."""

    def __init__(self):
        super().__init__()
        self.seg = torch.nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.boundary = torch.nn.Conv2d(3, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        return {
            "seg_logits": self.seg(x),
            "boundary_logits": self.boundary(x),
        }


class SpySampler:
    """Records ``set_epoch`` calls without driving a real DataLoader."""

    def __init__(self):
        self.epochs = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(int(epoch))

    def __iter__(self):
        return iter([])

    def __len__(self) -> int:
        return 0


class SpyEpochDataset(Dataset):
    """Synthetic dataset that records epoch updates for augmentation tests."""

    def __init__(self):
        self.epochs = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(int(epoch))

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        return {
            "image": torch.zeros(2, 3, SIZE, SIZE),
            "mask": torch.zeros(2, 1, SIZE, SIZE),
            "phi_fp": torch.zeros(2, 1, SIZE, SIZE),
            "phi_fn": torch.zeros(2, 1, SIZE, SIZE),
        }


def _loader(n: int = 4, batch_size: int = 2, seed: int = 0) -> DataLoader:
    return DataLoader(
        SyntheticSegDataset(n, seed=seed),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def _config(tmp_path: Path, **overrides) -> TrainingConfig:
    defaults = dict(
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
    defaults.update(overrides)
    return TrainingConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. A training step updates parameters
# ---------------------------------------------------------------------------
def test_training_step_updates_parameters():
    torch.manual_seed(0)
    model = TinyNet()
    loader = _loader(n=4, batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-2)

    before = [p.detach().clone() for p in model.parameters()]
    metrics = train_one_epoch(
        model, loader, MobileDFUCompoundLoss(), optimizer, "cpu"
    )
    after = [p.detach().clone() for p in model.parameters()]

    assert "total_loss" in metrics
    assert math.isfinite(metrics["total_loss"])
    assert any(not torch.equal(a, b) for a, b in zip(before, after))


# ---------------------------------------------------------------------------
# 2. Validation does not update parameters
# ---------------------------------------------------------------------------
def test_validation_does_not_update_parameters():
    torch.manual_seed(0)
    model = TinyNet()
    loader = _loader(n=4, batch_size=2)

    before = [p.detach().clone() for p in model.parameters()]
    metrics = validate_epoch(model, loader, MobileDFUCompoundLoss(), "cpu")
    after = [p.detach().clone() for p in model.parameters()]

    assert "dice" in metrics
    assert 0.0 <= metrics["dice"] <= 1.0
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    # Every parameter still has an autograd node with no accumulated gradient.
    assert all(p.grad is None for p in model.parameters())


# ---------------------------------------------------------------------------
# 3. Checkpoint save and resume
# ---------------------------------------------------------------------------
def test_checkpoint_save_and_resume(tmp_path: Path):
    torch.manual_seed(0)
    model = TinyNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    config = TrainingConfig(epochs=3, warmup_epochs=0, device="cpu")
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=0,
        total_epochs=3,
        base_learning_rate=config.learning_rate,
        min_learning_rate=config.min_learning_rate,
    )

    # Perform one real optimizer step so the optimizer state is populated.
    train_one_epoch(
        model, _loader(n=2, batch_size=2), MobileDFUCompoundLoss(), optimizer, "cpu"
    )

    path = save_checkpoint(
        tmp_path / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=2,
        best_metric=0.75,
        config=config,
        history=[{"epoch": 2, "validation_dice": 0.75}],
        device=torch.device("cpu"),
        best_epoch=1,
        epochs_without_improvement=2,
    )
    assert path.exists()

    restored = TinyNet()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1.0e-3)
    restored_scheduler = build_warmup_cosine_scheduler(
        restored_optimizer,
        warmup_epochs=0,
        total_epochs=3,
        base_learning_rate=config.learning_rate,
        min_learning_rate=config.min_learning_rate,
    )
    payload = load_checkpoint(
        path,
        model=restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        map_location="cpu",
    )

    for original, resumed in zip(model.state_dict().values(), restored.state_dict().values()):
        assert torch.equal(original, resumed)
    assert payload["epoch"] == 2
    assert payload["best_metric"] == pytest.approx(0.75)
    assert payload["best_epoch"] == 1
    assert payload["epochs_without_improvement"] == 2
    assert resolve_selection_state(payload) == (1, 2)
    assert payload["optimizer_state_dict"] is not None
    assert len(restored_optimizer.state_dict()["state"]) > 0
    assert "rng_state" in payload and "reproducibility" in payload


def test_fit_can_resume_from_last_checkpoint(tmp_path: Path):
    # G4-D8 freezes ``epochs`` for resume compatibility, so the checkpoint
    # must describe an unfinished run with the same total epoch count.
    config = _config(tmp_path, epochs=2, early_stopping_patience=0)
    checkpoint = _write_resume_checkpoint(
        tmp_path,
        config,
        epoch=0,
        best_metric=0.5,
        best_epoch=0,
        epochs_without_improvement=0,
    )

    resumed_model = TinyNet()
    summary = fit(
        model=resumed_model,
        train_loader=_loader(n=4),
        val_loader=_loader(n=4, seed=1),
        loss_fn=MobileDFUCompoundLoss(),
        optimizer=build_optimizer(resumed_model, config),
        config=config,
        resume_from=checkpoint,
    )
    # Epoch 0 is already in the checkpoint; epoch 1 is the only new epoch.
    assert summary["epochs_run"] == 1
    assert summary["last_epoch"] == 1
    assert len(summary["history"]) >= 2
    assert (tmp_path / "checkpoints" / "last.pt").exists()


# ---------------------------------------------------------------------------
# 4. Reproducible seed behaviour
# ---------------------------------------------------------------------------
def _draw(seed: int):
    seed_everything(seed)
    return (
        torch.rand(8),
        np.random.rand(8),
        [random.random() for _ in range(8)],
    )


def test_seed_is_reproducible_and_distinct():
    torch_a, numpy_a, python_a = _draw(7)
    torch_b, numpy_b, python_b = _draw(7)
    assert torch.equal(torch_a, torch_b)
    assert np.array_equal(numpy_a, numpy_b)
    assert python_a == python_b

    torch_c, numpy_c, python_c = _draw(8)
    assert not torch.equal(torch_a, torch_c)
    assert not np.array_equal(numpy_a, numpy_c)
    assert python_a != python_c


# ---------------------------------------------------------------------------
# 5. Sampler epoch handling
# ---------------------------------------------------------------------------
def test_fit_sets_sampler_epoch_each_epoch(tmp_path: Path):
    config = _config(tmp_path, epochs=3)
    model = TinyNet()
    spy = SpySampler()

    summary = fit(
        model=model,
        train_loader=_loader(n=4),
        val_loader=_loader(n=4, seed=1),
        loss_fn=MobileDFUCompoundLoss(),
        optimizer=build_optimizer(model, config),
        config=config,
        train_sampler=spy,
    )

    assert spy.epochs == [0, 1, 2]
    assert summary["epochs_run"] == 3
    assert (tmp_path / "checkpoints" / "best.pt").exists()
    assert (tmp_path / "checkpoints" / "last.pt").exists()
    assert (tmp_path / "logs" / "training_log.jsonl").exists()


def test_fit_sets_dataset_epoch_each_epoch(
    monkeypatch, tmp_path: Path
):
    config = _config(tmp_path, epochs=2, early_stopping_patience=0)
    model = TinyNet()
    spy_dataset = SpyEpochDataset()
    train_loader = DataLoader(spy_dataset, batch_size=2, shuffle=False)

    def _fake_train_one_epoch(*args, **kwargs):
        return {"total_loss": 0.5}

    def _fake_validate_epoch(*args, **kwargs):
        return {"total_loss": 0.5, "dice": 0.5}

    def _fake_save_checkpoint(*args, **kwargs):
        return None

    def _fake_append_jsonl(*args, **kwargs):
        return None

    monkeypatch.setattr(trainer_module, "train_one_epoch", _fake_train_one_epoch)
    monkeypatch.setattr(trainer_module, "validate_epoch", _fake_validate_epoch)
    monkeypatch.setattr(trainer_module, "save_checkpoint", _fake_save_checkpoint)
    monkeypatch.setattr(trainer_module, "_append_jsonl", _fake_append_jsonl)

    summary = fit(
        model=model,
        train_loader=train_loader,
        val_loader=_loader(n=2),
        loss_fn=MobileDFUCompoundLoss(),
        optimizer=build_optimizer(model, config),
        config=config,
        train_sampler=SpySampler(),
    )

    assert spy_dataset.epochs == [0, 1]
    assert summary["epochs_run"] == 2


def _install_fake_epoch_boundaries(monkeypatch) -> None:
    monkeypatch.setattr(
        trainer_module, "train_one_epoch", lambda *args, **kwargs: {"total_loss": 0.5}
    )
    monkeypatch.setattr(
        trainer_module, "validate_epoch", lambda *args, **kwargs: {"total_loss": 0.5, "dice": 0.5}
    )
    monkeypatch.setattr(trainer_module, "save_checkpoint", lambda *args, **kwargs: None)


def test_fit_log_records_contain_seed_and_effective_config(monkeypatch, tmp_path: Path):
    config = _config(tmp_path, epochs=2, early_stopping_patience=0)
    model = TinyNet()
    train_loader = DataLoader(SpyEpochDataset(), batch_size=2, shuffle=False)

    appended = []

    def _record_log(path, record):
        appended.append(record)

    monkeypatch.setattr(trainer_module, "_append_jsonl", _record_log)
    _install_fake_epoch_boundaries(monkeypatch)

    summary = fit(
        model=model,
        train_loader=train_loader,
        val_loader=_loader(n=2),
        loss_fn=MobileDFUCompoundLoss(),
        optimizer=build_optimizer(model, config),
        config=config,
        train_sampler=SpySampler(),
    )

    assert summary["epochs_run"] == 2
    assert len(appended) == 2

    expected_keys = {
        "epoch",
        "seed",
        "config",
        "learning_rate",
        "training",
        "validation",
        "validation_dice",
        "best_validation_dice",
        "improved",
        "duration_seconds",
    }
    for record in appended:
        assert set(record) >= expected_keys
        assert int(record["seed"]) == int(config.seed)
        assert int(record["config"]["seed"]) == int(config.seed)
        assert int(record["config"]["epochs"]) == int(config.epochs)
        assert int(record["config"]["batch_size"]) == int(config.batch_size)
        assert float(record["config"]["learning_rate"]) == float(config.learning_rate)

    # Existing metrics and fields must be preserved alongside the new metadata.
    for record in appended:
        assert "epoch" in record
        assert "learning_rate" in record
        assert "training" in record
        assert "validation" in record
        assert "validation_dice" in record
        assert "best_validation_dice" in record
        assert "improved" in record
        assert "duration_seconds" in record
        assert float(record["training"]["total_loss"]) == 0.5
        assert float(record["validation"]["dice"]) == 0.5


def test_fit_log_records_identify_distinct_seeds(monkeypatch, tmp_path: Path):
    records_by_seed: dict = {}

    def _record_log(path, record):
        records_by_seed.setdefault(int(record["seed"]), []).append(record)

    monkeypatch.setattr(trainer_module, "_append_jsonl", _record_log)
    _install_fake_epoch_boundaries(monkeypatch)

    for seed, root in ((42, tmp_path / "seed_42"), (1337, tmp_path / "seed_1337")):
        config = _config(
            root,
            epochs=1,
            early_stopping_patience=0,
            seed=seed,
        )
        model = TinyNet()
        summary = fit(
            model=model,
            train_loader=_loader(n=2),
            val_loader=_loader(n=2),
            loss_fn=MobileDFUCompoundLoss(),
            optimizer=build_optimizer(model, config),
            config=config,
            train_sampler=SpySampler(),
        )
        assert summary["epochs_run"] == 1
        assert len(records_by_seed[int(seed)]) == 1

    assert set(records_by_seed) == {42, 1337}
    for seed, records in records_by_seed.items():
        assert all(int(record["seed"]) == seed for record in records)
        assert all(int(record["config"]["seed"]) == seed for record in records)


# ---------------------------------------------------------------------------
# Strict JSONL serialization for per-epoch log records
# ---------------------------------------------------------------------------
def _strict_json_loads(text: str):
    def _reject(constant: str):
        raise ValueError(f"non-standard JSON constant: {constant}")

    return json.loads(text, parse_constant=_reject)


def test_append_jsonl_writes_strict_json_with_finite_values(tmp_path: Path):
    path = tmp_path / "logs" / "training_log.jsonl"
    record = {
        "epoch": 0,
        "seed": 42,
        "config": {
            "epochs": 100,
            "batch_size": 8,
            "learning_rate": 5e-4,
            "seed": 42,
        },
        "training": {"total_loss": 1.25},
        "validation": {"total_loss": 0.5, "dice": 0.25},
        "validation_dice": 0.25,
        "best_validation_dice": -1.0,
        "improved": True,
        "duration_seconds": 1.5,
    }
    trainer_module._append_jsonl(path, record)

    parsed = _strict_json_loads(path.read_text(encoding="utf-8").strip())
    assert parsed["seed"] == 42
    assert parsed["config"]["epochs"] == 100
    assert parsed["config"]["batch_size"] == 8
    assert float(parsed["config"]["learning_rate"]) == pytest.approx(5e-4)
    assert float(parsed["training"]["total_loss"]) == pytest.approx(1.25)
    assert float(parsed["validation"]["dice"]) == pytest.approx(0.25)
    assert float(parsed["validation_dice"]) == pytest.approx(0.25)
    assert float(parsed["best_validation_dice"]) == pytest.approx(-1.0)
    assert parsed["improved"] is True
    assert float(parsed["duration_seconds"]) == pytest.approx(1.5)


def test_append_jsonl_converts_non_finite_values_to_null(tmp_path: Path):
    path = tmp_path / "logs" / "training_log.jsonl"
    record = {
        "epoch": 0,
        "seed": 42,
        "training": {"total_loss": float("nan")},
        "validation": {"dice": float("inf")},
        "validation_dice": float("-inf"),
        "best_validation_dice": float("inf"),
        "nested": {
            "value": float("nan"),
            "list": [1.0, float("inf"), (float("nan"), 2.0)],
        },
    }

    trainer_module._append_jsonl(path, record)

    parsed = _strict_json_loads(path.read_text(encoding="utf-8").strip())
    assert parsed["training"]["total_loss"] is None
    assert parsed["validation"]["dice"] is None
    assert parsed["validation_dice"] is None
    assert parsed["best_validation_dice"] is None
    assert parsed["nested"]["value"] is None
    assert parsed["nested"]["list"] == [1.0, None, [None, 2.0]]


def test_append_jsonl_does_not_mutate_input_record(tmp_path: Path):
    path = tmp_path / "logs" / "training_log.jsonl"
    record = {
        "epoch": 0,
        "validation_dice": float("nan"),
        "best_validation_dice": float("inf"),
        "nested": {"value": float("nan")},
        "training": {"total_loss": 1.25},
    }

    trainer_module._append_jsonl(path, record)

    # The original record must be left untouched (internal sentinels preserved).
    assert math.isnan(record["validation_dice"])
    assert math.isinf(record["best_validation_dice"]) and record["best_validation_dice"] > 0
    assert math.isnan(record["nested"]["value"])
    assert math.isfinite(record["training"]["total_loss"])


# ---------------------------------------------------------------------------
# 6. Target isolation
# ---------------------------------------------------------------------------
def test_source_only_factory_never_constructs_target(monkeypatch, exclusion_population):
    requested = []

    class RecordingDataset:
        def __init__(self, *, cohort, expected_count=None, exclusion_registry=None, **kwargs):
            requested.append(cohort)
            self.cohort = cohort
            self.expected_count = expected_count
            self.exclusion_registry = exclusion_registry
            self.excluded_pair_ids = exclusion_registry.for_cohort(cohort)
            self.records = [{"pair_id": "not-excluded"}]

        def __len__(self) -> int:
            return 927 if self.cohort == "source_training" else 186

        def __getitem__(self, index: int):
            raise AssertionError("Synthetic factory must not iterate the dataset")

    import src.data.stage_4_dataloaders as stage4

    def _forbidden_combined_factory(*args, **kwargs):
        raise AssertionError(
            "The combined Stage 4 factory must never be used by training"
        )

    monkeypatch.setattr(stage4, "build_stage_4_dataloaders", _forbidden_combined_factory)
    monkeypatch.setattr(source_only_module, "DFUCanonicalDataset", RecordingDataset)

    loaders = build_source_only_dataloaders(
        project_root=str(exclusion_population.root),
        manifest_path=str(exclusion_population.manifest),
        canonical_dir=str(exclusion_population.canonical),
        boundary_field_dir=str(exclusion_population.fields),
        registry_path=str(exclusion_population.registry),
        registry_sidecar_path=str(exclusion_population.sidecar),
        source_image_root=str(exclusion_population.image_root),
        review_path=str(exclusion_population.review),
        expected_registry_csv_sha256=exclusion_population.registry_sha256,
        expected_sidecar_sha256=exclusion_population.sidecar_sha256,
        batch_size=1,
        num_workers=0,
    )

    assert requested == ["source_training", "internal_validation"]
    assert "native_target" not in requested
    assert "target_native_loader" not in loaders
    assert "target_native_dataset" not in loaders


# ---------------------------------------------------------------------------
# 7. The entry point cannot bypass registry validation
# ---------------------------------------------------------------------------
def test_entry_point_rejects_custom_loader_factory(tmp_path: Path):
    config = _config(tmp_path, epochs=1)
    model = TinyNet()

    def frozen_population_factory(**kwargs):
        return {
            "train_dataset": SyntheticSegDataset(945),
            "train_loader": _loader(n=2),
            "val_dataset": SyntheticSegDataset(191, seed=1),
            "val_loader": _loader(n=2, seed=1),
        }

    before = [p.detach().clone() for p in model.parameters()]
    with pytest.raises(SourceOnlyLoaderError, match="loader_factory"):
        run_source_only_training(
            model=model,
            config=config,
            loader_factory=frozen_population_factory,
        )
    after = [p.detach().clone() for p in model.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(before, after))


# ---------------------------------------------------------------------------
# 8. Separate training and validation loaders
# ---------------------------------------------------------------------------
def test_source_only_factory_builds_separate_loaders(exclusion_population):
    loaders = build_source_only_dataloaders(
        project_root=str(exclusion_population.root),
        manifest_path=str(exclusion_population.manifest),
        canonical_dir=str(exclusion_population.canonical),
        boundary_field_dir=str(exclusion_population.fields),
        registry_path=str(exclusion_population.registry),
        registry_sidecar_path=str(exclusion_population.sidecar),
        source_image_root=str(exclusion_population.image_root),
        review_path=str(exclusion_population.review),
        expected_registry_csv_sha256=exclusion_population.registry_sha256,
        expected_sidecar_sha256=exclusion_population.sidecar_sha256,
        batch_size=1,
        num_workers=0,
    )

    assert loaders["train_loader"] is not loaders["val_loader"]
    assert loaders["train_dataset"] is not loaders["val_dataset"]
    assert loaders["train_dataset"].cohort == "source_training"
    assert loaders["val_dataset"].cohort == "internal_validation"
    assert isinstance(loaders["val_loader"].sampler, torch.utils.data.SequentialSampler)
    assert loaders["train_sampler"].num_samples == 927
    assert loaders["val_loader"].dataset is loaders["val_dataset"]


# ---------------------------------------------------------------------------
# 10. Warm-up + cosine schedule shape
# ---------------------------------------------------------------------------
def test_warmup_cosine_schedule_shape():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=2,
        total_epochs=4,
        base_learning_rate=1.0,
        min_learning_rate=0.1,
    )

    observed = []
    for _ in range(4):
        observed.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    # Warm-up increases, then cosine decay decreases monotonically.
    assert observed[0] == pytest.approx(0.5)
    assert observed[1] == pytest.approx(1.0)
    assert observed[2] > observed[3]
    assert observed[-1] >= 0.1


# ---------------------------------------------------------------------------
# 11. Resumption preserves selection state and early-stopping progress
# ---------------------------------------------------------------------------
def _write_resume_checkpoint(
    tmp_path: Path,
    config: TrainingConfig,
    *,
    epoch: int,
    best_metric: float,
    best_epoch: int,
    epochs_without_improvement: int,
    history=None,
):
    model = TinyNet()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.learning_rate)
    )
    history = (
        list(history)
        if history is not None
        else [{"epoch": epoch, "validation_dice": best_metric, "improved": False}]
    )
    return save_checkpoint(
        tmp_path / "checkpoints" / "resume.pt",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=epoch,
        best_metric=best_metric,
        config=config,
        history=history,
        device=torch.device("cpu"),
        best_epoch=best_epoch,
        epochs_without_improvement=epochs_without_improvement,
    )


def _resume(
    tmp_path: Path,
    config: TrainingConfig,
    checkpoint: Path,
):
    model = TinyNet()
    return fit(
        model=model,
        train_loader=_loader(n=4),
        val_loader=_loader(n=4, seed=1),
        loss_fn=MobileDFUCompoundLoss(),
        optimizer=build_optimizer(model, config),
        config=config,
        resume_from=checkpoint,
    )


def test_resume_preserves_best_epoch_several_epochs_earlier(tmp_path: Path):
    # ``min_delta`` larger than any possible Dice makes every new epoch
    # non-improving, so the restored best epoch must remain visible.
    config = _config(
        tmp_path,
        epochs=8,
        early_stopping_patience=10,
        min_delta=1.0e9,
    )
    checkpoint = _write_resume_checkpoint(
        tmp_path,
        config,
        epoch=5,
        best_metric=0.9,
        best_epoch=1,
        epochs_without_improvement=4,
    )

    summary = _resume(tmp_path, config, checkpoint)

    assert summary["epochs_run"] == 2
    assert summary["last_epoch"] == 7
    assert summary["best_epoch"] == 1
    assert summary["best_validation_dice"] == pytest.approx(0.9)
    assert summary["stopped_early"] is False


def test_resume_does_not_reset_partially_consumed_patience(tmp_path: Path):
    config = _config(
        tmp_path,
        epochs=20,
        early_stopping_patience=5,
        min_delta=1.0e9,
    )
    checkpoint = _write_resume_checkpoint(
        tmp_path,
        config,
        epoch=4,
        best_metric=0.8,
        best_epoch=2,
        epochs_without_improvement=3,
    )

    summary = _resume(tmp_path, config, checkpoint)

    # 3 consumed epochs + 2 new non-improving epochs reaches patience 5.
    # Resetting patience to zero would instead have run 15 more epochs.
    assert summary["epochs_run"] == 2
    assert summary["last_epoch"] == 6
    assert summary["stopped_early"] is True
    assert summary["best_epoch"] == 2


def test_resume_with_exhausted_patience_stops_immediately(tmp_path: Path):
    config = _config(
        tmp_path,
        epochs=20,
        early_stopping_patience=5,
        min_delta=1.0e9,
    )
    checkpoint = _write_resume_checkpoint(
        tmp_path,
        config,
        epoch=9,
        best_metric=0.7,
        best_epoch=3,
        epochs_without_improvement=5,
        history=[
            {"epoch": 8, "validation_dice": 0.6, "improved": False},
            {"epoch": 9, "validation_dice": 0.7, "improved": False},
        ],
    )

    summary = _resume(tmp_path, config, checkpoint)

    assert summary["epochs_run"] == 0
    assert summary["last_epoch"] == 9
    assert summary["stopped_early"] is True
    assert summary["best_epoch"] == 3
    assert not (tmp_path / "logs" / "training_log.jsonl").exists()


def test_resume_rejects_incompatible_critical_configuration(tmp_path: Path):
    saved_config = _config(tmp_path, epochs=1)
    checkpoint = _write_resume_checkpoint(
        tmp_path,
        saved_config,
        epoch=0,
        best_metric=0.5,
        best_epoch=0,
        epochs_without_improvement=0,
    )

    incompatible = _config(tmp_path, epochs=2, learning_rate=1.0e-3)
    with pytest.raises(CheckpointConfigError, match="learning_rate"):
        _resume(tmp_path, incompatible, checkpoint)

    # Validation happens before training: no new checkpoint was written.
    assert not (tmp_path / "checkpoints" / "last.pt").exists()


@pytest.mark.parametrize(
    "field, changed_value",
    [("epochs", 6), ("early_stopping_patience", 5)],
)
def test_resume_rejects_frozen_epochs_and_patience(
    tmp_path: Path, field: str, changed_value: int
):
    """G4-D8: baseline resume must reject schedule/selection rewrites."""
    saved_config = _config(tmp_path, epochs=5, early_stopping_patience=4)
    checkpoint = _write_resume_checkpoint(
        tmp_path,
        saved_config,
        epoch=0,
        best_metric=0.5,
        best_epoch=0,
        epochs_without_improvement=0,
    )

    overrides = {"epochs": 5, "early_stopping_patience": 4}
    overrides[field] = changed_value
    changed_config = _config(tmp_path, **overrides)
    with pytest.raises(CheckpointConfigError, match=field):
        _resume(tmp_path, changed_config, checkpoint)

    assert not (tmp_path / "checkpoints" / "last.pt").exists()


def test_resume_accepts_intentional_changes_and_legacy_checkpoint(tmp_path: Path):
    saved_config = _config(
        tmp_path, epochs=5, early_stopping_patience=4, min_delta=1.0e9
    )
    model = TinyNet()
    history = [
        {"epoch": 0, "validation_dice": 0.5, "improved": True},
        {"epoch": 1, "validation_dice": 0.4, "improved": False},
        {"epoch": 2, "validation_dice": 0.3, "improved": False},
    ]
    legacy_path = tmp_path / "checkpoints" / "legacy.pt"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "epoch": 2,
            "best_metric": 0.5,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": None,
            "scheduler_state_dict": None,
            "config": saved_config.to_dict(),
            "history": history,
        },
        legacy_path,
    )

    # G4-D8 freezes ``epochs`` and ``early_stopping_patience``; runtime
    # settings such as ``num_workers`` may still change. The missing
    # selection-state keys are rebuilt from history.
    resume_config = _config(
        tmp_path,
        epochs=5,
        early_stopping_patience=4,
        min_delta=1.0e9,
        num_workers=2,
    )
    summary = _resume(tmp_path, resume_config, legacy_path)

    assert summary["epochs_run"] == 2
    assert summary["last_epoch"] == 4
    assert summary["stopped_early"] is True
    assert summary["best_epoch"] == 0

    payload = torch.load(legacy_path, map_location="cpu", weights_only=False)
    assert "best_epoch" not in payload
    assert resolve_selection_state(payload) == (0, 2)

def test_json_safe_converts_path_objects_to_strings(tmp_path: Path):
    """Path objects must be converted before strict JSON serialization."""
    record = {
        "checkpoint_dir": tmp_path,
        "nested": {"log_dir": tmp_path / "logs"},
    }

    result = trainer_module._json_safe(record)

    assert result["checkpoint_dir"] == str(tmp_path)
    assert result["nested"]["log_dir"] == str(tmp_path / "logs")
    json.dumps(result, allow_nan=False)
