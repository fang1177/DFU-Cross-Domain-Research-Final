"""Focused tests for the G4 three-seed source-only baseline orchestration.

These tests exercise :func:`run_source_only_baseline` with every training
boundary mocked out. No real training, validation, data loading, checkpointing
or log writing may occur here.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

import src.dfu_cross_domain.training.baseline as baseline_module
from src.dfu_cross_domain.training.baseline import (
    BaselineOutputArtifactError,
    run_source_only_baseline,
)
from src.dfu_cross_domain.training.config import TrainingConfig


class _FakeMobileDFUNet:
    """Records construction calls without building real parameters."""

    instances: list = []

    def __init__(self, config: Any = None) -> None:
        self.config = config
        _FakeMobileDFUNet.instances.append(self)


class _FakeOptimizer:
    pass


class _FakeScheduler:
    pass


def _install_mocks(monkeypatch, calls, seed_calls):
    def fake_run_source_only_training(
        *,
        model: Any,
        config: Any,
        loss_fn: Any = None,
        optimizer: Any = None,
        scheduler: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        calls.append(
            {
                "model": model,
                "config": config,
                "loss_fn": loss_fn,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "kwargs": kwargs,
            }
        )
        return {"seed": int(config.seed), "epochs_run": 0}

    def fake_seed_everything(seed: int) -> None:
        seed_calls.append(int(seed))

    def fake_build_optimizer(model: Any, config: Any) -> Any:
        return _FakeOptimizer()

    def fake_build_scheduler(optimizer: Any, **kwargs: Any) -> Any:
        return _FakeScheduler()

    monkeypatch.setattr(
        baseline_module, "run_source_only_training", fake_run_source_only_training
    )
    monkeypatch.setattr(baseline_module, "seed_everything", fake_seed_everything)
    monkeypatch.setattr(baseline_module, "MobileDFUNet", _FakeMobileDFUNet)
    monkeypatch.setattr(baseline_module, "build_optimizer", fake_build_optimizer)
    monkeypatch.setattr(baseline_module, "build_warmup_cosine_scheduler", fake_build_scheduler)


@pytest.fixture(autouse=True)
def _reset_fake_model_instances():
    _FakeMobileDFUNet.instances = []
    yield
    _FakeMobileDFUNet.instances = []


def _isolated_base_dir(tmp_path):
    return str(tmp_path / "baseline")


def test_three_seeds_dispatch_with_isolated_configs_and_fresh_state(
    monkeypatch, tmp_path
):
    calls = []
    seed_calls = []
    _install_mocks(monkeypatch, calls, seed_calls)

    config = TrainingConfig(epochs=10, warmup_epochs=2, learning_rate=1.0e-3)
    summaries = run_source_only_baseline(
        config=config, base_dir=_isolated_base_dir(tmp_path)
    )

    # Exactly three dispatches for the approved seeds, in approved order.
    assert len(calls) == 3
    assert [call["config"].seed for call in calls] == [42, 1337, 2024]

    models = [call["model"] for call in calls]
    assert len(models) == 3
    assert len(set(id(model) for model in models)) == 3
    assert len(_FakeMobileDFUNet.instances) == 3

    optimizers = [call["optimizer"] for call in calls]
    assert all(optimizer is not None for optimizer in optimizers)
    assert len(set(id(optimizer) for optimizer in optimizers)) == 3

    schedulers = [call["scheduler"] for call in calls]
    assert all(scheduler is not None for scheduler in schedulers)
    assert len(set(id(scheduler) for scheduler in schedulers)) == 3

    for call in calls:
        assert call["kwargs"] == {}

    # Isolated per-seed output paths.
    for call, seed in zip(calls, (42, 1337, 2024)):
        assert call["config"].seed == seed
        assert call["config"].checkpoint_dir.replace("\\", "/").endswith(
            f"seed_{seed}/checkpoints"
        )
        assert call["config"].log_dir.replace("\\", "/").endswith(f"seed_{seed}/logs")

    # Approved configuration values are preserved on each isolated config.
    for call in calls:
        assert call["config"].epochs == 10
        assert call["config"].warmup_epochs == 2
        assert call["config"].learning_rate == pytest.approx(1.0e-3)
        assert call["config"].monitor == "val_dice"
        assert call["config"].batch_size == 8


def test_fresh_seed_is_set_before_each_model_construction(
    monkeypatch, tmp_path
):
    calls = []
    seed_calls = []
    _install_mocks(monkeypatch, calls, seed_calls)

    run_source_only_baseline(
        config=TrainingConfig(), base_dir=_isolated_base_dir(tmp_path)
    )

    assert seed_calls == [42, 1337, 2024]
    # Each seed gets its own freshly constructed model instance.
    assert len(_FakeMobileDFUNet.instances) == 3
    assert len({id(model) for model in _FakeMobileDFUNet.instances}) == 3


def test_prior_checkpoint_artifact_blocks_dispatch(monkeypatch, tmp_path):
    calls = []
    seed_calls = []
    _install_mocks(monkeypatch, calls, seed_calls)

    checkpoint_dir = (
        tmp_path / "baseline" / "seed_42" / "checkpoints"
    )
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "best.pt").write_text("prior-artifact")

    with pytest.raises(BaselineOutputArtifactError):
        run_source_only_baseline(
            config=TrainingConfig(), base_dir=_isolated_base_dir(tmp_path)
        )

    # No training dispatch may happen once prior artifacts are detected.
    assert calls == []


def test_prior_log_artifact_blocks_dispatch(monkeypatch, tmp_path):
    calls = []
    seed_calls = []
    _install_mocks(monkeypatch, calls, seed_calls)

    log_dir = tmp_path / "baseline" / "seed_2024" / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "training_log.jsonl").write_text("{}\n")

    with pytest.raises(BaselineOutputArtifactError):
        run_source_only_baseline(
            config=TrainingConfig(), base_dir=_isolated_base_dir(tmp_path)
        )

    assert calls == []


def test_empty_seed_directories_do_not_block_dispatch(monkeypatch, tmp_path):
    calls = []
    seed_calls = []
    _install_mocks(monkeypatch, calls, seed_calls)

    # Directories that exist but contain no artifacts must not block.
    (tmp_path / "baseline" / "seed_42" / "checkpoints").mkdir(parents=True)
    (tmp_path / "baseline" / "seed_1337" / "logs").mkdir(parents=True)

    summaries = run_source_only_baseline(
        config=TrainingConfig(), base_dir=_isolated_base_dir(tmp_path)
    )

    assert len(calls) == 3
    assert len(summaries) == 3


def test_duplicate_seeds_are_rejected_without_dispatch(monkeypatch, tmp_path):
    calls = []
    seed_calls = []
    _install_mocks(monkeypatch, calls, seed_calls)

    with pytest.raises(ValueError, match="duplicate baseline seeds"):
        run_source_only_baseline(
            config=TrainingConfig(),
            seeds=(42, 1337, 2024, 42),
            base_dir=_isolated_base_dir(tmp_path),
        )

    assert calls == []
    assert seed_calls == []


def test_orchestration_never_enables_resume(monkeypatch, tmp_path):
    calls = []
    seed_calls = []
    _install_mocks(monkeypatch, calls, seed_calls)

    run_source_only_baseline(
        config=TrainingConfig(), base_dir=_isolated_base_dir(tmp_path)
    )

    assert len(calls) == 3
    for call in calls:
        assert "resume_from" not in call["kwargs"]
        assert call["optimizer"] is not None
        assert call["scheduler"] is not None


def test_relative_base_dir_resolves_against_project_root(monkeypatch, tmp_path):
    calls = []
    seed_calls = []
    _install_mocks(monkeypatch, calls, seed_calls)
    monkeypatch.setattr(
        baseline_module, "DEFAULT_PROJECT_ROOT", str(tmp_path / "project_root")
    )

    run_source_only_baseline(
        config=TrainingConfig(), base_dir="runs/baseline_source_only"
    )

    expected_base = (tmp_path / "project_root" / "runs" / "baseline_source_only").resolve()
    assert len(calls) == 3
    for call, seed in zip(calls, (42, 1337, 2024)):
        assert call["config"].checkpoint_dir == str(
            expected_base / f"seed_{seed}" / "checkpoints"
        )
        assert call["config"].log_dir == str(
            expected_base / f"seed_{seed}" / "logs"
        )


def test_absolute_custom_base_dir_is_preserved(monkeypatch, tmp_path):
    calls = []
    seed_calls = []
    _install_mocks(monkeypatch, calls, seed_calls)

    run_source_only_baseline(
        config=TrainingConfig(), base_dir=str(tmp_path / "custom_base")
    )

    assert len(calls) == 3
    for call, seed in zip(calls, (42, 1337, 2024)):
        assert call["config"].checkpoint_dir == str(
            tmp_path / "custom_base" / f"seed_{seed}" / "checkpoints"
        )
        assert call["config"].log_dir == str(
            tmp_path / "custom_base" / f"seed_{seed}" / "logs"
        )


def test_unexpected_file_in_seed_root_blocks_dispatch(monkeypatch, tmp_path):
    calls = []
    seed_calls = []
    _install_mocks(monkeypatch, calls, seed_calls)

    unexpected = tmp_path / "baseline" / "seed_1337" / "unexpected_note.txt"
    unexpected.parent.mkdir(parents=True)
    unexpected.write_text("unexpected")

    with pytest.raises(BaselineOutputArtifactError):
        run_source_only_baseline(
            config=TrainingConfig(), base_dir=_isolated_base_dir(tmp_path)
        )

    assert calls == []


def test_multiple_conflicting_seed_dirs_are_all_reported(monkeypatch, tmp_path):
    calls = []
    seed_calls = []
    _install_mocks(monkeypatch, calls, seed_calls)

    (tmp_path / "baseline" / "seed_42" / "checkpoints").mkdir(parents=True)
    (tmp_path / "baseline" / "seed_42" / "checkpoints" / "best.pt").write_text("prior")
    (tmp_path / "baseline" / "seed_2024" / "logs").mkdir(parents=True)
    (tmp_path / "baseline" / "seed_2024" / "logs" / "training_log.jsonl").write_text("{}\n")

    with pytest.raises(BaselineOutputArtifactError) as excinfo:
        run_source_only_baseline(
            config=TrainingConfig(), base_dir=_isolated_base_dir(tmp_path)
        )

    message = str(excinfo.value)
    assert "seed 42" in message
    assert "seed 2024" in message
    assert calls == []


def test_baseline_output_artifact_error_importable_from_package_namespace():
    from src.dfu_cross_domain.training import (
        BaselineOutputArtifactError as PackageBaselineOutputArtifactError,
    )

    assert PackageBaselineOutputArtifactError is BaselineOutputArtifactError
