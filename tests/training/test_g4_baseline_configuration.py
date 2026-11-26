"""Focused tests for the approved G4 baseline configuration changes.

These tests never construct research data, run a training epoch, execute an
optimizer step on real data, or touch the external target cohort. Scheduler
tests only advance a one-parameter optimizer with no gradients.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest
import torch

from src.data.config import Stage4DataLoaderConfig
from src.dfu_cross_domain.training.checkpoint import (
    CRITICAL_CONFIG_FIELDS,
    INTENTIONALLY_RESUMABLE_CONFIG_FIELDS,
    reproducibility_info,
)
from src.dfu_cross_domain.training.config import TrainingConfig
from src.dfu_cross_domain.training.determinism import (
    CUBLAS_WORKSPACE_CONFIG,
    configure_determinism,
)
from src.dfu_cross_domain.training.run_layout import (
    APPROVED_BASELINE_SEEDS,
    assert_distinct_seed_outputs,
    build_baseline_seed_configs,
    config_for_seed,
    seed_run_dirs,
)
from src.dfu_cross_domain.training.scheduler import build_warmup_cosine_scheduler
from src.dfu_cross_domain.training.trainer import (
    pin_memory_for_device,
    run_source_only_training,
)

# ---------------------------------------------------------------------------
# G4-D1 / D2: exact warm-up and cosine schedule
# ---------------------------------------------------------------------------
def _scheduler(total_epochs: int = 100):
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW(
        [parameter],
        lr=5.0e-4,
        betas=(0.9, 0.999),
        weight_decay=1.0e-4,
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=5,
        total_epochs=total_epochs,
        base_learning_rate=5.0e-4,
        min_learning_rate=1.0e-6,
    )
    return optimizer, scheduler


def _learning_rate_sequence(total_epochs: int = 100):
    optimizer, scheduler = _scheduler(total_epochs)
    observed = []
    for _ in range(total_epochs):
        observed.append(float(optimizer.param_groups[0]["lr"]))
        optimizer.step()
        scheduler.step()
    return observed


def test_scheduler_warmup_uses_discrete_one_fifth_start():
    """G4-D1: epoch 0 is 1e-4, not exactly zero."""
    observed = _learning_rate_sequence()
    assert observed[0] == pytest.approx(1.0e-4, rel=0, abs=1e-15)
    assert observed[1] == pytest.approx(2.0e-4, rel=0, abs=1e-15)
    assert observed[2] == pytest.approx(3.0e-4, rel=0, abs=1e-15)
    assert observed[3] == pytest.approx(4.0e-4, rel=0, abs=1e-15)
    assert observed[4] == pytest.approx(5.0e-4, rel=0, abs=1e-15)


def test_scheduler_peak_transition_and_decay_start():
    observed = _learning_rate_sequence()
    # The final warm-up epoch and the first decay epoch both carry the peak.
    assert observed[4] == pytest.approx(5.0e-4, rel=0, abs=1e-15)
    assert observed[5] == pytest.approx(5.0e-4, rel=0, abs=1e-15)
    # Decay is monotonic after the peak.
    for earlier, later in zip(observed[5:], observed[6:]):
        assert later <= earlier


def test_scheduler_final_epoch_is_exactly_one_micromolar():
    """G4-D2: the last executed epoch (99) must hit 1e-6 exactly."""
    observed = _learning_rate_sequence()
    assert observed[99] == pytest.approx(1.0e-6, rel=0, abs=1e-15)
    # It is the global minimum of the schedule.
    assert observed[99] == pytest.approx(min(observed), rel=0, abs=1e-15)


def test_scheduler_state_resume_preserves_trajectory():
    optimizer_a, scheduler_a = _scheduler()
    for _ in range(20):
        optimizer_a.step()
        scheduler_a.step()

    state = scheduler_a.state_dict()

    optimizer_b, scheduler_b = _scheduler()
    scheduler_b.load_state_dict(state)

    optimizer_a.step()
    scheduler_a.step()
    optimizer_b.step()
    scheduler_b.step()

    assert optimizer_a.param_groups[0]["lr"] == pytest.approx(
        optimizer_b.param_groups[0]["lr"], rel=0, abs=1e-15
    )


# ---------------------------------------------------------------------------
# G4-D4: strict determinism
# ---------------------------------------------------------------------------
def test_configure_determinism_sets_approved_flags(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)

    configure_determinism()

    assert torch.are_deterministic_algorithms_enabled() is True
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == CUBLAS_WORKSPACE_CONFIG


def test_configure_determinism_rejects_conflicting_cublas_value(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        configure_determinism()


def test_reproducibility_info_records_determinism_settings(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
    configure_determinism()

    info = reproducibility_info(42, torch.device("cpu"))

    assert info["deterministic_algorithms"] is True
    assert info["cudnn_deterministic"] is True
    assert info["cudnn_benchmark"] is False
    assert info["cublas_workspace_config"] == CUBLAS_WORKSPACE_CONFIG


# ---------------------------------------------------------------------------
# G4-D6 / D7 / D9 / D10: loader configuration
# ---------------------------------------------------------------------------
def test_loader_defaults_match_approved_baseline():
    config = TrainingConfig()
    stage4 = Stage4DataLoaderConfig()

    assert config.batch_size == 8
    assert stage4.batch_size == 8
    assert config.num_workers == 0
    assert config.grad_clip_norm is None
    assert config.min_delta == 0.0
    assert config.epochs == 100
    assert config.early_stopping_patience == 15


def test_pin_memory_follows_resolved_device():
    assert pin_memory_for_device(torch.device("cpu")) is False
    assert pin_memory_for_device(torch.device("cuda")) is True


def _capture_loader_kwargs(monkeypatch, requested_kwargs):
    from src.dfu_cross_domain.training import trainer as trainer_module

    captured = {}

    def _stop_before_construction(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop-before-loader-construction")

    monkeypatch.setattr(
        trainer_module, "build_source_only_dataloaders", _stop_before_construction
    )

    model = torch.nn.Conv2d(3, 1, kernel_size=1)
    with pytest.raises(RuntimeError, match="stop-before-loader-construction"):
        run_source_only_training(
            model=model,
            config=TrainingConfig(device="cpu"),
            **requested_kwargs,
        )
    return captured


def test_cpu_run_disables_pin_memory(monkeypatch):
    captured = _capture_loader_kwargs(monkeypatch, requested_kwargs={})
    assert captured["pin_memory"] is False
    assert captured["batch_size"] == 8
    assert captured["num_workers"] == 0


def test_explicit_pin_memory_override_is_honoured(monkeypatch):
    captured = _capture_loader_kwargs(monkeypatch, requested_kwargs={"pin_memory": True})
    assert captured["pin_memory"] is True


# ---------------------------------------------------------------------------
# G4-D8: frozen resume fields
# ---------------------------------------------------------------------------
def test_epochs_and_patience_are_frozen_resume_fields():
    assert "epochs" in CRITICAL_CONFIG_FIELDS
    assert "early_stopping_patience" in CRITICAL_CONFIG_FIELDS
    assert "epochs" not in INTENTIONALLY_RESUMABLE_CONFIG_FIELDS
    assert (
        "early_stopping_patience" not in INTENTIONALLY_RESUMABLE_CONFIG_FIELDS
    )


# ---------------------------------------------------------------------------
# Multi-seed isolation
# ---------------------------------------------------------------------------
def test_approved_seed_directories_are_distinct():
    base = TrainingConfig()
    configs = build_baseline_seed_configs(base)

    assert tuple(config.seed for config in configs) == APPROVED_BASELINE_SEEDS
    checkpoint_dirs = {config.checkpoint_dir for config in configs}
    log_dirs = {config.log_dir for config in configs}
    assert len(checkpoint_dirs) == 3
    assert len(log_dirs) == 3
    assert all(f"seed_{seed}" in dirs for seed, dirs in zip(
        APPROVED_BASELINE_SEEDS,
        (config.checkpoint_dir for config in configs),
    ))

    assert_distinct_seed_outputs(configs)


def test_duplicate_seed_is_rejected():
    with pytest.raises(ValueError, match="duplicate baseline seeds"):
        build_baseline_seed_configs(TrainingConfig(), seeds=(42, 42))


def test_seed_directory_collision_is_rejected():
    first = config_for_seed(TrainingConfig(), 42)
    second = replace(first, seed=1337)

    with pytest.raises(ValueError, match="collision"):
        assert_distinct_seed_outputs([first, second])


def test_seed_run_dirs_use_isolated_subdirectories():
    dirs = seed_run_dirs(2024)
    assert dirs["root"].replace("\\", "/").endswith("seed_2024")
    assert dirs["checkpoint_dir"].replace("\\", "/").endswith(
        "seed_2024/checkpoints"
    )
    assert dirs["log_dir"].replace("\\", "/").endswith("seed_2024/logs")