"""Checkpoint persistence for the baseline training pipeline.

Checkpoints are written atomically (temporary file + ``os.replace``) and carry
everything needed to resume: model, optimizer and scheduler state, the epoch
counter, the best observed validation metric and best epoch, the current
early-stopping counter, the training configuration, the history so far, RNG
state, and reproducibility metadata.

No research checkpoint is produced by this module unless a caller explicitly
runs a training loop. Tests use temporary directories.
"""

from __future__ import annotations

import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

from .config import TrainingConfig
from .seeding import capture_rng_state, restore_rng_state

CHECKPOINT_FORMAT_VERSION = 1

#: Settings that define the optimisation trajectory. A resumed run must match
#: these exactly or it is no longer an uninterrupted continuation.
CRITICAL_CONFIG_FIELDS = (
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "beta1",
    "beta2",
    "warmup_epochs",
    "min_learning_rate",
    "seed",
    "grad_clip_norm",
    "monitor",
    "min_delta",
    "early_stopping_patience",
    "max_train_batches",
    "max_val_batches",
)

#: Settings that may intentionally change on resume. G4-D8 froze ``epochs``
#: and ``early_stopping_patience`` for the baseline: changing either after a
#: run has started would rewrite the pre-registered LR trajectory or the model
#: selection rule, so both are validated as critical fields. Runtime/output
#: settings may still differ without invalidating the model trajectory.
INTENTIONALLY_RESUMABLE_CONFIG_FIELDS = (
    "device",
    "num_workers",
    "checkpoint_dir",
    "log_dir",
)


class CheckpointConfigError(ValueError):
    """Raised when a checkpoint cannot be resumed with the current config."""


def _config_value(config: Any, field: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(field)
    return getattr(config, field, None)


def _values_equal(saved: Any, current: Any) -> bool:
    if saved is None or current is None:
        return saved is None and current is None
    if isinstance(saved, bool) or isinstance(current, bool):
        return bool(saved) == bool(current)
    if isinstance(saved, (int, float)) and isinstance(current, (int, float)):
        return math.isclose(
            float(saved), float(current), rel_tol=0.0, abs_tol=1e-12
        )
    return saved == current


def validate_checkpoint_config(
    saved_config: Optional[Mapping[str, Any]],
    current_config: Any,
) -> None:
    """Reject a resume that changes optimisation-critical settings.

    ``saved_config`` is the config dict stored in the checkpoint. Missing
    entries are skipped for backward compatibility with older checkpoints.
    """
    if not saved_config:
        return
    if not isinstance(saved_config, Mapping):
        raise CheckpointConfigError(
            "Checkpoint config is not a mapping; cannot validate compatibility"
        )

    mismatches = []
    for field in CRITICAL_CONFIG_FIELDS:
        if field not in saved_config:
            continue
        saved = saved_config[field]
        current = _config_value(current_config, field)
        if not _values_equal(saved, current):
            mismatches.append(
                f"{field}: checkpoint={saved!r}, current={current!r}"
            )

    if mismatches:
        raise CheckpointConfigError(
            "Checkpoint configuration is incompatible with the current run. "
            "The following critical settings differ and would invalidate an "
            "uninterrupted run:\n  - "
            + "\n  - ".join(mismatches)
            + "\nIntentionally resumable settings are: "
            + ", ".join(INTENTIONALLY_RESUMABLE_CONFIG_FIELDS)
            + "."
        )


def _derive_resume_state(
    history: Optional[list],
) -> tuple[int, int]:
    """Best-effort recovery of selection state from pre-fix checkpoints."""
    best_epoch = -1
    epochs_without_improvement = 0
    best_dice = float("-inf")

    for record in history or []:
        if not isinstance(record, Mapping):
            continue

        improved = record.get("improved")
        if isinstance(improved, bool):
            if improved:
                best_epoch = int(record.get("epoch", best_epoch))
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            continue

        dice = record.get("validation_dice")
        if dice is None:
            continue
        dice = float(dice)
        if dice > best_dice:
            best_dice = dice
            best_epoch = int(record.get("epoch", best_epoch))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

    return best_epoch, epochs_without_improvement


def resolve_selection_state(payload: Mapping[str, Any]) -> tuple[int, int]:
    """Return ``(best_epoch, epochs_without_improvement)`` from a checkpoint.

    New checkpoints store both values explicitly. Checkpoints written before
    those keys existed fall back to reconstructing them from the recorded
    history, so resumption never silently restarts early-stopping patience.
    """
    derived_best_epoch, derived_epochs_without_improvement = _derive_resume_state(
        payload.get("history")
    )

    stored_best_epoch = payload.get("best_epoch")
    best_epoch = (
        derived_best_epoch
        if stored_best_epoch is None
        else int(stored_best_epoch)
    )

    stored_epochs_without_improvement = payload.get("epochs_without_improvement")
    epochs_without_improvement = (
        derived_epochs_without_improvement
        if stored_epochs_without_improvement is None
        else int(stored_epochs_without_improvement)
    )

    return best_epoch, max(0, epochs_without_improvement)


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), tmp_path)
    os.replace(tmp_path, path)


def reproducibility_info(seed: int, device: torch.device) -> Dict[str, Any]:
    """Collect environment information needed to describe a run."""
    return {
        "seed": int(seed),
        "device": str(device),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn_version": torch.backends.cudnn.version()
        if torch.backends.cudnn.is_available()
        else None,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    epoch: int,
    best_metric: float,
    config: TrainingConfig,
    history: Optional[list] = None,
    extra: Optional[Mapping[str, Any]] = None,
    device: Optional[torch.device] = None,
    best_epoch: int = -1,
    epochs_without_improvement: int = 0,
) -> Path:
    """Atomically write a resumable checkpoint and return its path.

    ``best_epoch`` and ``epochs_without_improvement`` are persisted so a
    resumed run keeps its best-checkpoint pointer and consumed early-stopping
    patience. Older readers that do not know these keys can still load the
    checkpoint because the format version is unchanged.
    """
    destination = Path(path)
    resolved_device = torch.device("cpu") if device is None else torch.device(device)

    payload: Dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "epochs_without_improvement": max(0, int(epochs_without_improvement)),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None
        if optimizer is None
        else optimizer.state_dict(),
        "scheduler_state_dict": None
        if scheduler is None
        else scheduler.state_dict(),
        "config": config.to_dict(),
        "history": list(history or []),
        "rng_state": capture_rng_state(),
        "reproducibility": reproducibility_info(config.seed, resolved_device),
    }
    if extra:
        payload["extra"] = dict(extra)

    _atomic_torch_save(payload, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
    strict: bool = True,
    expected_config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Load a checkpoint into the supplied objects and return its payload.

    Parameters
    ----------
    path:
        Checkpoint file produced by :func:`save_checkpoint`.
    model, optimizer, scheduler:
        Objects to restore. ``optimizer`` and ``scheduler`` are optional.
    map_location:
        Passed to :func:`torch.load`.
    restore_rng:
        Restore the captured RNG state so the resumed run continues the same
        stochastic stream.
    strict:
        Passed to ``model.load_state_dict``.
    expected_config:
        Optional current configuration. When supplied, optimisation-critical
        settings are compared with the checkpoint before any state is restored,
        so a mismatched resume fails before training continues.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(
        checkpoint_path, map_location=map_location, weights_only=False
    )
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} is not a dict payload")

    version = payload.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format_version {version!r}; expected "
            f"{CHECKPOINT_FORMAT_VERSION}"
        )

    required = {"model_state_dict", "epoch", "best_metric"}
    missing = required - set(payload)
    if missing:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is missing keys: {sorted(missing)}"
        )

    if expected_config is not None:
        validate_checkpoint_config(payload.get("config"), expected_config)

    model.load_state_dict(payload["model_state_dict"], strict=strict)

    optimizer_state = payload.get("optimizer_state_dict")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    scheduler_state = payload.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    if restore_rng and payload.get("rng_state") is not None:
        restore_rng_state(payload["rng_state"])

    return payload