"""Training orchestration for the minimal source-only baseline.

``fit`` is responsible for the epoch loop, sampler epoch updates, scheduler
stepping, logging, best/last checkpointing and early stopping. It is agnostic
to where the loaders came from.

``run_source_only_training`` is the real entry point. It builds loaders through
the source-only factory (never the combined Stage 4 factory), which validates
the hash-verified D-S1-03 exclusion registry and enforces the approved
927 / 186 analysis population. There is no custom-factory or population-check
override.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch

from src.models.mobile_dfu_net.losses import MobileDFUCompoundLoss

from .checkpoint import (
    load_checkpoint,
    reproducibility_info,
    resolve_selection_state,
    save_checkpoint,
)
from .config import TrainingConfig
from .determinism import configure_determinism
from .engine import train_one_epoch, validate_epoch
from .scheduler import build_warmup_cosine_scheduler
from .seeding import seed_everything
from .source_only import (
    SourceOnlyLoaderError,
    assert_approved_primary_population,
    build_source_only_dataloaders,
    validate_source_only_loaders,
)


def resolve_device(device: str) -> torch.device:
    """Resolve ``"auto"`` to CUDA when available, otherwise CPU."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def pin_memory_for_device(device: torch.device) -> bool:
    """Return the approved G4-D7 pinned-memory setting for ``device``.

    Pinned (page-locked) host memory only helps when a CUDA device is used;
    CPU-only execution keeps it disabled.
    """
    return torch.device(device).type == "cuda"


def build_optimizer(
    model: torch.nn.Module, config: TrainingConfig
) -> torch.optim.Optimizer:
    """Build the approved AdamW optimizer."""
    config.validate()
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        betas=(float(config.beta1), float(config.beta2)),
        weight_decay=float(config.weight_decay),
    )


def _json_safe(value: Any) -> Any:
    """Return a JSON-safe copy of ``value`` with non-finite floats mapped to ``None``.

    Recursively converts nested dictionaries, lists and tuples. The input value
    is never mutated. Non-finite floating-point numbers (``NaN``, ``+inf``,
    ``-inf``) become ``None`` so that strict JSON serialization can proceed.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_record = _json_safe(record)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(safe_record, sort_keys=True, allow_nan=False) + "\n")


def _is_improvement(current: float, best: float, min_delta: float) -> bool:
    return float(current) > float(best) + float(min_delta)


def _fit_summary(
    *,
    history_records: list,
    epochs_run: int,
    best_epoch: int,
    best_metric: float,
    stopped_early: bool,
    checkpoint_path: Path,
    log_file: Path,
    resolved_device: torch.device,
    config: TrainingConfig,
) -> Dict[str, Any]:
    return {
        "history": history_records,
        "epochs_run": epochs_run,
        "last_epoch": int(history_records[-1]["epoch"]) if history_records else -1,
        "best_epoch": best_epoch,
        "best_validation_dice": float(best_metric),
        "stopped_early": stopped_early,
        "checkpoint_dir": str(checkpoint_path),
        "log_file": str(log_file),
        "device": str(resolved_device),
        "reproducibility": reproducibility_info(config.seed, resolved_device),
    }


def fit(
    *,
    model: torch.nn.Module,
    train_loader: Any,
    val_loader: Any,
    loss_fn: Any,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    scheduler: Optional[Any] = None,
    train_sampler: Optional[Any] = None,
    device: Optional[str | torch.device] = None,
    checkpoint_dir: Optional[str | Path] = None,
    log_dir: Optional[str | Path] = None,
    start_epoch: int = 0,
    best_metric: Optional[float] = None,
    history: Optional[Iterable[Dict[str, Any]]] = None,
    resume_from: Optional[str | Path] = None,
    use_amp: bool = False,
) -> Dict[str, Any]:
    """Run the baseline training loop and return a run summary.

    Best-checkpoint criterion (explicit): the epoch with the highest mean hard
    validation Dice at probability threshold 0.5. Ties within ``min_delta`` do
    not replace the current best. Early stopping counts consecutive epochs
    without an improvement of at least ``min_delta``.
    """
    config.validate()

    # G4-D4: strict determinism is applied before any device work. This is
    # idempotent and also covers resume paths, which do not reseed the RNGs.
    configure_determinism()

    resolved_device = torch.device(
        device if device is not None else resolve_device(config.device)
    )
    model.to(resolved_device)

    scaler = None
    if use_amp:
        scaler = torch.amp.GradScaler("cuda") if resolved_device.type == "cuda" else None

    checkpoint_path = Path(checkpoint_dir or config.checkpoint_dir)
    log_path = Path(log_dir or config.log_dir)
    history_records = list(history or [])

    if best_metric is None:
        best_metric = float("-inf")
    best_epoch = -1
    epochs_without_improvement = 0

    resumed = resume_from is not None
    if resumed:
        payload = load_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location="cpu",
            restore_rng=True,
            expected_config=config,
        )
        start_epoch = int(payload["epoch"]) + 1
        best_metric = float(payload["best_metric"])
        if payload.get("history"):
            history_records = list(payload["history"])
        # Preserve the best-checkpoint pointer and consumed early-stopping
        # patience instead of restarting them at zero.
        best_epoch, epochs_without_improvement = resolve_selection_state(payload)
    else:
        seed_everything(int(config.seed))

    log_file = log_path / "training_log.jsonl"
    stopped_early = False
    epochs_run = 0

    # A checkpoint can be produced after patience was already exhausted
    # (for example, a final ``last.pt`` written before the loop broke). Do not
    # run additional epochs in that case.
    if (
        resumed
        and int(config.early_stopping_patience) > 0
        and epochs_without_improvement >= int(config.early_stopping_patience)
    ):
        stopped_early = True
        print(
            "Early stopping already satisfied at resume: "
            f"{epochs_without_improvement} epoch(s) without improvement "
            f"(patience={config.early_stopping_patience})."
        )
        return _fit_summary(
            history_records=history_records,
            epochs_run=epochs_run,
            best_epoch=best_epoch,
            best_metric=best_metric,
            stopped_early=stopped_early,
            checkpoint_path=checkpoint_path,
            log_file=log_file,
            resolved_device=resolved_device,
            config=config,
        )

    if start_epoch >= int(config.epochs):
        raise ValueError(
            f"start_epoch {start_epoch} is not less than epochs {config.epochs}"
        )

    for epoch in range(int(start_epoch), int(config.epochs)):
        # Deterministic, epoch-dependent shuffle for the training loader.
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Keep the source-training dataset's augmentation epoch in sync with
        # the deterministic epoch-aware sampler before each training epoch.
        dataset = getattr(train_loader, "dataset", None)
        if dataset is not None and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        started = time.perf_counter()

        train_metrics = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            resolved_device,
            max_batches=config.max_train_batches,
            grad_clip_norm=config.grad_clip_norm,
            use_amp=use_amp,
            scaler=scaler,
        )
        val_metrics = validate_epoch(
            model,
            val_loader,
            loss_fn,
            resolved_device,
            max_batches=config.max_val_batches,
        )

        if scheduler is not None:
            scheduler.step()

        validation_dice = float(val_metrics["dice"])
        improved = _is_improvement(
            validation_dice, float(best_metric), float(config.min_delta)
        )
        if improved:
            best_metric = validation_dice
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        record = {
            "epoch": epoch,
            "seed": int(config.seed),
            "config": config.to_dict(),
            "learning_rate": epoch_learning_rate,
            "training": train_metrics,
            "validation": val_metrics,
            "validation_dice": validation_dice,
            "best_validation_dice": float(best_metric),
            "improved": bool(improved),
            "duration_seconds": time.perf_counter() - started,
        }
        history_records.append(record)
        _append_jsonl(log_file, record)

        if improved:
            save_checkpoint(
                checkpoint_path / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=float(best_metric),
                config=config,
                history=history_records,
                device=resolved_device,
                extra={"selection_criterion": config.monitor},
                best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
            )
        save_checkpoint(
            checkpoint_path / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=float(best_metric),
            config=config,
            history=history_records,
            device=resolved_device,
            extra={"selection_criterion": config.monitor},
            best_epoch=best_epoch,
            epochs_without_improvement=epochs_without_improvement,
        )

        epochs_run += 1
        print(
            f"[epoch {epoch + 1}/{config.epochs}] "
            f"train_loss={train_metrics['total_loss']:.6f} "
            f"val_loss={val_metrics['total_loss']:.6f} "
            f"val_dice={validation_dice:.6f} "
            f"lr={epoch_learning_rate:.3e}"
            + (" *" if improved else "")
        )

        if (
            int(config.early_stopping_patience) > 0
            and epochs_without_improvement >= int(config.early_stopping_patience)
        ):
            stopped_early = True
            print(
                f"Early stopping: no validation Dice improvement for "
                f"{epochs_without_improvement} epoch(s)."
            )
            break

    return _fit_summary(
        history_records=history_records,
        epochs_run=epochs_run,
        best_epoch=best_epoch,
        best_metric=best_metric,
        stopped_early=stopped_early,
        checkpoint_path=checkpoint_path,
        log_file=log_file,
        resolved_device=resolved_device,
        config=config,
    )


def run_source_only_training(
    *,
    model: torch.nn.Module,
    config: TrainingConfig,
    loss_fn: Optional[Any] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    use_amp: bool = False,
    **loader_kwargs: Any,
) -> Dict[str, Any]:
    """Entry point for the source-only baseline.

    Always builds loaders through :func:`build_source_only_dataloaders`, which
    validates the hash-verified D-S1-03 registry, excludes exactly the approved
    23 records, and never constructs the native target cohort. There is no
    custom loader factory and no switch to disable the population check.
    """
    config.validate()

    use_amp = bool(use_amp or getattr(config, "use_amp", False))

    forbidden = {
        "loader_factory",
        "require_approved_population",
        "train_expected_count",
        "val_expected_count",
        "exclusion_registry",
    } & set(loader_kwargs)
    if forbidden:
        raise SourceOnlyLoaderError(
            "run_source_only_training does not accept registry-bypass options: "
            f"{sorted(forbidden)}. Loaders are always built through the "
            "validated source-only factory."
        )

    # G4-D4: configure strict determinism before resolving the device or
    # constructing any loader.
    configure_determinism()

    resolved_device = resolve_device(config.device)

    # G4-D7: pinned memory follows the resolved execution device. An explicit
    # caller override is still honoured, but the default is CUDA-only.
    if "pin_memory" not in loader_kwargs:
        loader_kwargs["pin_memory"] = pin_memory_for_device(resolved_device)

    loaders = build_source_only_dataloaders(
        seed=int(config.seed),
        batch_size=int(config.batch_size),
        num_workers=int(config.num_workers),
        **loader_kwargs,
    )

    # Validate again here so a patched factory still cannot disable the policy.
    validate_source_only_loaders(loaders)
    assert_approved_primary_population(
        len(loaders["train_dataset"]), len(loaders["val_dataset"])
    )

    if loss_fn is None:
        loss_fn = MobileDFUCompoundLoss()
    if optimizer is None:
        optimizer = build_optimizer(model, config)
    if scheduler is None:
        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            warmup_epochs=config.warmup_epochs,
            total_epochs=config.epochs,
            base_learning_rate=config.learning_rate,
            min_learning_rate=config.min_learning_rate,
        )

    return fit(
        model=model,
        train_loader=loaders["train_loader"],
        val_loader=loaders["val_loader"],
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_sampler=loaders.get("train_sampler"),
        config=config,
        device=resolved_device,
        use_amp=use_amp,
    )