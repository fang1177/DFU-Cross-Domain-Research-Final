"""Single-epoch training and validation routines.

The engine is deliberately agnostic to the concrete dataset and model. It
expects:

* batches as mappings containing ``image``, ``mask``, ``phi_fp`` and
  ``phi_fn`` tensors (the contract of ``DFUCanonicalDataset``);
* ``model(images)`` to return a dict with at least ``seg_logits`` and
  ``boundary_logits`` (the ``MobileDFUNet`` forward contract);
* ``loss_fn(model_outputs, targets, phi_fp, phi_fn)`` to return a dict with a
  scalar ``total_loss`` (the ``MobileDFUCompoundLoss`` contract).

Validation never calls the optimizer and runs under ``torch.no_grad()``.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch

from .metrics import dice_per_sample

REQUIRED_BATCH_KEYS = ("image", "mask", "phi_fp", "phi_fn")

LOSS_COMPONENT_KEYS = (
    "total_loss",
    "loss_dice",
    "loss_bce",
    "loss_boundary",
    "loss_aux",
)


def _require_mapping(batch: Any) -> Mapping[str, Any]:
    if not isinstance(batch, Mapping):
        raise TypeError(
            "Expected each batch to be a mapping, got "
            f"{type(batch).__name__}"
        )
    missing = [key for key in REQUIRED_BATCH_KEYS if key not in batch]
    if missing:
        raise KeyError(f"Batch is missing required keys: {missing}")
    return batch


def _move_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def _accumulate(
    totals: Dict[str, float], values: Mapping[str, torch.Tensor], weight: int
) -> None:
    for key in LOSS_COMPONENT_KEYS:
        if key not in values:
            continue
        value = values[key]
        if not torch.is_tensor(value):
            raise TypeError(f"loss component {key!r} must be a tensor")
        if value.numel() != 1:
            raise ValueError(
                f"loss component {key!r} must be scalar, got shape "
                f"{tuple(value.shape)}"
            )
        totals[key] = totals.get(key, 0.0) + float(value.detach()) * weight


def _averages(totals: Dict[str, float], sample_count: int) -> Dict[str, float]:
    if sample_count <= 0:
        raise ValueError("No samples were processed")
    return {key: value / float(sample_count) for key, value in totals.items()}


def train_one_epoch(
    model: torch.nn.Module,
    loader: Any,
    loss_fn: Any,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    *,
    max_batches: Optional[int] = None,
    grad_clip_norm: Optional[float] = None,
    use_amp: bool = False,
    scaler: Optional[Any] = None,
) -> Dict[str, float]:
    """Run exactly one supervised training epoch.

    Returns a dict of sample-weighted mean loss components, including
    ``total_loss``. Parameters are updated exactly once per batch.
    """
    if len(loader) == 0:
        raise ValueError("Training loader is empty")
    if max_batches is not None and int(max_batches) <= 0:
        raise ValueError("max_batches must be positive when set")

    device = torch.device(device)
    model.train()

    totals: Dict[str, float] = {}
    sample_count = 0

    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break

        batch = _move_to_device(_require_mapping(raw_batch), device)
        images = batch["image"]
        targets = batch["mask"]
        phi_fp = batch["phi_fp"]
        phi_fn = batch["phi_fn"]

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            if scaler is None:
                scaler = torch.amp.GradScaler("cuda")
            with torch.amp.autocast("cuda"):
                model_outputs = model(images)
            # Keep loss computation in FP32 for numerical stability
            fp32_outputs = {k: (v.float() if torch.is_tensor(v) else v) for k, v in model_outputs.items()}
            losses = loss_fn(fp32_outputs, targets, phi_fp, phi_fn)
            total_loss = losses["total_loss"]
            if not torch.isfinite(total_loss).all():
                raise FloatingPointError(
                    f"Non-finite total loss at batch {batch_index}: "
                    f"{float(total_loss.detach())}"
                )
            scaler.scale(total_loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(grad_clip_norm)
                )
            # gradient finiteness check
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise FloatingPointError(f"Non-finite gradient at batch {batch_index}")
            scaler.step(optimizer)
            scaler.update()
        else:
            model_outputs = model(images)
            losses = loss_fn(model_outputs, targets, phi_fp, phi_fn)

            if "total_loss" not in losses:
                raise KeyError("loss_fn output is missing 'total_loss'")
            total_loss = losses["total_loss"]
            if not torch.isfinite(total_loss).all():
                raise FloatingPointError(
                    f"Non-finite total loss at batch {batch_index}: "
                    f"{float(total_loss.detach())}"
                )

            total_loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(grad_clip_norm)
                )
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise FloatingPointError(f"Non-finite gradient at batch {batch_index}")
            optimizer.step()

        batch_size = int(images.shape[0])
        if batch_size <= 0:
            raise ValueError("Encountered an empty batch")
        sample_count += batch_size
        _accumulate(totals, losses, batch_size)

    return _averages(totals, sample_count)


@torch.no_grad()
def validate_epoch(
    model: torch.nn.Module,
    loader: Any,
    loss_fn: Any,
    device: str | torch.device,
    *,
    threshold: float = 0.5,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate one full validation pass without updating any parameters.

    Returns sample-weighted mean loss components plus ``dice`` (the mean hard
    Dice coefficient across samples).
    """
    if len(loader) == 0:
        raise ValueError("Validation loader is empty")
    if max_batches is not None and int(max_batches) <= 0:
        raise ValueError("max_batches must be positive when set")

    device = torch.device(device)
    model.eval()

    totals: Dict[str, float] = {}
    sample_count = 0
    dice_total = 0.0

    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break

        batch = _move_to_device(_require_mapping(raw_batch), device)
        images = batch["image"]
        targets = batch["mask"]
        phi_fp = batch["phi_fp"]
        phi_fn = batch["phi_fn"]

        model_outputs = model(images)
        losses = loss_fn(model_outputs, targets, phi_fp, phi_fn)

        if "total_loss" not in losses:
            raise KeyError("loss_fn output is missing 'total_loss'")
        total_loss = losses["total_loss"]
        if not torch.isfinite(total_loss).all():
            raise FloatingPointError(
                f"Non-finite validation loss at batch {batch_index}: "
                f"{float(total_loss.detach())}"
            )

        if "seg_logits" not in model_outputs:
            raise KeyError("model output is missing 'seg_logits'")
        dice = dice_per_sample(
            model_outputs["seg_logits"], targets, threshold=threshold
        )

        batch_size = int(images.shape[0])
        if batch_size <= 0:
            raise ValueError("Encountered an empty batch")
        sample_count += batch_size
        dice_total += float(dice.sum().detach())
        _accumulate(totals, losses, batch_size)

    metrics = _averages(totals, sample_count)
    metrics["dice"] = dice_total / float(sample_count)
    return metrics