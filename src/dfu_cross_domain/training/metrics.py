"""Segmentation metrics used for checkpoint selection.

Only the hard Dice coefficient is implemented here. It is evaluated on the
primary segmentation head and used solely to rank checkpoints on the internal
validation split supplied to the trainer. It does not access target data.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def dice_per_sample(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute a hard Dice coefficient for every sample in a batch.

    Empty-target convention: when both the target and the binarised prediction
    are empty the numerator and denominator are both ``eps``, so Dice is 1.0.
    When the target is empty but the prediction is not, Dice is close to 0.0.
    This mirrors the standard Smooth-Dice convention and avoids division by
    zero.

    Parameters
    ----------
    logits:
        ``[B, 1, H, W]`` raw segmentation logits.
    targets:
        ``[B, 1, H, W]`` binary targets in ``{0, 1}``.
    threshold:
        Probability threshold applied to ``sigmoid(logits)``.
    eps:
        Numerical stabiliser.

    Returns
    -------
    torch.Tensor
        ``[B]`` Dice values in ``[0, 1]``.
    """
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} does not match targets shape "
            f"{tuple(targets.shape)}"
        )
    if logits.ndim != 4:
        raise ValueError(f"expected [B, 1, H, W] tensors, got {tuple(logits.shape)}")
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")

    probs = torch.sigmoid(logits)
    predictions = (probs >= float(threshold)).to(targets.dtype)
    binary_targets = (targets >= 0.5).to(targets.dtype)

    reduce_dims = tuple(range(1, predictions.ndim))
    intersection = (predictions * binary_targets).sum(dim=reduce_dims)
    pred_sum = predictions.sum(dim=reduce_dims)
    target_sum = binary_targets.sum(dim=reduce_dims)

    return (2.0 * intersection + eps) / (pred_sum + target_sum + eps)