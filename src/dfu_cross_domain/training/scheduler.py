"""Learning-rate schedule for the baseline training pipeline.

Implements the approved protocol: linear warm-up followed by cosine annealing
down to a configured minimum. ``torch.optim.lr_scheduler.LambdaLR`` is used so
the schedule state is captured by the standard ``state_dict`` and can be
restored from a checkpoint.

Discrete warm-up interpretation (G4-D1)
---------------------------------------
The protocol describes warm-up as ``0 -> eta``. Training advances in discrete
optimizer steps, so the first warm-up epoch uses ``eta / warmup_epochs``
(``1e-4`` for the approved 5-epoch, ``5e-4`` schedule) rather than exactly zero;
a zero-LR epoch would discard a full pass over the data. The final warm-up
epoch reaches the peak ``eta`` and the cosine phase begins at the following
epoch.

Exact cosine endpoint (G4-D2)
-----------------------------
With 100 total epochs and a 5-epoch warm-up, the decay phase must land exactly
on ``eta_min`` at the last executed epoch (index 99). The cosine progress
therefore runs from epoch 5 to epoch 99 inclusive, i.e. across
``total_epochs - warmup_epochs - 1`` intervals. Using ``total - warmup``
intervals would leave the final epoch at approximately ``1.136e-6`` instead of
the approved ``1e-6``.
"""

from __future__ import annotations

import math
from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    *,
    warmup_epochs: int,
    total_epochs: int,
    base_learning_rate: float,
    min_learning_rate: float,
) -> Optional[LambdaLR]:
    """Build a linear-warm-up + cosine-annealing ``LambdaLR``.

    The multiplier at epoch ``e`` is:

    * warm-up (``e < warmup_epochs``): ``(e + 1) / warmup_epochs``
    * decay: ``min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(pi * p))`` divided
      by ``base_lr``, where ``p`` advances from 0 to 1 across the decay phase
      and reaches exactly 1 at the final executed epoch.

    Returns ``None`` when ``total_epochs <= 0`` or ``warmup_epochs`` covers the
    whole run, in which case a constant schedule is acceptable.
    """
    warmup_epochs = int(warmup_epochs)
    total_epochs = int(total_epochs)
    base_learning_rate = float(base_learning_rate)
    min_learning_rate = float(min_learning_rate)

    if total_epochs <= 0:
        return None
    if base_learning_rate <= 0:
        raise ValueError("base_learning_rate must be positive")
    if min_learning_rate <= 0:
        raise ValueError("min_learning_rate must be positive")
    if min_learning_rate > base_learning_rate:
        raise ValueError("min_learning_rate must not exceed base_learning_rate")
    if warmup_epochs < 0 or warmup_epochs > total_epochs:
        raise ValueError("warmup_epochs must be in [0, total_epochs]")

    # Inclusive endpoint: epoch ``total_epochs - 1`` must reach ``p == 1``.
    decay_epochs = max(1, total_epochs - warmup_epochs - 1)

    def _lr_multiplier(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)

        progress = float(epoch - warmup_epochs) / float(decay_epochs)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        current = min_learning_rate + (base_learning_rate - min_learning_rate) * cosine
        return current / base_learning_rate

    return LambdaLR(optimizer, lr_lambda=_lr_multiplier)