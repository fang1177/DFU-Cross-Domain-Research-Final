"""Configuration for the minimal source-only baseline training pipeline.

The optimizer and schedule defaults mirror the project's approved optimization
protocol in ``results/temporary/stage_2/loss_function_design.md`` section 3:

* AdamW with ``beta1=0.9``, ``beta2=0.999``, ``weight_decay=1e-4``
* Base learning rate ``5e-4``
* Linear warm-up over 5 epochs followed by cosine annealing to ``1e-6``
* Best checkpoint selected strictly by validation Dice

This module does not change model or loss interfaces, and it does not encode
any label-exclusion policy. The population supplied to the trainer is decided
by the caller (see ``source_only.py`` and the S1 decision register).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class TrainingConfig:
    """Runtime configuration for one baseline training run.

    Attributes
    ----------
    epochs:
        Total number of epochs to run.
    batch_size:
        Mini-batch size. The default follows the approved protocol (8).
    learning_rate:
        Peak learning rate reached at the end of warm-up.
    weight_decay:
        AdamW decoupled weight decay.
    beta1, beta2:
        AdamW momentum parameters.
    warmup_epochs:
        Number of linear warm-up epochs. ``0`` disables warm-up. The first
        warm-up epoch uses ``learning_rate / warmup_epochs`` (``1e-4`` for the
        approved schedule); see ``scheduler.py`` for the approved discrete
        interpretation of "0 -> eta".
    min_learning_rate:
        Lower bound of the cosine decay.
    seed:
        Global seed for Python, NumPy and PyTorch RNGs.
    num_workers:
        DataLoader worker count. ``0`` keeps everything in the main process.
    device:
        ``"auto"`` selects CUDA when available, otherwise CPU. Any value
        accepted by :class:`torch.device` is also allowed.
    grad_clip_norm:
        Optional gradient-norm clipping value. ``None`` disables clipping.
    monitor:
        Validation metric used for best-checkpoint selection. Only
        ``"val_dice"`` is currently supported.
    min_delta:
        Minimum improvement required to refresh the best checkpoint.
    early_stopping_patience:
        Number of consecutive non-improving epochs tolerated before stopping.
        Set to ``0`` to disable early stopping.
    checkpoint_dir, log_dir:
        Output directories for checkpoints and the JSONL training log.
    max_train_batches, max_val_batches:
        Optional per-epoch batch limits. Intended for smoke tests; ``None``
        means "use the whole loader".
    """

    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 5.0e-4
    weight_decay: float = 1.0e-4
    beta1: float = 0.9
    beta2: float = 0.999
    warmup_epochs: int = 5
    min_learning_rate: float = 1.0e-6

    seed: int = 42
    num_workers: int = 0
    device: str = "auto"
    grad_clip_norm: Optional[float] = None

    monitor: str = "val_dice"
    min_delta: float = 0.0
    early_stopping_patience: int = 15

    checkpoint_dir: str = "experiments/baseline_source_only/checkpoints"
    log_dir: str = "experiments/baseline_source_only/logs"

    max_train_batches: Optional[int] = None
    max_val_batches: Optional[int] = None

    def validate(self) -> None:
        """Fail loudly on configurations that cannot describe a valid run."""
        if int(self.epochs) <= 0:
            raise ValueError("epochs must be a positive integer")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be a positive integer")
        if float(self.learning_rate) <= 0:
            raise ValueError("learning_rate must be positive")
        if float(self.min_learning_rate) <= 0:
            raise ValueError("min_learning_rate must be positive")
        if float(self.min_learning_rate) > float(self.learning_rate):
            raise ValueError(
                "min_learning_rate must not exceed learning_rate"
            )
        if int(self.warmup_epochs) < 0:
            raise ValueError("warmup_epochs must be non-negative")
        if int(self.warmup_epochs) > int(self.epochs):
            raise ValueError("warmup_epochs must not exceed epochs")
        if float(self.weight_decay) < 0:
            raise ValueError("weight_decay must be non-negative")
        if not (0.0 <= float(self.beta1) < 1.0):
            raise ValueError("beta1 must be in [0, 1)")
        if not (0.0 <= float(self.beta2) < 1.0):
            raise ValueError("beta2 must be in [0, 1)")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")
        if int(self.num_workers) < 0:
            raise ValueError("num_workers must be non-negative")
        if self.grad_clip_norm is not None and float(self.grad_clip_norm) <= 0:
            raise ValueError("grad_clip_norm must be positive when set")
        if self.monitor != "val_dice":
            raise ValueError(
                f"Unsupported monitor {self.monitor!r}; only 'val_dice' is "
                "implemented"
            )
        if float(self.min_delta) < 0:
            raise ValueError("min_delta must be non-negative")
        if int(self.early_stopping_patience) < 0:
            raise ValueError("early_stopping_patience must be non-negative")
        if self.max_train_batches is not None and int(self.max_train_batches) <= 0:
            raise ValueError("max_train_batches must be positive when set")
        if self.max_val_batches is not None and int(self.max_val_batches) <= 0:
            raise ValueError("max_val_batches must be positive when set")

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation for logging/checkpoints."""
        return asdict(self)