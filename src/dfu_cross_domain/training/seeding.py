"""Reproducible seeding helpers for the baseline training pipeline.

All randomness that can affect training is seeded here: Python ``random``,
NumPy, the PyTorch CPU RNG and (when present) the PyTorch CUDA RNG. RNG state
capture/restore is used by checkpointing so a resumed run continues the same
stochastic stream.
"""

from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np
import torch

from .determinism import configure_determinism


def seed_everything(seed: int) -> None:
    """Seed the global RNGs and enable strict deterministic execution.

    Parameters
    ----------
    seed:
        Non-negative integer seed.
    """
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")

    configure_determinism()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_worker_init_fn(base_seed: int):
    """Create a deterministic DataLoader ``worker_init_fn``.

    Each worker receives ``base_seed + worker_id`` so data-order helpers that
    use worker-local NumPy/Python RNGs remain reproducible for a fixed seed.
    """

    base_seed = int(base_seed)

    def _worker_init_fn(worker_id: int) -> None:
        worker_seed = base_seed + int(worker_id)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return _worker_init_fn


def capture_rng_state() -> Dict[str, Any]:
    """Capture the RNG states needed to resume a run exactly."""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG states previously produced by :func:`capture_rng_state`."""
    if not isinstance(state, dict):
        raise TypeError("rng state must be a dict produced by capture_rng_state")

    required = {"python", "numpy", "torch"}
    missing = required - set(state)
    if missing:
        raise ValueError(f"rng state is missing keys: {sorted(missing)}")

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])