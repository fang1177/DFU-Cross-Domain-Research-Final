"""Seed-isolated output layout for the approved baseline runs.

G4 requires the three approved seeds (42, 1337, 2024) to write to separate
directories so that checkpoints and logs can never overwrite one another. This
module only constructs paths; it does not create them or launch training.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable, Tuple

from .config import TrainingConfig

#: Seeds approved for baseline replication.
APPROVED_BASELINE_SEEDS: Tuple[int, ...] = (42, 1337, 2024)

#: Root under which each seed gets its own directory.
DEFAULT_BASELINE_ROOT = "experiments/baseline_source_only"


def seed_run_dirs(
    seed: int,
    base_dir: str = DEFAULT_BASELINE_ROOT,
) -> dict:
    """Return the isolated checkpoint/log directories for one seed.

    The returned mapping has ``root``, ``checkpoint_dir`` and ``log_dir``.
    ``seed_<n>`` is always used, including for the default seed 42, so a run
    cannot accidentally write into the legacy shared directory.
    """
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")

    root = Path(base_dir) / f"seed_{seed}"
    return {
        "root": str(root),
        "checkpoint_dir": str(root / "checkpoints"),
        "log_dir": str(root / "logs"),
    }


def config_for_seed(
    config: TrainingConfig,
    seed: int,
    base_dir: str = DEFAULT_BASELINE_ROOT,
) -> TrainingConfig:
    """Return a copy of ``config`` pinned to one seed's isolated directories."""
    dirs = seed_run_dirs(seed, base_dir=base_dir)
    return replace(
        config,
        seed=int(seed),
        checkpoint_dir=dirs["checkpoint_dir"],
        log_dir=dirs["log_dir"],
    )


def build_baseline_seed_configs(
    config: TrainingConfig,
    seeds: Iterable[int] = APPROVED_BASELINE_SEEDS,
    base_dir: str = DEFAULT_BASELINE_ROOT,
) -> Tuple[TrainingConfig, ...]:
    """Build one isolated config per seed and reject collisions.

    Raises
    ------
    ValueError
        If ``seeds`` contains duplicates or two seeds resolve to the same
        output directory.
    """
    resolved_seeds = [int(seed) for seed in seeds]
    if len(set(resolved_seeds)) != len(resolved_seeds):
        raise ValueError(f"duplicate baseline seeds: {resolved_seeds}")

    configs = []
    seen_paths: dict = {}
    for seed in resolved_seeds:
        candidate = config_for_seed(config, seed, base_dir=base_dir)
        paths = (
            candidate.checkpoint_dir,
            candidate.log_dir,
        )
        for path in paths:
            if path in seen_paths:
                raise ValueError(
                    "seed output directory collision: "
                    f"{path!r} is shared by seeds {seen_paths[path]} and {seed}"
                )
            seen_paths[path] = seed
        configs.append(candidate)

    return tuple(configs)


def assert_distinct_seed_outputs(configs: Iterable[TrainingConfig]) -> None:
    """Fail if any two configs share a checkpoint or log directory."""
    seen: dict = {}
    for config in configs:
        for path in (config.checkpoint_dir, config.log_dir):
            if path in seen:
                raise ValueError(
                    f"seed output directory collision at {path!r}: "
                    f"seeds {seen[path]} and {config.seed}"
                )
            seen[path] = int(config.seed)