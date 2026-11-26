"""Minimal source-only baseline training infrastructure.

This package reuses the existing Stage 4A model and loss components without
modifying them. It never constructs or references the native external target
cohort. See ``source_only.py`` for the approved-population guard.
"""

from .config import TrainingConfig
from .determinism import configure_determinism
from .engine import train_one_epoch, validate_epoch
from .checkpoint import load_checkpoint, save_checkpoint
from .run_layout import (
    APPROVED_BASELINE_SEEDS,
    assert_distinct_seed_outputs,
    build_baseline_seed_configs,
    config_for_seed,
    seed_run_dirs,
)
from .scheduler import build_warmup_cosine_scheduler
from .seeding import seed_everything
from .source_only import (
    APPROVED_PRIMARY_POPULATION,
    FROZEN_PRIMARY_COUNTS,
    PendingAnalysisPopulationError,
    SourceOnlyLoaderError,
    assert_approved_primary_population,
    build_source_only_dataloaders,
    validate_source_only_loaders,
)
from .trainer import (
    build_optimizer,
    fit,
    resolve_device,
    run_source_only_training,
)
from .baseline import BaselineOutputArtifactError, run_source_only_baseline

__all__ = [
    "TrainingConfig",
    "train_one_epoch",
    "validate_epoch",
    "save_checkpoint",
    "load_checkpoint",
    "build_warmup_cosine_scheduler",
    "seed_everything",
    "configure_determinism",
    "APPROVED_BASELINE_SEEDS",
    "seed_run_dirs",
    "config_for_seed",
    "build_baseline_seed_configs",
    "assert_distinct_seed_outputs",
    "APPROVED_PRIMARY_POPULATION",
    "FROZEN_PRIMARY_COUNTS",
    "PendingAnalysisPopulationError",
    "SourceOnlyLoaderError",
    "assert_approved_primary_population",
    "build_source_only_dataloaders",
    "validate_source_only_loaders",
    "build_optimizer",
    "fit",
    "resolve_device",
    "run_source_only_training",
    "run_source_only_baseline",
    "BaselineOutputArtifactError",
]