"""Three-seed orchestration for the approved G4 source-only baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.data.stage_4_dataloaders import DEFAULT_PROJECT_ROOT
from src.models.mobile_dfu_net import (
    MobileDFUCompoundLoss,
    MobileDFUNet,
    MobileDFUNetConfig,
)

from .config import TrainingConfig
from .run_layout import (
    APPROVED_BASELINE_SEEDS,
    DEFAULT_BASELINE_ROOT,
    assert_distinct_seed_outputs,
    build_baseline_seed_configs,
)
from .seeding import seed_everything
from .trainer import (
    build_optimizer,
    build_warmup_cosine_scheduler,
    run_source_only_training,
)


class BaselineOutputArtifactError(RuntimeError):
    """Raised when prior baseline artifacts exist for an approved seed."""


def _resolve_base_dir(base_dir: str = DEFAULT_BASELINE_ROOT) -> str:
    """Resolve a possibly-relative baseline root against the project root."""
    base = Path(base_dir)
    if base.is_absolute():
        return str(base)
    return str((Path(DEFAULT_PROJECT_ROOT) / base).resolve())


def _has_prior_artifacts(checkpoint_dir: str, log_dir: str) -> bool:
    """Return True if the seed output tree already contains any artifact.

    Scans the seed run root together with the ``checkpoints`` and ``logs``
    output subdirectories recursively, so an unexpected existing file anywhere
    under the seed directory blocks a fresh run. Existing artifacts are never
    deleted or overwritten.
    """
    candidates = (
        Path(checkpoint_dir).parent,  # <base>/seed_<n> run root
        Path(checkpoint_dir),
        Path(log_dir),
    )
    for directory in candidates:
        if not directory.is_dir():
            continue
        try:
            if any(path.is_file() for path in directory.rglob("*")):
                return True
        except OSError:
            return True
    return False


def run_source_only_baseline(
    *,
    config: TrainingConfig,
    seeds: Iterable[int] = APPROVED_BASELINE_SEEDS,
    base_dir: str = DEFAULT_BASELINE_ROOT,
    loss_fn: Optional[Any] = None,
    activation_checkpointing: bool = False,
    use_amp: bool = False,
    colour_ablation: bool = False,
    boundary_ablation: bool = False,
    **loader_kwargs: Any,
) -> List[Dict[str, Any]]:
    """Run the approved source-only baseline for each seed with isolated outputs.

    The orchestration:

    * builds seed-isolated :class:`TrainingConfig` objects;
    * refuses to dispatch if any approved seed already has artifacts;
    * creates a fresh, independently initialized model and optimizer/scheduler
      per seed;
    * delegates to the existing source-only training entry point.

    This function does not enable implicit resume.
    """
    config.validate()

    resolved_base_dir = _resolve_base_dir(base_dir)
    seed_configs = build_baseline_seed_configs(
        config, seeds=seeds, base_dir=resolved_base_dir
    )
    assert_distinct_seed_outputs(seed_configs)

    # Preflight: inspect all approved seed output trees before any training
    # dispatch. Collect every conflict and refuse once, together, if any
    # artifact is found. Never delete or overwrite existing results.
    conflicts: List[str] = []
    for seed_config in seed_configs:
        if _has_prior_artifacts(
            str(seed_config.checkpoint_dir), str(seed_config.log_dir)
        ):
            conflicts.append(
                f"seed {int(seed_config.seed)}: "
                f"checkpoint_dir={seed_config.checkpoint_dir} "
                f"log_dir={seed_config.log_dir}"
            )

    if conflicts:
        raise BaselineOutputArtifactError(
            "Prior baseline artifacts detected for approved seeds; refusing "
            "to launch a fresh three-seed baseline run:\n"
            + "\n".join(conflicts)
        )

    summaries: List[Dict[str, Any]] = []
    for seed_config in seed_configs:
        seed = int(seed_config.seed)
        seed_config.use_amp = bool(use_amp)
        seed_everything(seed)
        model = MobileDFUNet(MobileDFUNetConfig())
        # Optional activation checkpointing (disabled by default). Set as an
        # attribute so mocked models in tests (which lack encoder/decoder) are
        # unaffected and the default FP32 path stays unchanged.
        model.activation_checkpointing = bool(activation_checkpointing)
        if getattr(model, "encoder", None) is not None:
            model.encoder.activation_checkpointing = bool(activation_checkpointing)
        if getattr(model, "decoder", None) is not None:
            model.decoder.activation_checkpointing = bool(activation_checkpointing)
        model.colour_ablation = bool(colour_ablation)
        model.boundary_ablation = bool(boundary_ablation)
        # Rebuild stem for colour ablation (simple stem instead of DCS-Stem)
        if model.colour_ablation and getattr(model, "stem", None) is not None:
            import torch.nn as nn
            from src.models.mobile_dfu_net.blocks import ConvBNAct
            model.stem = nn.Sequential(
                ConvBNAct(3, 24, kernel_size=3, stride=1, act_layer="hardswish"),
            )
        optimizer = build_optimizer(model, seed_config)
        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            warmup_epochs=int(seed_config.warmup_epochs),
            total_epochs=int(seed_config.epochs),
            base_learning_rate=float(seed_config.learning_rate),
            min_learning_rate=float(seed_config.min_learning_rate),
        )

        summary = run_source_only_training(
            model=model,
            config=seed_config,
            loss_fn=loss_fn if loss_fn is not None else MobileDFUCompoundLoss(),
            optimizer=optimizer,
            scheduler=scheduler,
            **loader_kwargs,
        )
        summaries.append(summary)

    return summaries
