"""Shared synthetic fixtures for focused data-loader and training tests."""

from __future__ import annotations

import os

import pytest
import torch

from tests.support_exclusion_registry import build_population


@pytest.fixture(autouse=True)
def _restore_global_determinism_state():
    """Keep strict-determinism settings from leaking between test modules."""
    previous_algorithms = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    previous_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    yield

    torch.use_deterministic_algorithms(
        previous_algorithms, warn_only=previous_warn_only
    )
    torch.backends.cudnn.deterministic = previous_cudnn_deterministic
    torch.backends.cudnn.benchmark = previous_cudnn_benchmark
    if previous_cublas is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_cublas


@pytest.fixture(scope="session")
def exclusion_population(tmp_path_factory):
    """A synthetic 945 / 191 project with the approved 23 exclusions."""
    return build_population(tmp_path_factory.mktemp("exclusion_population"))