"""Strict deterministic execution settings for the baseline training pipeline.

G4-D4 approved strict determinism. The settings are:

* ``torch.use_deterministic_algorithms(True)``
* ``torch.backends.cudnn.deterministic = True``
* ``torch.backends.cudnn.benchmark = False``
* ``CUBLAS_WORKSPACE_CONFIG=:4096:8``

``CUBLAS_WORKSPACE_CONFIG`` is a process-level environment variable. It must be
present before the first CUDA context (and therefore before cuBLAS is
initialised). This module sets a default as soon as it is imported, which is
the earliest safe point available to the training package, and
:func:`configure_determinism` re-checks it before every training run.

The pipeline deliberately does not fall back to non-deterministic execution. If
the environment already requests a different cuBLAS workspace configuration,
:func:`configure_determinism` raises instead of silently overriding it.
"""

from __future__ import annotations

import os

CUBLAS_WORKSPACE_CONFIG = ":4096:8"

_CUBLAS_ENV_VAR = "CUBLAS_WORKSPACE_CONFIG"


def _install_default_cublas_workspace_config() -> None:
    """Set the approved cuBLAS workspace value if the process has no setting.

    Existing environments are never overwritten here; a conflicting value is
    reported by :func:`configure_determinism` so the operator can decide.
    """
    if _CUBLAS_ENV_VAR not in os.environ:
        os.environ[_CUBLAS_ENV_VAR] = CUBLAS_WORKSPACE_CONFIG


# Install the environment default before importing torch. cuBLAS reads this
# value when it creates its first handle, which can happen before an explicit
# :func:`configure_determinism` call.
_install_default_cublas_workspace_config()

import torch  # noqa: E402  (intentionally imported after the environment setup)


def configure_determinism() -> None:
    """Enable the approved strict deterministic execution settings.

    Raises
    ------
    RuntimeError
        If the process environment pins a different ``CUBLAS_WORKSPACE_CONFIG``.
        Falling back silently would violate the approved G4-D4 decision.
    """
    _install_default_cublas_workspace_config()

    configured = os.environ.get(_CUBLAS_ENV_VAR)
    if configured != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "Strict determinism requires "
            f"{_CUBLAS_ENV_VAR}={CUBLAS_WORKSPACE_CONFIG!r}, but the process "
            f"environment has {configured!r}. Set the approved value before "
            "starting Python; do not weaken deterministic execution."
        )

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False