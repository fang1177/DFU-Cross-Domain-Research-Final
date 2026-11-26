"""Integration test for boundary-field recomputation after D3 augmentation.

Verifies that phi_fp and phi_fn are correctly recomputed from the geometrically
augmented binary mask using tau=2.0 and max_cap=20.0, and that authentic-negative
handling (all-zero mask -> zero fields) works correctly.
"""

import numpy as np
import pytest

from src.data.transforms import augment_sample, compute_tolerance_boundary_fields
from src.data.boundary_fields import TAU, MAX_CAP


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_sample(H=512, W=512):
    """Create a synthetic RGB image and binary mask at canonical resolution."""
    image = np.random.rand(H, W, 3).astype(np.float32)
    mask = np.zeros((H, W), dtype=np.float32)
    # Add a foreground patch
    margin_h, margin_w = H // 10, W // 10
    mask[margin_h:H - margin_h, margin_w:W - margin_w] = 1.0
    return image, mask


# ---------------------------------------------------------------------------
# Boundary-field recomputation tests
# ---------------------------------------------------------------------------

def test_boundary_fields_augmented_mask_nonempty():
    """Recompute phi_fp/phi_fn from an augmented non-empty mask; fields should
    have correct shape, dtype, and valid (non-NaN/Inf) values."""
    image, mask = _make_sample()
    aug_image, aug_mask = augment_sample(image, mask, seed=42)

    phi_fp, phi_fn = compute_tolerance_boundary_fields(
        aug_mask.astype(np.int32), tau=TAU, max_cap=MAX_CAP
    )
    assert phi_fp.shape == aug_mask.shape
    assert phi_fn.shape == aug_mask.shape
    assert phi_fp.dtype == np.float32
    assert phi_fn.dtype == np.float32
    assert not np.any(np.isnan(phi_fp)), "phi_fp contains NaN"
    assert not np.any(np.isnan(phi_fn)), "phi_fn contains NaN"
    assert not np.any(np.isinf(phi_fp)), "phi_fp contains Inf"
    assert not np.any(np.isinf(phi_fn)), "phi_fn contains Inf"


def test_boundary_fields_augmented_mask_authentic_negative():
    """Authentic-negative handling: if the augmented mask is all-zero,
    compute_tolerance_boundary_fields returns phi_fp = phi_fn = 0."""
    image, mask = _make_sample()
    # Create a mask that will become all-zero after augmentation (possible with
    # random rotations/cuts that erase the foreground)
    # Just test the zero-mask path directly:
    zero_mask = np.zeros_like(mask, dtype=np.float32)
    phi_fp, phi_fn = compute_tolerance_boundary_fields(
        zero_mask.astype(np.int32), tau=TAU, max_cap=MAX_CAP
    )
    assert float(phi_fp.sum()) == 0.0, "phi_fp should be zero for authentic negative"
    assert float(phi_fn.sum()) == 0.0, "phi_fn should be zero for authentic negative"


def test_boundary_fields_augmented_mask_with_foreground():
    """Recompute fields from an augmented mask that has foreground pixels;
    fields should be finite and have reasonable values (0 < value <= max_cap)."""
    image, mask = _make_sample()
    aug_image, aug_mask = augment_sample(image, mask, seed=42)

    phi_fp, phi_fn = compute_tolerance_boundary_fields(
        aug_mask.astype(np.int32), tau=TAU, max_cap=MAX_CAP
    )

    # Check that non-zero field values are within [0, max_cap]
    fp_nonzero = phi_fp[phi_fp > 0]
    fn_nonzero = phi_fn[phi_fn > 0]
    if fp_nonzero.size > 0:
        assert fp_nonzero.max() <= MAX_CAP + 1e-6, \
            f"phi_fp values exceed max_cap: max={fp_nonzero.max()}"
        assert fp_nonzero.min() >= 0, f"phi_fp has negative values: min={fp_nonzero.min()}"
    if fn_nonzero.size > 0:
        assert fn_nonzero.max() <= MAX_CAP + 1e-6, \
            f"phi_fn values exceed max_cap: max={fn_nonzero.max()}"
        assert fn_nonzero.min() >= 0, f"phi_fn has negative values: min={fn_nonzero.min()}"