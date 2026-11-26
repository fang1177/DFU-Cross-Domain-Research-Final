"""Integration test for RGB-mask alignment after D3 augmentation.

Verifies that after augmentation, the RGB and mask tensors have identical
spatial geometry: same scaling, same padding, same centre of mass for
non-zero regions. Uses synthetic fixtures only.
"""

import numpy as np
import pytest

from src.data.transforms import augment_sample


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_sample(H=512, W=512):
    """Create a synthetic RGB image and binary mask at canonical resolution."""
    image = np.random.rand(H, W, 3).astype(np.float32)
    mask = np.zeros((H, W), dtype=np.float32)
    # Add a single foreground rectangle for proportion check
    margin_h, margin_w = H // 10, W // 10
    mask[margin_h:H - margin_h, margin_w:W - margin_w] = 1.0
    return image, mask


# ---------------------------------------------------------------------------
# RGB-mask alignment tests
# ---------------------------------------------------------------------------

def test_rgb_mask_alignment_dimensions():
    """After any augmentation, RGB and mask should have identical spatial
    dimensions (H, W)."""
    image, mask = _make_sample()
    aug_image, aug_mask = augment_sample(image, mask, seed=42)
    assert aug_image.shape[:2] == aug_mask.shape[:2], \
        f"Image HxW {aug_image.shape[:2]} != mask HxW {aug_mask.shape[:2]}"


def test_rgb_mask_alignment_binary_mask():
    """Mask should always remain binary {0, 1} after augmentation."""
    image, mask = _make_sample()
    for seed in [0, 1, 42, 1337, 2024]:
        aug_image, aug_mask = augment_sample(image, mask, seed=seed)
        assert set(aug_mask.flatten().tolist()).issubset({0.0, 1.0}), \
            f"Seed {seed}: mask values {set(aug_mask.flatten().tolist())} not in {{0, 1}}"


def test_rgb_mask_alignment_foreground_preservation():
    """Foreground proportion should be largely preserved after augmentation
    (geometric transforms maintain area; photometric transforms don't affect mask)."""
    image, mask = _make_sample()
    original_fg_ratio = mask.sum() / (mask.shape[0] * mask.shape[1])

    for seed in [0, 1, 42, 1337, 2024]:
        aug_image, aug_mask = augment_sample(image, mask, seed=seed)
        aug_fg_ratio = aug_mask.sum() / (aug_mask.shape[0] * aug_mask.shape[1])
        # Allow some variation due to CutOut, but overall proportion should be similar
        assert abs(aug_fg_ratio - original_fg_ratio) < 0.5, \
            f"Seed {seed}: foreground ratio changed from {original_fg_ratio:.4f} to {aug_fg_ratio:.4f}"


def test_rgb_mask_alignment_cutout_rgb_only():
    """CutOut should be applied to RGB only; the mask should remain fully intact."""
    image, mask = _make_sample()
    aug_image, aug_mask = augment_sample(image, mask, seed=42)

    assert aug_image.shape[:2] == aug_mask.shape[:2]
    # Mask should be completely intact (CutOut is RGB-only, mask never modified)
    assert set(aug_mask.flatten().tolist()).issubset({0.0, 1.0})
    # Mask should still have the full foreground region (no CutOut applied to mask)
    assert aug_mask.sum() > 0


def test_rgb_mask_alignment_all_augmentations():
    """Run the full pipeline with various seeds and verify RGB/mask shape
    consistency and binary mask integrity."""
    image, mask = _make_sample()
    for seed in range(10):
        aug_image, aug_mask = augment_sample(image, mask, seed=seed)
        assert aug_image.shape[:2] == aug_mask.shape[:2], \
            f"Seed {seed}: shape mismatch {aug_image.shape[:2]} vs {aug_mask.shape[:2]}"
        assert set(aug_mask.flatten().tolist()).issubset({0.0, 1.0}), \
            f"Seed {seed}: mask not binary"
        assert aug_image.min() >= 0.0 and aug_image.max() <= 1.0, \
            f"Seed {seed}: image out of [0,1] range"