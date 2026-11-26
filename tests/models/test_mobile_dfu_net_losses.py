"""
Unit and integration tests for MobileDFU-Net loss functions:
1. Smooth Dice loss with non-empty masks.
2. Smooth Dice loss with 100% all-zero empty masks (no division by zero).
3. Weighted BCE loss with positive weighting (pos_weight=2.0).
4. Tolerance-bounded boundary loss using signed distance fields (phi_fp - phi_fn) and a linear sigmoid weighting.
5. Tolerance-Bounded Boundary loss zero penalty on empty negative masks.
6. Boundary loss deadband tolerance (zero penalty inside tau=2.0 band).
7. Auxiliary boundary loss supervision.
8. Compound loss forward computation and component dictionary.
9. End-to-end backpropagation through all loss components to model logits.
10. Numerical stability with extreme logits (+50.0, -50.0).
11. Mixed batch handling (one empty mask, one micro-ulcer, one normal ulcer).
"""

import pytest
import torch
import torch.nn.functional as F

from src.models.mobile_dfu_net.losses import (
    SmoothDiceLoss,
    WeightedBCELoss,
    ToleranceBoundedBoundaryLoss,
    AuxiliaryBoundaryLoss,
    MobileDFUCompoundLoss
)
from src.models.mobile_dfu_net.config import MobileDFULossConfig


@pytest.fixture
def loss_config():
    return MobileDFULossConfig()


def test_01_smooth_dice_nonempty():
    """Test 1: Smooth Dice loss with non-empty masks."""
    dice_fn = SmoothDiceLoss(eps=1e-6)
    # Perfect prediction: logits >> 0 where target == 1, logits << 0 where target == 0
    targets = torch.zeros(1, 1, 64, 64)
    targets[:, :, 10:20, 10:20] = 1.0
    logits = torch.ones_like(targets) * -20.0
    logits[:, :, 10:20, 10:20] = 20.0
    
    loss = dice_fn(logits, targets)
    assert loss.item() < 0.01  # Perfect match should yield near zero loss


def test_02_smooth_dice_empty_mask():
    """Test 2: Smooth Dice loss with all-zero empty mask does not divide by zero or NaN."""
    dice_fn = SmoothDiceLoss(eps=1e-6)
    targets = torch.zeros(2, 1, 64, 64)  # 100% empty
    # Model predicts near zero (large negative logits)
    logits = torch.ones_like(targets) * -15.0
    loss = dice_fn(logits, targets)
    assert torch.isfinite(loss)
    assert loss.item() < 0.05


def test_03_weighted_bce_loss():
    """Test 3: Weighted BCE loss applies pos_weight=2.0 to foreground."""
    bce_weighted = WeightedBCELoss(pos_weight=2.0)
    bce_unweighted = WeightedBCELoss(pos_weight=1.0)
    
    targets = torch.ones(1, 1, 10, 10)
    logits = torch.zeros(1, 1, 10, 10)  # probs = 0.5
    
    loss_w = bce_weighted(logits, targets)
    loss_u = bce_unweighted(logits, targets)
    
    # Foreground error should be penalized roughly 2x
    assert torch.allclose(loss_w, loss_u * 2.0, atol=1e-5)


def test_04_tolerance_bounded_boundary_loss():
    """Test 4: Tolerance-Bounded Boundary loss penalizes predictions outside tolerance band."""
    boundary_fn = ToleranceBoundedBoundaryLoss(tau=2.0, max_cap=20.0)
    
    # Exterior false positive distance field
    phi_fp = torch.zeros(1, 1, 64, 64)
    phi_fp[:, :, 30:40, 30:40] = 10.0  # Far away from boundary
    phi_fn = torch.zeros(1, 1, 64, 64)
    
    # False positive prediction in the exterior zone
    logits = torch.ones(1, 1, 64, 64) * -10.0
    logits[:, :, 30:40, 30:40] = 10.0  # Strong false positive
    
    loss = boundary_fn(logits, phi_fp, phi_fn)
    assert loss.item() > 0.0
    assert torch.isfinite(loss)


def test_05_boundary_loss_empty_negative_masks():
    """Test 5: Boundary loss gives exactly 0.0 penalty for authentic negative masks where distance fields are 0."""
    boundary_fn = ToleranceBoundedBoundaryLoss(tau=2.0, max_cap=20.0)
    phi_fp = torch.zeros(2, 1, 64, 64)  # 0 on authentic negatives
    phi_fn = torch.zeros(2, 1, 64, 64)
    
    # Arbitrary logits
    logits = torch.randn(2, 1, 64, 64)
    loss = boundary_fn(logits, phi_fp, phi_fn)
    assert loss.item() == 0.0


def test_06_boundary_loss_differentiability():
    """Test 6: Boundary loss is fully differentiable with respect to logits."""
    boundary_fn = ToleranceBoundedBoundaryLoss(tau=2.0, max_cap=20.0)
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    phi_fp = torch.rand(2, 1, 64, 64) * 10.0
    phi_fn = torch.rand(2, 1, 64, 64) * 10.0
    
    loss = boundary_fn(logits, phi_fp, phi_fn)
    loss.backward()
    
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad.abs().sum() > 0).item()


def test_06b_boundary_exact_manual_formula():
    """Test 6b: loss equals mean((phi_fp - phi_fn) * sigmoid(logits)) exactly."""
    boundary_fn = ToleranceBoundedBoundaryLoss(tau=2.0, max_cap=20.0)
    logits = torch.tensor([[[[-2.0, 0.0], [1.0, 3.0]]]])
    phi_fp = torch.tensor([[[[0.0, 2.0], [0.0, 5.0]]]])
    phi_fn = torch.tensor([[[[1.0, 0.0], [4.0, 0.0]]]])

    loss = boundary_fn(logits, phi_fp, phi_fn)
    expected = torch.mean((phi_fp - phi_fn) * torch.sigmoid(logits))
    assert torch.allclose(loss, expected, atol=1e-6)


def test_06c_boundary_positive_exterior_only():
    """Test 6c: exterior-only signed field produces a positive loss."""
    boundary_fn = ToleranceBoundedBoundaryLoss()
    logits = torch.full((1, 1, 4, 4), 10.0)
    phi_fp = torch.full_like(logits, 10.0)
    phi_fn = torch.zeros_like(logits)

    loss = boundary_fn(logits, phi_fp, phi_fn)
    assert loss.item() > 0.0
    assert torch.isfinite(loss)


def test_06d_boundary_negative_interior_only():
    """Test 6d: interior-only signed field produces a negative loss."""
    boundary_fn = ToleranceBoundedBoundaryLoss()
    logits = torch.full((1, 1, 4, 4), 10.0)
    phi_fp = torch.zeros_like(logits)
    phi_fn = torch.full_like(logits, 10.0)

    loss = boundary_fn(logits, phi_fp, phi_fn)
    assert loss.item() < 0.0
    assert torch.isfinite(loss)


def test_06e_boundary_sign_convention():
    """Test 6e: phi_fp - phi_fn gives positive exterior and negative interior."""
    boundary_fn = ToleranceBoundedBoundaryLoss()
    logits = torch.full((1, 1, 4, 4), 10.0)

    exterior = boundary_fn(logits, torch.full_like(logits, 5.0), torch.zeros_like(logits))
    interior = boundary_fn(logits, torch.zeros_like(logits), torch.full_like(logits, 5.0))

    assert exterior.item() > 0.0
    assert interior.item() < 0.0
    assert torch.allclose(exterior, -interior, atol=1e-6)


def test_06f_boundary_zero_fields_exactly_zero():
    """Test 6f: zero distance fields produce exactly zero boundary loss."""
    boundary_fn = ToleranceBoundedBoundaryLoss()
    logits = torch.randn(2, 1, 8, 8)
    phi_fp = torch.zeros_like(logits)
    phi_fn = torch.zeros_like(logits)

    loss = boundary_fn(logits, phi_fp, phi_fn)
    assert loss.item() == 0.0


def test_06g_boundary_shape_mismatch_raises():
    """Test 6g: mismatched shapes raise ValueError; (H,W) fields do not broadcast."""
    boundary_fn = ToleranceBoundedBoundaryLoss()
    logits = torch.zeros(1, 1, 4, 4)
    phi_fp = torch.zeros(1, 1, 4, 4)

    with pytest.raises(ValueError, match="Shape mismatch"):
        boundary_fn(logits, phi_fp, torch.zeros(1, 1, 4, 5))

    # A (H,W) field must not silently broadcast against [B,1,H,W] logits.
    with pytest.raises(ValueError, match="Shape mismatch"):
        boundary_fn(logits, phi_fp, torch.zeros(4, 4))


def test_06h_boundary_dtype_mismatch_raises():
    """Test 6h: mismatched dtypes raise TypeError."""
    boundary_fn = ToleranceBoundedBoundaryLoss()
    logits = torch.zeros(1, 1, 4, 4, dtype=torch.float32)
    phi_fp = torch.zeros(1, 1, 4, 4, dtype=torch.float32)
    phi_fn = torch.zeros(1, 1, 4, 4, dtype=torch.float64)

    with pytest.raises(TypeError, match="Dtype mismatch"):
        boundary_fn(logits, phi_fp, phi_fn)


def test_06i_boundary_device_mismatch_raises():
    """Test 6i: mismatched devices raise RuntimeError (meta device used on CPU-only)."""
    boundary_fn = ToleranceBoundedBoundaryLoss()
    logits = torch.zeros(1, 1, 4, 4)
    phi_fp = torch.zeros(1, 1, 4, 4)
    other_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("meta")
    phi_fn = torch.zeros(1, 1, 4, 4, device=other_device)

    with pytest.raises(RuntimeError, match="Device mismatch"):
        boundary_fn(logits, phi_fp, phi_fn)


def test_06j_boundary_analytical_gradient():
    """Test 6j: gradient equals mean(phi_signed * p * (1 - p))."""
    torch.manual_seed(0)
    boundary_fn = ToleranceBoundedBoundaryLoss()
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    phi_fp = torch.rand(2, 1, 8, 8)
    phi_fn = torch.rand(2, 1, 8, 8)

    loss = boundary_fn(logits, phi_fp, phi_fn)
    loss.backward()

    probs = torch.sigmoid(logits.detach())
    expected_grad = (phi_fp - phi_fn) * probs * (1.0 - probs) / logits.numel()
    assert torch.allclose(logits.grad, expected_grad, atol=1e-6)


def test_06k_boundary_extreme_logits_finite():
    """Test 6k: +1000/-1000 logits produce finite loss and finite gradients."""
    boundary_fn = ToleranceBoundedBoundaryLoss()
    logits = torch.tensor([-1000.0, 1000.0]).view(1, 1, 1, 2).requires_grad_(True)
    phi_fp = torch.tensor([20.0, 0.0]).view(1, 1, 1, 2)
    phi_fn = torch.tensor([0.0, 20.0]).view(1, 1, 1, 2)

    loss = boundary_fn(logits, phi_fp, phi_fn)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_06l_boundary_not_clamped():
    """Test 6l: the signed loss is not clamped at zero."""
    boundary_fn = ToleranceBoundedBoundaryLoss()
    logits = torch.full((1, 1, 4, 4), 10.0)
    phi_fp = torch.zeros_like(logits)
    phi_fn = torch.full_like(logits, 10.0)

    loss = boundary_fn(logits, phi_fp, phi_fn)
    assert loss.item() < -9.0


def test_07_auxiliary_boundary_loss():
    """Test 7: Auxiliary boundary loss supervises edge head."""
    aux_fn = AuxiliaryBoundaryLoss()
    b_logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    b_targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
    
    loss = aux_fn(b_logits, b_targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert b_logits.grad is not None


def test_08_compound_loss_components(loss_config):
    """Test 8: MobileDFUCompoundLoss returns all 4 component losses plus total_loss."""
    compound_fn = MobileDFUCompoundLoss(loss_config)
    
    model_outputs = {
        "seg_logits": torch.randn(2, 1, 64, 64, requires_grad=True),
        "boundary_logits": torch.randn(2, 1, 64, 64, requires_grad=True)
    }
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
    phi_fp = torch.rand(2, 1, 64, 64) * 5.0
    phi_fn = torch.rand(2, 1, 64, 64) * 5.0
    
    losses = compound_fn(model_outputs, targets, phi_fp, phi_fn)
    
    assert "total_loss" in losses
    assert "loss_dice" in losses
    assert "loss_bce" in losses
    assert "loss_boundary" in losses
    assert "loss_aux" in losses
    
    for k, v in losses.items():
        assert torch.isfinite(v), f"Loss component {k} is not finite!"


def test_09_numerical_stability_extreme_logits(loss_config):
    """Test 9: Numerical stability with extreme logits (+100.0, -100.0)."""
    compound_fn = MobileDFUCompoundLoss(loss_config)
    
    model_outputs = {
        "seg_logits": torch.tensor([[-100.0, 100.0], [50.0, -50.0]]).view(1, 1, 2, 2).requires_grad_(True),
        "boundary_logits": torch.tensor([[100.0, -100.0], [-50.0, 50.0]]).view(1, 1, 2, 2).requires_grad_(True)
    }
    targets = torch.tensor([[0.0, 1.0], [1.0, 0.0]]).view(1, 1, 2, 2)
    phi_fp = torch.zeros(1, 1, 2, 2)
    phi_fn = torch.zeros(1, 1, 2, 2)
    
    losses = compound_fn(model_outputs, targets, phi_fp, phi_fn)
    assert torch.isfinite(losses["total_loss"])
    losses["total_loss"].backward()
    assert torch.isfinite(model_outputs["seg_logits"].grad).all()


def test_10_mixed_batch_handling(loss_config):
    """Test 10: Mixed batch handling (one empty mask, one micro-ulcer, one normal ulcer)."""
    compound_fn = MobileDFUCompoundLoss(loss_config)
    
    b = 3
    targets = torch.zeros(b, 1, 64, 64)
    # Sample 0: All zero empty mask
    # Sample 1: Micro-ulcer (3 pixels)
    targets[1, 0, 10:13, 10] = 1.0
    # Sample 2: Normal ulcer (20x20 pixels)
    targets[2, 0, 20:40, 20:40] = 1.0
    
    phi_fp = torch.rand(b, 1, 64, 64) * 3.0
    phi_fp[0] = 0.0  # Sample 0 is authentic negative control: 0 boundary fields
    phi_fn = torch.rand(b, 1, 64, 64) * 3.0
    phi_fn[0] = 0.0
    
    model_outputs = {
        "seg_logits": torch.randn(b, 1, 64, 64, requires_grad=True),
        "boundary_logits": torch.randn(b, 1, 64, 64, requires_grad=True)
    }
    
    losses = compound_fn(model_outputs, targets, phi_fp, phi_fn)
    assert torch.isfinite(losses["total_loss"])
    losses["total_loss"].backward()
    assert torch.isfinite(model_outputs["seg_logits"].grad).all()


def test_11_compound_loss_api_and_weights_unchanged():
    """Test 11: compound API, keys, and approved weights remain unchanged."""
    config = MobileDFULossConfig()
    assert config.lambda_dice == 1.0
    assert config.lambda_bce == 0.5
    assert config.lambda_boundary == 0.5
    assert config.lambda_aux == 0.3

    compound_fn = MobileDFUCompoundLoss(config)
    model_outputs = {
        "seg_logits": torch.zeros(1, 1, 8, 8, requires_grad=True),
        "boundary_logits": torch.zeros(1, 1, 8, 8)
    }
    targets = torch.zeros(1, 1, 8, 8)
    targets[:, :, 2:6, 2:6] = 1.0
    phi_fp = torch.zeros(1, 1, 8, 8)
    phi_fn = torch.ones(1, 1, 8, 8) * 5.0

    losses = compound_fn(model_outputs, targets, phi_fp, phi_fn)

    assert set(losses.keys()) == {
        "total_loss", "loss_dice", "loss_bce", "loss_boundary", "loss_aux"
    }
    expected_total = (
        config.lambda_dice * losses["loss_dice"] +
        config.lambda_bce * losses["loss_bce"] +
        config.lambda_boundary * losses["loss_boundary"] +
        config.lambda_aux * losses["loss_aux"]
    )
    assert torch.allclose(losses["total_loss"], expected_total, atol=1e-6)
    # The signed boundary component must remain negative, not clamped.
    assert losses["loss_boundary"].item() < 0.0
    assert torch.isfinite(losses["total_loss"])

