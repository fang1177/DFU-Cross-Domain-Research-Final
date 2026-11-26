"""
Compound Loss Formulations for MobileDFU-Net:
1. Smooth Dice Loss (L_Dice): Stable for micro-lesions and authentic empty negative masks.
2. Weighted BCE Loss (L_BCE): Class-weighted with pos_weight=2.0 with logits.
3. Signed-Distance Linear Boundary Loss (L_T-Boundary): Linear penalty using
   ground-truth tolerance-bounded distance fields (Phi_FP, Phi_FN); tau and
   max_cap are applied during boundary-field generation.
4. Auxiliary Boundary Loss (L_Aux-Edge): Joint Dice + BCE on auxiliary boundary head predictions.
5. Combined MobileDFUObjective: Full compound objective.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mobile_dfu_net.config import MobileDFULossConfig


class SmoothDiceLoss(nn.Module):
    """
    Smooth Soft Dice Loss operating on logits:
    Dice = (2 * sum(Y * Y_hat) + eps) / (sum(Y) + sum(Y_hat) + eps)
    L_Dice = 1 - Dice
    Handles empty masks without division by zero.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Flatten spatial dimensions per batch sample: [B, 1, H, W] -> [B, -1]
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = torch.sum(probs_flat * targets_flat, dim=1)
        cardinality = torch.sum(probs_flat, dim=1) + torch.sum(targets_flat, dim=1)

        # Standard Dice Formulation:
        # When both ground truth and prediction are empty (cardinality == 0), Dice is defined as 1.0 (loss = 0.0)
        # Using eps in both numerator and denominator ensures numerical stability.
        # For continuous probabilities, when target is completely 0 (empty mask),
        # the model should be encouraged to predict 0 foreground:
        # If target is empty (targets_flat.sum(1) == 0), penalize any false positive predictions:
        # dice = 1.0 - (probs_flat.sum(1) / (probs_flat.size(1)))
        empty_targets = (torch.sum(targets_flat, dim=1) == 0)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        
        if empty_targets.any():
            # For empty masks, Dice score is 1.0 - (false positive rate)
            fp_rate = torch.sum(probs_flat, dim=1) / probs_flat.size(1)
            dice_empty = (1.0 - fp_rate).clamp(0.0, 1.0)
            dice = torch.where(empty_targets, dice_empty, dice)

        return torch.mean(1.0 - dice)


class WeightedBCELoss(nn.Module):
    """
    Numerically stable Weighted Binary Cross Entropy Loss with Logits:
    L_BCE = - (1/N) * sum[ w * Y * log(sigma(logits)) + (1 - Y) * log(1 - sigma(logits)) ]
    """
    def __init__(self, pos_weight: float = 2.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # pos_weight as tensor matching device of logits
        pw = torch.tensor([self.pos_weight], dtype=logits.dtype, device=logits.device)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)


class ToleranceBoundedBoundaryLoss(nn.Module):
    """
    Tolerance-Aware Signed-Distance Linear Boundary Loss.

    This is an adaptation of the Kervadec et al. boundary loss, not an exact
    reproduction. Ground-truth tolerance-bounded fields are combined into a
    signed field:

        phi_signed = phi_fp - phi_fn

    where phi_fp is positive in exterior background beyond the tolerance band
    and phi_fn is positive in interior foreground beyond the tolerance band.
    Therefore phi_signed is positive outside the object, zero inside the
    tolerance band, and negative in the deep interior.

        L_T-Boundary = mean( phi_signed * sigmoid(logits) )

    Minimising this term pushes the sigmoid probabilities toward 1 where
    phi_signed is negative (interior) and toward 0 where phi_signed is positive
    (exterior). Because the interior contribution is negative, the loss value
    can be negative; this is mathematically expected for a signed-distance
    region energy and must not be clamped.

    Empty masks: phi_fp = phi_fn = 0, so the boundary loss is exactly zero.

    Args:
        tau: Tolerance-band width in pixels. Retained for backward
            compatibility and provenance only; it is applied by the
            boundary-field generator, not by this loss.
        max_cap: Maximum distance cap in pixels. Retained for backward
            compatibility and provenance only; it is applied by the
            boundary-field generator, not by this loss.
    """
    def __init__(self, tau: float = 2.0, max_cap: float = 20.0):
        super().__init__()
        # Field-generation provenance parameters. They are intentionally not
        # used in forward(): the tolerance band and distance cap are already
        # baked into phi_fp and phi_fn by compute_tolerance_boundary_fields().
        self.tau = tau
        self.max_cap = max_cap

    @staticmethod
    def _validate_inputs(
        logits: torch.Tensor,
        phi_fp: torch.Tensor,
        phi_fn: torch.Tensor
    ) -> None:
        """Reject silent broadcasting and invalid dtype/device combinations."""
        if logits.shape != phi_fp.shape or logits.shape != phi_fn.shape:
            raise ValueError(
                "Shape mismatch in boundary loss: "
                f"logits {tuple(logits.shape)}, "
                f"phi_fp {tuple(phi_fp.shape)}, "
                f"phi_fn {tuple(phi_fn.shape)}. "
                "All tensors must have identical [B, 1, H, W] shapes."
            )
        if not torch.is_floating_point(logits):
            raise TypeError(
                f"logits must be a floating-point tensor, got {logits.dtype}."
            )
        if not torch.is_floating_point(phi_fp) or not torch.is_floating_point(phi_fn):
            raise TypeError(
                "phi_fp and phi_fn must be floating-point tensors, got "
                f"{phi_fp.dtype} and {phi_fn.dtype}."
            )
        if logits.dtype != phi_fp.dtype or logits.dtype != phi_fn.dtype:
            raise TypeError(
                "Dtype mismatch in boundary loss: "
                f"logits {logits.dtype}, phi_fp {phi_fp.dtype}, "
                f"phi_fn {phi_fn.dtype}. All tensors must share dtype."
            )
        if logits.device != phi_fp.device or logits.device != phi_fn.device:
            raise RuntimeError(
                "Device mismatch in boundary loss: "
                f"logits {logits.device}, phi_fp {phi_fp.device}, "
                f"phi_fn {phi_fn.device}. All tensors must share device."
            )

    def forward(
        self,
        logits: torch.Tensor,
        phi_fp: torch.Tensor,
        phi_fn: torch.Tensor
    ) -> torch.Tensor:
        self._validate_inputs(logits, phi_fp, phi_fn)

        phi_signed = phi_fp - phi_fn
        probs = torch.sigmoid(logits)
        return torch.mean(phi_signed * probs)


class AuxiliaryBoundaryLoss(nn.Module):
    """
    Supervision for Auxiliary Boundary Prediction Head:
    L_Aux = L_Dice(boundary_logits, boundary_target) + L_BCE(boundary_logits, boundary_target)
    """
    def __init__(self, eps: float = 1e-6, pos_weight: float = 2.0):
        super().__init__()
        self.dice = SmoothDiceLoss(eps=eps)
        self.bce = WeightedBCELoss(pos_weight=pos_weight)

    def forward(self, boundary_logits: torch.Tensor, boundary_target: torch.Tensor) -> torch.Tensor:
        return self.dice(boundary_logits, boundary_target) + self.bce(boundary_logits, boundary_target)


class MobileDFUCompoundLoss(nn.Module):
    """
    Full Composite Objective:
    L_total = lambda_dice * L_Dice + lambda_bce * L_BCE + lambda_boundary * L_T-Boundary + lambda_aux * L_Aux
    Default weights: lambda_dice=1.0, lambda_bce=0.5, lambda_boundary=0.5, lambda_aux=0.3.
    """
    def __init__(self, config: Optional[MobileDFULossConfig] = None):
        super().__init__()
        self.config = config or MobileDFULossConfig()
        
        self.dice_loss = SmoothDiceLoss(eps=self.config.smooth_dice_eps)
        self.bce_loss = WeightedBCELoss(pos_weight=self.config.bce_pos_weight)
        self.boundary_loss = ToleranceBoundedBoundaryLoss(tau=self.config.tau, max_cap=self.config.max_cap)
        self.aux_loss = AuxiliaryBoundaryLoss(eps=self.config.smooth_dice_eps, pos_weight=self.config.aux_bce_pos_weight)

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        phi_fp: torch.Tensor,
        phi_fn: torch.Tensor,
        boundary_targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Computes composite loss and individual components.
        Args:
          model_outputs: Dict with "seg_logits" and "boundary_logits".
          targets: [B, 1, H, W] binary ground truth mask {0, 1}.
          phi_fp: [B, 1, H, W] ground-truth exterior distance field.
          phi_fn: [B, 1, H, W] ground-truth interior distance field.
          boundary_targets: [B, 1, H, W] optional morphological 1-pixel boundary target.
                            If None, derived dynamically via morphological erosion of targets.
        """
        seg_logits = model_outputs["seg_logits"]
        boundary_logits = model_outputs["boundary_logits"]

        # 1. Primary Region Losses
        loss_dice = self.dice_loss(seg_logits, targets)
        loss_bce = self.bce_loss(seg_logits, targets)

        # 2. Tolerance-Bounded Boundary Loss (supervised by precomputed fields)
        loss_boundary = self.boundary_loss(seg_logits, phi_fp, phi_fn)

        # 3. Auxiliary Edge Loss
        if boundary_targets is None:
            # Derive morphological 1-pixel boundary on target mask via max pooling
            # boundary = targets - min_pool(targets) == max_pool(1-targets)
            eroded = -F.max_pool2d(-targets, kernel_size=3, stride=1, padding=1)
            boundary_targets = (targets - eroded).clamp(0.0, 1.0)
            
        loss_aux = self.aux_loss(boundary_logits, boundary_targets)

        # 4. Total Weighted Composite Loss
        total_loss = (
            self.config.lambda_dice * loss_dice +
            self.config.lambda_bce * loss_bce +
            self.config.lambda_boundary * loss_boundary +
            self.config.lambda_aux * loss_aux
        )

        return {
            "total_loss": total_loss,
            "loss_dice": loss_dice,
            "loss_bce": loss_bce,
            "loss_boundary": loss_boundary,
            "loss_aux": loss_aux
        }
