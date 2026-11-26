"""
Uncertainty inference functions for MobileDFU-Net.

Implements:
- MC Dropout stochastic forward passes (corrected protocol)
- Predictive entropy (aleatory uncertainty)
- Mutual information (epistemic uncertainty)
- Horizontal-flip test-time augmentation (TTA)

All functions preserve model state, keep BN in eval mode,
and restore original module training flags in finally blocks.
"""

import math
from typing import Dict, Tuple, List
import torch
import torch.nn as nn


def _clamp_tiny_negative_mi(
    value: torch.Tensor, tolerance: float = 1e-7
) -> torch.Tensor:
    """Clamp tiny negative mutual-information values caused by round-off.

    Negative values greater than or equal to ``-tolerance`` are clamped to
    zero. Values strictly below ``-tolerance`` are preserved so that genuine
    inconsistencies remain detectable. Zero and positive values are returned
    unchanged.

    Uses tensor-safe operations and preserves the input shape and dtype.

    Args:
        value: Mutual-information value(s) as a torch.Tensor.
        tolerance: Non-negative clamp tolerance (default 1e-7).

    Raises:
        TypeError: If ``value`` is not a torch.Tensor.
        ValueError: If ``tolerance`` is not finite or is negative.
    """
    if not torch.is_tensor(value):
        raise TypeError(
            f"_clamp_tiny_negative_mi expected a torch.Tensor, "
            f"but got {type(value).__name__}."
        )
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError(
            f"_clamp_tiny_negative_mi requires a finite, non-negative tolerance, "
            f"but got tolerance={tolerance}."
        )

    tiny_negative = (value < 0) & (value >= -tolerance)
    return torch.where(tiny_negative, torch.zeros_like(value), value)


def _get_dropout_modules(model: nn.Module) -> list:
    """Find the four designated Dropout2d modules in the encoder.
    
    Requires the four explicitly designated encoder modules:
    dropout1, dropout2, dropout3, and dropout4 (model.encoder pattern).
    
    Validates that all four exist and are nn.Dropout2d.
    Raises a clear ValueError if any is missing or incorrectly configured.
    Does not silently fall back to arbitrary modules found elsewhere.
    
    Returns exactly 4 nn.Dropout2d modules.
    """
    # Try the MobileDFU-Net pattern: model.encoder.dropout1/2/3/4
    try:
        modules = [
            model.encoder.dropout1,
            model.encoder.dropout2,
            model.encoder.dropout3,
            model.encoder.dropout4,
        ]
        # Verify they are all Dropout2d
        for i, m in enumerate(modules):
            if not isinstance(m, nn.Dropout2d):
                raise ValueError(
                    f"Module {i} is not nn.Dropout2d (got {type(m).__name__}); "
                    "the encoder dropout modules were not found or were reconfigured. "
                    "Ensure the model has model.encoder.dropout1 through dropout4 "
                    "as nn.Dropout2d modules."
                )
        return modules
    except AttributeError:
        # model.encoder exists but dropout attributes are missing
        raise ValueError(
            "Could not find encoder dropout modules. "
            "The model must have model.encoder.dropout1, dropout2, dropout3, "
            "and dropout4 as nn.Dropout2d modules."
        )


def mc_dropout_inference(
    model: nn.Module,
    x: torch.Tensor,
    T: int = 15,
) -> Dict[str, torch.Tensor]:
    """
    Run T stochastic forward passes with corrected MC Dropout protocol.

    Corrected protocol (single inference loop, model globally eval, only 4
    designated Dropout2d modules activated individually):

    1. Validate that T >= 2.
    2. Record the original `.training` flag of every module BEFORE changing
       model mode.
    3. Enter a try/finally block that covers mode changes, dropout discovery
       and inference.
    4. Set the model globally to evaluation mode.
    5. Identify and activate only the four designated Dropout2d modules
       by setting each to training=True (model stays in eval mode).
    6. Keep every BatchNorm module in evaluation mode during all stochastic
       passes.
    7. Use torch.no_grad() context.
    8. Run exactly T forward passes using model(x, return_auxiliary=False).
    9. For every pass:
       - Obtain segmentation logits.
       - Calculate probabilities using sigmoid.
       - Accumulate probabilities for mean computation.
       - Accumulate raw logits for mean logits computation.
       - Calculate per-pixel binary entropy from that pass's probabilities
         and accumulate it.
    10. Calculate mean probability from the accumulated probabilities.
    11. Calculate predictive entropy from the mean probability.
    12. Calculate average per-pass entropy from the accumulated per-pixel
        entropy.
    13. Calculate per-pixel mutual information:

        MI = H(mean_probability) - mean(H(pass_probability))

    14. Calculate image-level summary values.
    15. Restore every original module `.training` flag exactly in finally,
        including if dropout discovery or inference raises an exception.

    IMPORTANT:
    - Exactly T stochastic forward passes are performed.
    - Dropout modules are activated throughout all T passes.
    - BatchNorm remains in evaluation mode throughout.
    - No optimizer steps or parameter modifications.
    - mean_logits is the arithmetic mean of the T raw logits.
    - Output shapes and return keys are preserved.
    - Avoid unnecessary retention of all T full-resolution predictions;
      use running accumulators where possible.

    Parameters:
        model (nn.Module): MobileDFU-Net model. The function will set it
          to eval() at the start, temporarily activate 4 Dropout2d modules,
          and restore original state in finally.
        x (torch.Tensor): Input tensor [B, 3, 512, 512].
        T (int): Number of stochastic forward passes. Must be >= 2.

    Returns:
        Dict with keys:
            - "mean_probability": [B, 1, 512, 512] mean probability across T passes
            - "mean_logits": [B, 1, 512, 512] arithmetic mean of T raw logits
            - "predictive_entropy_map": [B, 1, 512, 512] per-pixel predictive entropy
            - "image_predictive_entropy": scalar mean predictive entropy across pixels
            - "mutual_information_map": [B, 1, 512, 512] per-pixel mutual information
            - "image_mutual_information": scalar mean mutual information across pixels
            - "num_passes": int number of forward passes (=T)

    Raises:
        ValueError: If T < 2.
    """
    if T < 2:
        raise ValueError(f"T must be >= 2 for MC inference, got T={T}")

    # Step 1 (critical): Record original .training flags BEFORE changing model mode
    training_snapshots = {}
    for m in model.modules():
        training_snapshots[id(m)] = m.training

    # Step 2 (critical): Immediately enter try block after snapshot
    try:
        # Set model to evaluation mode to freeze BN running statistics
        model.eval()

        # Identify and activate only the four designated dropout modules
        dropout_modules = _get_dropout_modules(model)
        for m in dropout_modules:
            m.training = True

        # Accumulators for single-pass loop
        sum_probs = None         # accumulated sigmoid probabilities
        sum_logits = None        # accumulated raw logits
        sum_entropy_pixel = 0.0  # accumulated per-pixel entropy

        with torch.no_grad():
            for t in range(T):
                # Forward pass — returns seg_logits tensor when return_auxiliary=False
                seg_logits = model(x, return_auxiliary=False)  # [B, 1, 512, 512]

                # Apply sigmoid separately to each pass
                probs = torch.sigmoid(seg_logits)  # [B, 1, 512, 512]

                # Accumulate probabilities for mean computation
                if sum_probs is None:
                    sum_probs = probs
                else:
                    sum_probs = sum_probs + probs

                # Accumulate raw logits for mean logits computation
                if sum_logits is None:
                    sum_logits = seg_logits
                else:
                    sum_logits = sum_logits + seg_logits

                # Per-pass binary entropy (clipped for numerical stability)
                eps = 1e-7
                p_clipped = probs.clip(eps, 1.0 - eps)
                entropy_pass = -(p_clipped * torch.log(p_clipped) +
                                 (1.0 - p_clipped) * torch.log(1.0 - p_clipped))
                # Accumulate per-pixel entropy (keep full spatial shape)
                sum_entropy_pixel = sum_entropy_pixel + entropy_pass

        # Mean probability across T passes (probability-average approach)
        mean_prob = sum_probs / T  # [B, 1, 512, 512]

        # Arithmetic mean of the T raw logits (NOT inverse sigmoid of mean probability)
        mean_logits = sum_logits / T  # [B, 1, 512, 512]

        # Predictive entropy from mean probability (aleatory uncertainty)
        p_clipped_mean = mean_prob.clip(eps, 1.0 - eps)
        predictive_entropy_map = -(p_clipped_mean * torch.log(p_clipped_mean) +
                                   (1.0 - p_clipped_mean) * torch.log(1.0 - p_clipped_mean))
        # Image-level: mean across all pixels and batch dims
        image_predictive_entropy = predictive_entropy_map.mean()

        # Average per-pass entropy across T passes and pixels
        # sum_entropy_pixel shape: [B, 1, 512, 512] (per-pixel, accumulated over T passes)
        mean_entropy_pixel = sum_entropy_pixel / T  # [B, 1, 512, 512]

        # Per-pixel mutual information: I = H_pred - mean_H_pass
        mutual_information_map = predictive_entropy_map - mean_entropy_pixel

        # Image-level mutual information: mean across all elements
        # Clamp only tiny negatives within the documented 1e-7 tolerance;
        # preserve substantial negative values so errors remain detectable.
        image_mutual_information = _clamp_tiny_negative_mi(
            mutual_information_map.mean()
        )

        # Build result dictionary
        result = {
            "mean_probability": mean_prob,
            "mean_logits": mean_logits,
            "predictive_entropy_map": predictive_entropy_map,
            "image_predictive_entropy": image_predictive_entropy,
            "mutual_information_map": mutual_information_map,
            "image_mutual_information": image_mutual_information,
            "num_passes": torch.tensor(T, dtype=torch.int),
        }

        return result

    finally:
        # Step 6 (critical): Restore every module's original training flag exactly.
        # We do NOT simply force all modules to evaluation mode; each module
        # gets back exactly the .training state it had before the function started.
        # This must happen even if an exception was raised above.
        for m_id, original_training in training_snapshots.items():
            for m in model.modules():
                if id(m) == m_id:
                    m.training = original_training
                    break


def predictive_entropy(mean_probability: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute binary predictive entropy from mean probability.

    Uses the probability-average approach: clip p to [eps, 1-eps],
    then compute -p*log(p) - (1-p)*log(1-p).

    This is the aleatory uncertainty measure: high entropy when the model
    is uncertain about its prediction (p near 0.5), low entropy when
    confident (p near 0 or 1).

    Parameters:
        mean_probability: Tensor of predicted probabilities, any shape.
        eps: Clipping epsilon for numerical stability (default 1e-7).

    Returns:
        Element-wise entropy with same shape as mean_probability,
        values in [0, log(2)] ≈ [0, 0.693].
    """
    p_clipped = mean_probability.clip(eps, 1.0 - eps)
    return -(p_clipped * torch.log(p_clipped) + (1.0 - p_clipped) * torch.log(1.0 - p_clipped))


def mutual_information(
    pass_probabilities: torch.Tensor, eps: float = 1e-7
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mutual information (epistemic uncertainty) from pass probabilities.

    Given T stochastic forward passes producing probabilities per pixel,
    computes the mutual information estimate:

        I = H_pred - H_pass

    where:
        - H_pred = -p_mean*log(p_mean) - (1-p_mean)*log(1-p_mean)
            is the predictive entropy from the mean probability
        - H_pass = (1/T) * sum_t [-p_t*log(p_t) - (1-p_t)*log(1-p_t)]
            is the average per-pass entropy

    A positive MI indicates epistemic uncertainty (model uncertainty
    due to limited data/weights); MI ≈ 0 means the model's predictions
    are consistent across stochastic passes.

    Numerical rounding may produce tiny negative MI values (e.g., -1e-10);
    these are clamped to zero with a documented note that this is
    numerical protection, not calibration.

    The standalone mutual-information function must continue to calculate
    from actual per-pass probabilities with shape [T, B, 1, H, W].

    Input validation: ALL CHECKS ARE PERFORMED BEFORE ANY TENSOR REDUCTIONS,
    CLIPPING, LOGARITHMS, OR ENTROPY CALCULATIONS.

    Requirements:
    - pass_probabilities must be a torch.Tensor.
    - Tensor must have exactly five dimensions: [T, B, 1, H, W].
    - Every dimension must be non-empty.
    - T must be >= 2.
    - Channel dimension must equal 1.
    - All probabilities must be finite (no NaN or Inf).
    - Every probability must lie in [0, 1].
    - eps must be finite and strictly between 0 and 0.5.

    Raises:
        TypeError: If pass_probabilities is not a torch.Tensor.
        ValueError: If tensor does not have exactly 5 dimensions.
        ValueError: If any dimension is empty.
        ValueError: If T < 2.
        ValueError: If channel dimension != 1.
        ValueError: If non-finite probabilities are found.
        ValueError: If probabilities outside [0, 1] range.
        ValueError: If eps is not finite or not in (0, 0.5).

    The standalone mutual-information function must continue to calculate
    from actual per-pass probabilities with shape [T, B, 1, H, W].

    Example:
        >>> # With T=5 passes, probability tensor of shape [5, B, 1, H, W]
        >>> mi_image, mi_pixel = mutual_information(pass_probs, eps=1e-7)
    """
    # === INPUT VALIDATION: All checks before any tensor operations ===

    # Require pass_probabilities to be a torch.Tensor
    if not torch.is_tensor(pass_probabilities):
        raise TypeError(
            f"mutual_inference expected a torch.Tensor, but got {type(pass_probabilities).__name__}."
        )

    # Require exactly five dimensions: [T, B, 1, H, W]
    if pass_probabilities.dim() != 5:
        raise ValueError(
            f"mutual_information expected a tensor with exactly 5 dimensions [T, B, 1, H, W], "
            f"but got {pass_probabilities.dim()} dimensions."
        )

    # Get tensor shape components
    T, B, C, H, W = pass_probabilities.shape

    # Require every dimension to be non-empty
    if T < 1 or B < 1 or C < 1 or H < 1 or W < 1:
        raise ValueError(
            f"mutual_information encountered an empty dimension in tensor shape {pass_probabilities.shape}. "
            "All dimensions [T, B, 1, H, W] must be non-empty."
        )

    # Require T >= 2
    if T < 2:
        raise ValueError(
            f"mutual_information requires T >= 2 stochastic passes, but got T={T}."
        )

    # Require channel dimension equals 1
    if C != 1:
        raise ValueError(
            f"mutual_information expected channel dimension 1, but got C={C} in tensor shape {pass_probabilities.shape}."
        )

    # Require all probabilities to be finite (no NaN or Inf)
    if not torch.isfinite(pass_probabilities).all():
        raise ValueError(
            "mutual_information received non-finite probabilities (NaN or Inf). "
            "Please check the pass probability inputs."
        )

    # Require every probability to lie in [0, 1]
    if (pass_probabilities < 0).any() or (pass_probabilities > 1).any():
        raise ValueError(
            f"mutual_information received probabilities outside [0, 1] range. "
            f"Found min={pass_probabilities.min().item()}, max={pass_probabilities.max().item()}"
        )

    # Require eps to be finite and strictly between 0 and 0.5
    if not math.isfinite(eps) or eps <= 0 or eps >= 0.5:
        raise ValueError(
            f"mutual_information requires eps to be finite and strictly between 0 and 0.5, "
            f"but got eps={eps}."
        )

    # === Now safe to compute: mean probability, entropy, and MI ===

    # Mean probability across passes (probability-average approach)
    mean_prob = pass_probabilities.mean(dim=0)  # [B, 1, H, W]

    # Predictive entropy from mean probability
    H_pred = predictive_entropy(mean_prob, eps)  # [B, 1, H, W]

    # Average per-pass entropy
    # Per-pass entropy: -p*log(p) - (1-p)*log(1-p), then mean over T passes
    # Clip for numerical stability per pass
    pass_entropy = -(pass_probabilities.clip(eps, 1.0 - eps) *
                     torch.log(pass_probabilities.clip(eps, 1.0 - eps)) +
                     (1.0 - pass_probabilities.clip(eps, 1.0 - eps)) *
                     torch.log(1.0 - pass_probabilities.clip(eps, 1.0 - eps)))
    # Shape [T, B, 1, H, W] -> mean over dim 0 -> [B, 1, H, W]
    H_pass = pass_entropy.mean(dim=0)  # [B, 1, H, W]

    # Mutual information: I = H_pred - H_pass
    per_pixel_mi = H_pred - H_pass

    # Image-level: mean across batch and spatial dims.
    # Clamp only tiny negatives within the documented 1e-7 tolerance;
    # preserve substantial negative values so errors remain detectable.
    image_mi = _clamp_tiny_negative_mi(per_pixel_mi.mean())

    return image_mi, per_pixel_mi


def tta_horizontal_flip(
    model: nn.Module, x: torch.Tensor, return_auxiliary: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Test-time augmentation with horizontal flip only.

    For clinical DFU images, the only safely invertible transformation
    is horizontal flip (left-right reflection). No rotation, scaling,
    or color jitter is applied to avoid distorting wound geometry.

    Protocol:
        1. Snapshot all original module training flags before changing model mode.
        2. Set the model to evaluation mode.
        3. Use torch.no_grad().
        4. Run inference on the original image.
        5. Run inference on the horizontally flipped image.
        6. Invert the flipped prediction along width dimension.
        7. Average the aligned logits.
        8. Apply sigmoid to the averaged logits.
        9. Restore all original module states in finally.

    Parameters:
        model: MobileDFU-Net model. Original module training states are
            snapshot and restored; model is set to eval() during the pass.
        x: Input tensor [B, 3, 512, 512].
        return_auxiliary: If True, handle dictionary outputs correctly.
            Otherwise return only segmentation logits.

    Returns:
        Dict with keys:
            - "mean_logits": [B, 1, 512, 512] averaged logits
            - "mean_probability": [B, 1, 512, 512] averaged probabilities

    Note:
        TTA does NOT measure epistemic uncertainty; it is a deterministic
        augmentation for improving segmentation reliability, not for
        quantifying uncertainty sources.

    Raises:
        ValueError: If return_auxiliary=True and the model output dictionary
            does not contain expected keys ('seg_logits').
        ValueError: If original and flipped outputs have incompatible types.
        ValueError: If seg_logits are missing from dictionary outputs.
        ValueError: If original and flipped logits have mismatched shapes.
        ValueError: If logits contain non-finite values (NaN/Inf).
    """
    # Snapshot original training states to restore later
    training_snapshots = {}
    for m in model.modules():
        training_snapshots[id(m)] = m.training

    try:
        # Set model to eval mode for deterministic TTA
        model.eval()

        with torch.no_grad():
            # Forward on original image
            out_orig = model(x, return_auxiliary=return_auxiliary)

            # Forward on horizontally flipped image
            x_flipped = x.flip(dims=[3])
            out_flipped = model(x_flipped, return_auxiliary=return_auxiliary)

            # Validate outputs are compatible
            _validate_tta_outputs(out_orig, out_flipped, return_auxiliary)

            # Flip predictions back (horizontal invert for alignment)
            # Handle both tensor and dictionary outputs
            if isinstance(out_flipped, dict):
                # Dictionary output: flip the seg_logits back and extract tensor
                if 'seg_logits' in out_flipped:
                    out_flipped_seg = out_flipped['seg_logits'].flip(dims=[3])
                    # Use the flipped tensor, not the dict
                    out_flipped_back = out_flipped_seg
                else:
                    # Cannot handle this dictionary format
                    raise ValueError(
                        "Dictionary output does not contain 'seg_logits' key; "
                        "cannot perform horizontal-flip TTA. "
                        "The model must return a tensor or a dictionary "
                        "containing 'seg_logits'."
                    )
            else:
                # Tensor output: flip directly
                out_flipped_back = out_flipped.flip(dims=[3])

            # Average logits
            # If out_orig is also a dict, take seg_logits
            if isinstance(out_orig, dict):
                out_orig_seg = out_orig['seg_logits']
            else:
                out_orig_seg = out_orig

            # Validate shapes match
            if out_orig_seg.shape != out_flipped_back.shape:
                raise ValueError(
                    f"Shape mismatch between original and flipped logits: "
                    f"original {out_orig_seg.shape} vs flipped {out_flipped_back.shape}"
                )

            # Validate logits are finite
            if not torch.isfinite(out_orig_seg).all():
                raise ValueError(
                    "Original logits contain non-finite values (NaN or Inf)."
                )
            if not torch.isfinite(out_flipped_back).all():
                raise ValueError(
                    "Flipped logits contain non-finite values (NaN or Inf)."
                )

            mean_logits = (out_orig_seg + out_flipped_back) / 2
            mean_probability = torch.sigmoid(mean_logits)

            result = {
                "mean_logits": mean_logits,
                "mean_probability": mean_probability,
            }

        return result

    finally:
        # Restore original training states exactly
        for m_id, original_training in training_snapshots.items():
            for m in model.modules():
                if id(m) == m_id:
                    m.training = original_training
                    break


def _validate_tta_outputs(
    out_orig, out_flipped, return_auxiliary: bool
) -> None:
    """Validate TTA outputs for compatibility and consistency.

    Checks:
    - Each output is explicitly a torch.Tensor or a dictionary containing tensor seg_logits.
    - Reject unsupported output objects (list, tuple, string, etc.) with ValueError.
    - seg_logits have 4D shape [B, 1, H, W] with one output channel and non-empty dimensions, including B > 0.
    - Complete shapes match between original and flipped (not just batch).
    - Original and flipped output types are compatible.
    - Logits are finite (no NaN/Inf).

    Raises ValueError with clear messages if any check fails.
    """
    # === Explicitly accept only torch.Tensor or dict ===
    if not isinstance(out_orig, (torch.Tensor, dict)):
        raise ValueError(
            f"TTA original output must be a torch.Tensor or a dictionary containing tensor seg_logits, "
            f"but got {type(out_orig).__name__}."
        )
    if not isinstance(out_flipped, (torch.Tensor, dict)):
        raise ValueError(
            f"TTA flipped output must be a torch.Tensor or a dictionary containing tensor seg_logits, "
            f"but got {type(out_flipped).__name__}."
        )

    # === Require original and flipped output types to be compatible ===
    orig_is_dict = isinstance(out_orig, dict)
    flipped_is_dict = isinstance(out_flipped, dict)

    if orig_is_dict != flipped_is_dict:
        raise ValueError(
            f"Incompatible output types for TTA: original is {'dict' if orig_is_dict else 'tensor'}, "
            f"flipped is {'dict' if flipped_is_dict else 'tensor'}. "
            "Both original and flipped outputs must be of the same type "
            "(both tensors or both dictionaries with tensor-valued seg_logits)."
        )

    # === If dictionary outputs, validate seg_logits presence, type, and shape ===
    if orig_is_dict:
        if 'seg_logits' not in out_orig:
            raise ValueError(
                "Original dictionary output does not contain 'seg_logits' key. "
                "TTA requires dictionary outputs to contain tensor-valued 'seg_logits'."
            )
        if 'seg_logits' not in out_flipped:
            raise ValueError(
                "Flipped dictionary output does not contain 'seg_logits' key. "
                "TTA requires dictionary outputs to contain tensor-valued 'seg_logits'."
            )

        # Validate that seg_logits values are tensors (reject lists, strings, etc.)
        orig_seg = out_orig['seg_logits']
        flipped_seg = out_flipped['seg_logits']

        if not torch.is_tensor(orig_seg):
            raise ValueError(
                "Original dictionary 'seg_logits' value is not a tensor. "
                "TTA requires seg_logits to be tensor-valued in the dictionary."
            )
        if not torch.is_tensor(flipped_seg):
            raise ValueError(
                "Flipped dictionary 'seg_logits' value is not a tensor. "
                "TTA requires seg_logits to be tensor-valued in the dictionary."
            )

        # Validate that seg_logits have 4D shape [B, 1, H, W]
        # with one output channel and non-empty dimensions, including B > 0
        if orig_seg.dim() != 4 or flipped_seg.dim() != 4:
            raise ValueError(
                f"Expected 4D seg_logits tensors [B, 1, H, W], "
                f"but got original with {orig_seg.dim()}D and flipped with {flipped_seg.dim()}D. "
                "All dimensions [B, 1, H, W] must be non-empty with B > 0."
            )

        # Check exactly one output channel (shape[1] == 1)
        if orig_seg.shape[1] != 1 or flipped_seg.shape[1] != 1:
            raise ValueError(
                f"Expected seg_logits with 1 output channel (shape[1] == 1), "
                f"but original has {orig_seg.shape[1]} channels and flipped has {flipped_seg.shape[1]} channels."
            )

        # Validate spatial dimensions are non-empty
        if orig_seg.shape[2] < 1 or orig_seg.shape[3] < 1:
            raise ValueError(
                f"Expected non-empty spatial dimensions in seg_logits, "
                f"but original has H={orig_seg.shape[2]}, W={orig_seg.shape[3]}."
            )
        if flipped_seg.shape[2] < 1 or flipped_seg.shape[3] < 1:
            raise ValueError(
                f"Expected non-empty spatial dimensions in seg_logits, "
                f"but flipped has H={flipped_seg.shape[2]}, W={flipped_seg.shape[3]}."
            )

        # Validate batch dimension is non-empty (B > 0)
        if orig_seg.shape[0] < 1 or flipped_seg.shape[0] < 1:
            raise ValueError(
                f"Expected non-empty batch dimension (B > 0) in seg_logits, "
                f"but original has B={orig_seg.shape[0]}, flipped has B={flipped_seg.shape[0]}."
            )

        # Validate complete shapes match (not just batch dimension)
        if orig_seg.shape != flipped_seg.shape:
            raise ValueError(
                f"Shape mismatch between original and flipped seg_logits: "
                f"original {orig_seg.shape} vs flipped {flipped_seg.shape}. "
                "Complete shapes must match exactly."
            )
    else:
        # Tensor outputs - validate dimensions, channel, non-empty, complete shape match
        if out_orig.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor logits [B, 1, H, W], but got {out_orig.dim()}D tensor. "
                "All dimensions [B, 1, H, W] must be non-empty with B > 0."
            )
        if out_flipped.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor logits [B, 1, H, W], but got {out_flipped.dim()}D tensor. "
                "All dimensions [B, 1, H, W] must be non-empty with B > 0."
            )

        # Validate exactly one output channel (shape[1] == 1)
        if out_orig.shape[1] != 1 or out_flipped.shape[1] != 1:
            raise ValueError(
                f"Expected tensor logits with 1 output channel (shape[1] == 1), "
                f"but original has {out_orig.shape[1]} channels and flipped has {out_flipped.shape[1]} channels."
            )

        # Validate spatial dimensions are non-empty
        if out_orig.shape[2] < 1 or out_orig.shape[3] < 1:
            raise ValueError(
                f"Expected non-empty spatial dimensions in tensor logits, "
                f"but original has H={out_orig.shape[2]}, W={out_orig.shape[3]}."
            )
        if out_flipped.shape[2] < 1 or out_flipped.shape[3] < 1:
            raise ValueError(
                f"Expected non-empty spatial dimensions in tensor logits, "
                f"but flipped has H={out_flipped.shape[2]}, W={out_flipped.shape[3]}."
            )
        # Validate batch dimension is non-empty (B > 0) -- added per audit
        if out_orig.shape[0] < 1 or out_flipped.shape[0] < 1:
            raise ValueError(
                f"Expected non-empty batch dimension (B > 0) in tensor logits, "
                f"but original has B={out_orig.shape[0]}, flipped has B={out_flipped.shape[0]}."
            )

        # Validate complete shapes match (not just batch dimension)
        if out_orig.shape != out_flipped.shape:
            raise ValueError(
                f"Shape mismatch between original and flipped logits: "
                f"original {out_orig.shape} vs flipped {out_flipped.shape}. "
                "Complete shapes must match exactly."
            )

    # === Validate logits are finite (no NaN/Inf) ===
    if orig_is_dict:
        if not torch.isfinite(orig_seg).all():
            raise ValueError("Original dictionary seg_logits contain non-finite values (NaN or Inf).")
        if not torch.isfinite(flipped_seg).all():
            raise ValueError("Flipped dictionary seg_logits contain non-finite values (NaN or Inf).")
    else:
        if not torch.isfinite(out_orig).all():
            raise ValueError("Original tensor logits contain non-finite values (NaN or Inf).")
        if not torch.isfinite(out_flipped).all():
            raise ValueError("Flipped tensor logits contain non-finite values (NaN or Inf).")

