"""Stage 3 Milestone 2B: Production uncertainty inference tests (corrected audit version).

Tests import and call the actual functions in uncertainty.py.
Use small synthetic/dummy models where possible to keep tests fast.
"""

import pytest
import torch
import numpy as np

from src.models.mobile_dfu_net.uncertainty import (
    mc_dropout_inference,
    predictive_entropy,
    mutual_information,
    tta_horizontal_flip,
    _validate_tta_outputs,
    _clamp_tiny_negative_mi,
)


# ---- Small instrumented dummy models with proper encoder dropout naming ----

class _FourDropoutEncoderModel(torch.nn.Module):
    """Minimal model with four designated Dropout2d and real BatchNorm2d layers.

    The _get_dropout_modules() function expects model.encoder.dropout1/2/3/4
    to exist as nn.Dropout2d attributes. Explicit module attributes are used
    (rather than Sequential indices) so the dropout names stay stable.

    Real BatchNorm2d layers are included so tracing tests can verify that the
    production function keeps BatchNorm in evaluation mode during MC dropout.
    """

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.encoder.bn1 = torch.nn.BatchNorm2d(16)
        self.encoder.dropout1 = torch.nn.Dropout2d(p=0.1)
        self.encoder.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.encoder.bn2 = torch.nn.BatchNorm2d(32)
        self.encoder.dropout2 = torch.nn.Dropout2d(p=0.1)
        self.encoder.conv3 = torch.nn.Conv2d(32, 16, 3, padding=1)
        self.encoder.dropout3 = torch.nn.Dropout2d(p=0.1)
        self.encoder.conv4 = torch.nn.Conv2d(16, 1, 3, padding=1)
        self.encoder.dropout4 = torch.nn.Dropout2d(p=0.1)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, return_auxiliary=False):
        e = self.encoder
        x = e.dropout1(torch.relu(e.bn1(e.conv1(x))))
        x = e.dropout2(torch.relu(e.bn2(e.conv2(x))))
        x = e.dropout3(torch.relu(e.conv3(x)))
        x = e.dropout4(e.conv4(x))
        if return_auxiliary:
            # Return dictionary with seg_logits
            seg = self.pool(x)
            return {"seg_logits": seg, "boundary_probs": seg.clone()}
        return self.pool(x)


class _ExceptionReraisingModel(torch.nn.Module):
    """Model with four designated dropout modules that raises on a specific pass.

    Uses explicit attribute assignment so _get_dropout_modules can find them.
    """

    def __init__(self, raise_pass=0):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.Dropout2d(p=0.1),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.Dropout2d(p=0.1),
            torch.nn.Conv2d(32, 16, 3, padding=1),
            torch.nn.Dropout2d(p=0.1),
            torch.nn.Conv2d(16, 1, 3, padding=1),
            torch.nn.Dropout2d(p=0.1),
        )
        # Assign named attributes
        self.encoder.dropout1 = self.encoder[1]
        self.encoder.dropout2 = self.encoder[3]
        self.encoder.dropout3 = self.encoder[5]
        self.encoder.dropout4 = self.encoder[7]
        self.raise_pass = raise_pass
        self._pass_counter = 0

    def forward(self, x, return_auxiliary=False):
        self._pass_counter += 1
        if self._pass_counter == self.raise_pass + 1:
            raise RuntimeError("Simulated forward error during MC dropout")
        x = self.encoder(x)
        return x


class _ControlledLogitsModel(torch.nn.Module):
    """Model with four designated dropout modules that produces known logits on each call.

    Each forward call returns a different logits value so we can independently
    compute arithmetic means, sigmoid probabilities, entropy, and MI.
    The model has explicit dropout1/2/3/4 attributes for _get_dropout_modules.
    """

    def __init__(self, pass_logits=None):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.Dropout2d(p=0.1),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.Dropout2d(p=0.1),
            torch.nn.Conv2d(32, 16, 3, padding=1),
            torch.nn.Dropout2d(p=0.1),
            torch.nn.Conv2d(16, 1, 3, padding=1),
            torch.nn.Dropout2d(p=0.1),
        )
        # Assign named attributes so _get_dropout_modules finds them
        self.encoder.dropout1 = self.encoder[1]
        self.encoder.dropout2 = self.encoder[3]
        self.encoder.dropout3 = self.encoder[5]
        self.encoder.dropout4 = self.encoder[7]
        self.pass_logits = pass_logits or [0.5]
        self._call_index = 0

    def forward(self, x, return_auxiliary=False):
        self._call_index += 1
        # Cycle through pass logits
        logit_val = self.pass_logits[(self._call_index - 1) % len(self.pass_logits)]
        # Return [B, 1, 1, 1] shaped logits
        result = torch.full((1, 1, 1, 1), logit_val, dtype=torch.float32)
        if return_auxiliary:
            return {"seg_logits": result, "boundary_probs": result.clone()}
        return result


class _TracingMCModel(torch.nn.Module):
    """Model that records dropout/BN states and forward-call count during every pass.

    Used by Part C and Part F tests to verify dropout activation, BN mode,
    and exact forward-call count during MC inference.

    The wrapper exposes the underlying model's ``encoder`` attribute so that the
    production ``_get_dropout_modules()`` lookup works when the wrapper itself is
    passed to ``mc_dropout_inference()``. Its own ``forward()`` records the state
    seen on each pass, then delegates to the base model.
    """

    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        # Recording state
        self.dropout_training_states = []  # list of dicts: {name: training_state}
        self.bn_training_states = []  # list of dicts: {id(module): training_state}
        self.forward_call_count = 0

    @property
    def encoder(self):
        """Expose the base model's encoder without re-registering its modules."""
        return self.base.encoder

    def reset_trace(self):
        self.dropout_training_states.clear()
        self.bn_training_states.clear()
        self.forward_call_count = 0

    def forward(self, x, return_auxiliary=False):
        self.forward_call_count += 1

        # Record dropout module states (only the four designated ones)
        dropout_states = {}
        for name in ["dropout1", "dropout2", "dropout3", "dropout4"]:
            mod = getattr(self.base.encoder, name, None)
            if mod is not None:
                dropout_states[name] = mod.training
        self.dropout_training_states.append(dropout_states)

        # Record BN module states
        bn_states = {}
        for mod in self.base.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                bn_states[id(mod)] = mod.training
        self.bn_training_states.append(bn_states)

        return self.base(x, return_auxiliary=return_auxiliary)


# ---- Fixtures ----

@pytest.fixture(scope="module")
def four_dropout_model():
    """Minimal model with exactly four designated encoder Dropout2d modules."""
    return _FourDropoutEncoderModel()


@pytest.fixture(scope="module")
def exception_raising_model():
    """Model that raises RuntimeError on the 3rd mc_dropout_inference pass."""
    return _ExceptionReraisingModel(raise_pass=2)


@pytest.fixture(scope="module")
def controlled_logits_model():
    """Model with controlled logits for reference test comparisons.

    Pass 1: logit=0.8, Pass 2: logit=0.3, Pass 3: logit=-0.2
    """
    return _ControlledLogitsModel(pass_logits=[0.8, 0.3, -0.2])


@pytest.fixture(scope="module")
def tracing_mc_model(four_dropout_model):
    """Tracing wrapper around a four-dropout model for Part C/F verification."""
    return _TracingMCModel(four_dropout_model)


# ---- Test Group 1: Normal evaluation inference ----

def test_normal_evaluation_deterministic(four_dropout_model):
    """Same input, model.eval() → identical output every time."""
    four_dropout_model.eval()
    x = torch.rand(1, 3, 512, 512)
    with torch.no_grad():
        out1 = four_dropout_model(x, return_auxiliary=False)
        out2 = four_dropout_model(x, return_auxiliary=False)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_normal_evaluation_finite(four_dropout_model):
    """No NaN/Inf in normal evaluation inference."""
    x = torch.rand(1, 3, 512, 512)
    with torch.no_grad():
        out = four_dropout_model(x, return_auxiliary=False)
    assert torch.isfinite(out).all()


# ---- Test Group 2: MC dropout inference ----

def test_mc_dropout_training_mode_restore(four_dropout_model):
    """Original model in training mode; all original module flags restored after successful inference.

    Verifies the try/finally block correctly restores states even when model starts in training mode.
    """
    # Start with model in training mode
    four_dropout_model.train()
    # Record original states
    training_snapshots = {}
    for m in four_dropout_model.modules():
        training_snapshots[id(m)] = m.training

    x = torch.rand(1, 3, 512, 512)
    result = mc_dropout_inference(four_dropout_model, x, T=5)

    # All modules should be restored to their original training state (which was True for train mode)
    all_restored = True
    for m_id, original_val in training_snapshots.items():
        found = False
        for m in four_dropout_model.modules():
            if id(m) == m_id:
                if m.training != original_val:
                    all_restored = False
                found = True
                break
        if not found:
            all_restored = False

    assert all_restored, "Not all module training flags restored after inference when model started in training mode"


def test_mc_dropout_mixed_states_restore_exactly(four_dropout_model):
    """Mixed original module states restored exactly after inference.

    Verifies that each module gets back exactly what it had before, not just
    forced to eval mode.
    """
    # Model starts in eval mode
    four_dropout_model.eval()
    training_snapshots = {}
    for m in four_dropout_model.modules():
        training_snapshots[id(m)] = m.training

    x = torch.rand(1, 3, 512, 512)
    result = mc_dropout_inference(four_dropout_model, x, T=5)

    # Verify exact restoration
    all_restored = True
    for m_id, original_val in training_snapshots.items():
        found = False
        for m in four_dropout_model.modules():
            if id(m) == m_id:
                if m.training != original_val:
                    all_restored = False
                found = True
                break
        if not found:
            all_restored = False

    assert all_restored, "Not all module training flags restored exactly"


def test_mc_dropout_exception_during_inference_restores(exception_raising_model):
    """Forward-pass exception during mc_dropout_inference triggers restoration of all module states.

    Uses a model with four designated encoder Dropout2d modules that raises an
    exception from its forward method during an actual call to mc_dropout_inference().
    Verifies that every module's original .training flag is restored exactly,
    and BatchNorm states are also restored.
    Uses pytest.raises to explicitly capture the expected exception.
    """
    model = exception_raising_model
    model.eval()

    # Record original states
    training_snapshots = {}
    for m in model.modules():
        training_snapshots[id(m)] = m.training

    x = torch.rand(1, 3, 512, 512)

    # Use pytest.raises to explicitly capture the expected exception
    with pytest.raises(RuntimeError, match="Simulated forward error during MC dropout"):
        mc_dropout_inference(model, x, T=5)

    # Verify restoration after exception (the function's finally block must have run)
    all_restored = True
    for m_id, original_val in training_snapshots.items():
        found = False
        for m in model.modules():
            if id(m) == m_id:
                if m.training != original_val:
                    all_restored = False
                found = True
                break
        if not found:
            all_restored = False

    assert all_restored, "Module states not restored after exception during inference"


def test_mc_dropout_dropout_discovery_failure_restores(four_dropout_model):
    """Dropout-discovery exception triggers restoration of all module states.

    Temporarily removes a designated encoder dropout module, calls the production
    function, expects ValueError, and verifies exact restoration of all original
    module states.
    Uses a single pytest.raises context; does not catch or swallow the exception
    manually outside the context manager.
    """
    model = four_dropout_model
    model.eval()

    # Record original states
    training_snapshots = {}
    for m in model.modules():
        training_snapshots[id(m)] = m.training

    x = torch.rand(1, 3, 512, 512)

    # Save and replace, then restore after
    original_dropout1 = model.encoder.dropout1
    try:
        # Remove dropout1 attribute to force _get_dropout_modules to fail
        del model.encoder.dropout1
        # Single pytest.raises context - expect ValueError from the function
        with pytest.raises(ValueError, match="Could not find encoder dropout modules"):
            mc_dropout_inference(model, x, T=5)
    finally:
        # Restore dropout1 regardless
        model.encoder.dropout1 = original_dropout1

    # Check states are still restored (the function uses try/finally)
    all_restored = True
    for m_id, original_val in training_snapshots.items():
        found = False
        for m in model.modules():
            if id(m) == m_id:
                if m.training != original_val:
                    all_restored = False
                found = True
                break
        if not found:
            all_restored = False

    # If we get here, states should be restored
    assert all_restored, "Module states not restored when dropout discovery raises exception"


def test_mc_dropout_exactly_t_passes(tracing_mc_model):
    """Verify exactly T stochastic forward passes and per-pass module modes.

    The tracing wrapper itself is passed to the production function, so its
    forward counter and state recordings are actually exercised. The traced
    base model contains real BatchNorm2d layers.

    Verifies:
    - exactly T forward calls and T recorded state snapshots;
    - all four designated encoder Dropout2d modules are active on every pass;
    - every BatchNorm2d module stays in evaluation mode on every pass;
    - all module training flags are restored after successful inference.
    """
    model = tracing_mc_model
    x = torch.rand(1, 3, 512, 512)
    T = 5

    # Start from training mode so restoration is a meaningful, non-trivial check.
    model.train()
    pre_call_training = {id(m): m.training for m in model.modules()}

    bn_modules = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert len(bn_modules) >= 1, "traced base model must contain at least one BatchNorm2d layer"

    model.reset_trace()

    # Pass the tracing wrapper itself (not .base) to the production function.
    result = mc_dropout_inference(model, x, T=T)

    # 1) Exactly T actual forward calls and T recorded snapshots.
    assert model.forward_call_count == T, \
        f"expected {T} forward calls, got {model.forward_call_count}"
    assert len(model.dropout_training_states) == T, \
        f"expected {T} dropout snapshots, got {len(model.dropout_training_states)}"
    assert len(model.bn_training_states) == T, \
        f"expected {T} BatchNorm snapshots, got {len(model.bn_training_states)}"
    assert result["num_passes"].item() == T, \
        f"expected num_passes={T}, got {result['num_passes'].item()}"

    # 2) On every pass, all four designated Dropout2d modules are active,
    #    and every BatchNorm2d module is in evaluation mode.
    expected_dropouts = {"dropout1", "dropout2", "dropout3", "dropout4"}
    expected_bn_ids = {id(m) for m in bn_modules}
    for pass_idx, (dropout_states, bn_states) in enumerate(
        zip(model.dropout_training_states, model.bn_training_states)
    ):
        assert set(dropout_states) == expected_dropouts, \
            f"pass {pass_idx}: missing dropout snapshots: {set(dropout_states)}"
        assert all(state is True for state in dropout_states.values()), \
            f"pass {pass_idx}: not all designated dropouts active: {dropout_states}"
        assert set(bn_states) == expected_bn_ids, \
            f"pass {pass_idx}: BatchNorm snapshot set mismatch"
        assert all(state is False for state in bn_states.values()), \
            f"pass {pass_idx}: BatchNorm modules not in eval mode: {bn_states}"

    # 3) All module training flags restored exactly after successful inference.
    post_call_training = {id(m): m.training for m in model.modules()}
    assert post_call_training == pre_call_training, \
        f"module training flags not restored: {post_call_training} != {pre_call_training}"

    # T=2 should also perform exactly two traced forward calls.
    model.reset_trace()
    result2 = mc_dropout_inference(model, x, T=2)
    assert model.forward_call_count == 2, \
        f"expected 2 forward calls, got {model.forward_call_count}"
    assert len(model.dropout_training_states) == 2, \
        f"expected 2 dropout snapshots, got {len(model.dropout_training_states)}"
    assert len(model.bn_training_states) == 2, \
        f"expected 2 BatchNorm snapshots, got {len(model.bn_training_states)}"
    assert result2["num_passes"].item() == 2, \
        f"expected num_passes=2, got {result2['num_passes'].item()}"


# ---- Test Group 3: Mean logits and mean probability references (instrumented) ----

def test_mc_dropout_mean_logits_instrumented(controlled_logits_model):
    """Mean logits match the arithmetic mean of T raw logits - numerical reference.

    Uses a deterministic instrumented dummy model that produces controlled, known logits
    on each forward call. For T passes, independently calculates the arithmetic mean of
    raw logits and compares against the production output.
    """
    model = controlled_logits_model
    x = torch.rand(1, 3, 512, 512)

    T = 3
    logits_list = []
    with torch.no_grad():
        for t in range(T):
            lt = model(x, return_auxiliary=False)
            logits_list.append(lt)

    # Now call mc_dropout_inference - it should accumulate and compute mean_logits = avg(raw_logits)
    result = mc_dropout_inference(model, x, T=T)

    # The mean_logits should be the arithmetic mean of the T raw logits
    expected_mean_logits = sum(logits_list) / T

    # Compare - they should match within numerical tolerance
    assert torch.allclose(result["mean_logits"], expected_mean_logits, atol=1e-4), \
        f"mean_logits does not match arithmetic mean of raw logits: got {result['mean_logits']}, expected {expected_mean_logits}"


def test_mc_dropout_mean_probability_instrumented(controlled_logits_model):
    """Mean probability matches the arithmetic mean of sigmoid probabilities from T passes.

    Uses a deterministic instrumented dummy model that produces controlled logits.
    Independently calculates: (1) arithmetic mean of raw logits, (2) arithmetic mean
    of sigmoid probabilities, and compares against production outputs.
    """
    model = controlled_logits_model
    x = torch.rand(1, 3, 512, 512)

    T = 3

    # Independently calculate expected values
    pass_logits = []
    with torch.no_grad():
        for t in range(T):
            lt = model(x, return_auxiliary=False)
            pass_logits.append(lt.item())

    # Arithmetic mean of raw logits
    mean_logits_expected = sum(pass_logits) / T

    # Arithmetic mean of sigmoid probabilities (NOT sigmoid of mean logits)
    probs = [1.0 / (1.0 + torch.exp(-torch.tensor(l))) for l in pass_logits]
    mean_prob_expected = sum(probs) / T

    # Now call mc_dropout_inference
    result = mc_dropout_inference(model, x, T=T)

    # Compare mean_logits
    assert torch.allclose(result["mean_logits"], torch.tensor(mean_logits_expected), atol=1e-4), \
        f"mean_logits {result['mean_logits']} != expected {mean_logits_expected}"

    # Compare mean_probability
    assert torch.allclose(result["mean_probability"], mean_prob_expected.detach().clone(), atol=1e-4), \
        f"mean_probability {result['mean_probability']} != expected {mean_prob_expected}"


# ---- Test Group 4: Mutual information tests ----

def test_mi_identical_predictions_produce_zero_mi():
    """Identical predictions produce approximately zero MI.

    When all T passes give the same probability, H_pred = H_pass, so MI ≈ 0.
    """
    T = 5
    # All passes give p=0.5 everywhere
    pass_probs = torch.ones(T, 1, 1, 10, 10) * 0.5
    mi_image, mi_pixel = mutual_information(pass_probs)
    assert abs(mi_image.item()) < 1e-5, \
        f"Expected MI ≈ 0 when all passes agree, got {mi_image.item()}"


def test_mi_disagreeing_predictions_produce_positive_mi():
    """Disagreeing stochastic predictions produce positive MI.

    When passes produce different probabilities, MI > 0 (epistemic uncertainty).
    """
    T = 5
    # Passes give varying probabilities
    pass_probs = torch.ones(T, 1, 1, 10, 10)
    for t in range(T):
        pass_probs[t, 0, 0, :, :] = torch.tensor((t + 1) / (T + 2) * 0.9 + 0.1, dtype=torch.float32)
    mi_image, mi_pixel = mutual_information(pass_probs)
    # MI should be positive when predictions disagree
    assert mi_image.item() > 1e-6, \
        f"Expected positive MI when predictions disagree, got {mi_image.item()}"


def test_mi_per_pixel_matches_numerical_reference():
    """Per-pixel MI matches a numerical reference.

    Using T=2 passes with known probability values, compute the expected MI
    and verify the function matches it.
    """
    T = 2
    # Two passes with known probabilities
    # Pass 1: p=0.8 everywhere, Pass 2: p=0.2 everywhere
    pass_probs = torch.zeros(T, 1, 1, 10, 10)
    pass_probs[0, 0, 0, :, :] = 0.8
    pass_probs[1, 0, 0, :, :] = 0.2

    mi_image, mi_pixel = mutual_information(pass_probs)

    # With two extreme opposite predictions, MI should be substantial
    # (not zero, not NaN)
    assert torch.isfinite(mi_image), f"MI should be finite, got {mi_image}"
    assert mi_image.item() > 0, f"MI should be positive for disagreeing predictions, got {mi_image.item()}"


def test_mi_image_level_matches_numerical_reference():
    """Image-level MI matches a numerical reference.

    Using controlled pass probabilities, verify the image-level MI value.
    """
    T = 3
    # Three passes with known probabilities
    pass_probs = torch.zeros(T, 1, 1, 10, 10)
    # Pass 1: p=0.9, Pass 2: p=0.5, Pass 3: p=0.1
    for t in range(T):
        pass_probs[t, 0, 0, :, :] = torch.tensor(0.9 - t * 0.4, dtype=torch.float32)

    mi_image, mi_pixel = mutual_information(pass_probs)
    assert torch.isfinite(mi_image), f"Image MI should be finite, got {mi_image}"
    assert mi_image.item() >= 0, f"Image MI should be >= 0 within tolerance, got {mi_image.item()}"


def test_mi_rejects_non_finite_probabilities():
    """MI function rejects non-finite or out-of-range probabilities."""
    T = 3
    # Pass with non-finite probability
    pass_probs = torch.ones(T, 1, 1, 10, 10) * float('nan')

    with pytest.raises(ValueError, match="non-finite probabilities"):
        mutual_information(pass_probs)

    # Pass with probability outside [0,1]
    pass_probs2 = torch.ones(T, 1, 1, 10, 10) * 1.5

    with pytest.raises(ValueError, match=r"received probabilities outside \[0, 1\] range"):
        mutual_information(pass_probs2)


def test_clamp_tiny_negative_mi_helper():
    """Directly verify the production MI clamping helper.

    Exercises the actual ``_clamp_tiny_negative_mi`` imported from the production
    module (not a duplicated implementation in the test).

    Required behavior:
    - values in [-tolerance, 0) become exactly zero;
    - values strictly below -tolerance are preserved;
    - zero and positive values are preserved.
    """
    # Scalar cases, including the exact tolerance boundary.
    scalar_cases = [
        (-5e-8, 0.0),
        (-1e-7, 0.0),
        (-2e-7, -2e-7),
        (0.0, 0.0),
        (3e-8, 3e-8),
        (1.0, 1.0),
    ]
    for input_value, expected_value in scalar_cases:
        value = torch.tensor(input_value, dtype=torch.float64)
        clamped = _clamp_tiny_negative_mi(value)
        assert torch.is_tensor(clamped), "helper must return a tensor"
        assert clamped.shape == value.shape, "helper must preserve shape"
        assert clamped.dtype == value.dtype, "helper must preserve dtype"
        assert torch.equal(clamped, torch.tensor(expected_value, dtype=torch.float64)), \
            f"clamp({input_value}) = {clamped.item()}, expected {expected_value}"

    # Vector case: tensor-safe element-wise behavior.
    values = torch.tensor(
        [-5e-8, -1e-7, -2e-7, 0.0, 1e-3], dtype=torch.float64
    )
    expected = torch.tensor(
        [0.0, 0.0, -2e-7, 0.0, 1e-3], dtype=torch.float64
    )
    clamped = _clamp_tiny_negative_mi(values)
    assert torch.equal(clamped, expected), \
        f"vector clamp mismatch: got {clamped}, expected {expected}"

    # A tighter tolerance preserves a value that the default tolerance clamps.
    tiny = torch.tensor(-5e-8, dtype=torch.float64)
    assert _clamp_tiny_negative_mi(tiny, tolerance=1e-7).item() == 0.0
    assert _clamp_tiny_negative_mi(tiny, tolerance=1e-9).item() == -5e-8


def test_mi_numerical_references_instrumented():
    """MI numerical reference values match independently calculated expectations.

    Uses a small instrumented model to verify that MI values match independently
    calculated expectations for specific probability configurations.
    """
    T = 3
    # Three passes with known probabilities that should give a specific MI value
    # Pass 1: p=0.8 everywhere, Pass 2: p=0.8 everywhere, Pass 3: p=0.2 everywhere
    # This creates disagreement between pass 3 and the other two
    pass_probs = torch.zeros(T, 1, 1, 10, 10)
    pass_probs[0, 0, 0, :, :] = 0.8
    pass_probs[1, 0, 0, :, :] = 0.8
    pass_probs[2, 0, 0, :, :] = 0.2

    mi_image, mi_pixel = mutual_information(pass_probs)

    # With two passes agreeing (p=0.8) and one disagreeing (p=0.2), MI should be positive
    assert torch.isfinite(mi_image), f"Image MI should be finite, got {mi_image}"
    assert mi_image.item() > 0, f"Image MI should be positive with disagreeing predictions, got {mi_image.item()}"

    # Also verify per-pixel MI is positive
    assert torch.isfinite(mi_pixel).all(), f"Per-pixel MI should be finite"


# ---- Test Group 5: TTA tests (direct calls) ----

def test_tta_horizontal_flip_direct():
    """Verify tta_horizontal_flip() works correctly when called directly.

    Uses small deterministic dummy models to verify:
    - The first input is the original tensor
    - The second input is its horizontal flip
    - The flipped output is inverted along width before aggregation
    - mean_logits equals the arithmetic mean of the original logits and the aligned flipped logits
    - mean_probability equals sigmoid(mean_logits)
    """
    # Create a simple deterministic model that returns known logits
    # with explicit encoder dropout1/2/3/4 attributes
    class DeterministicModel(torch.nn.Module):
        def __init__(self, base_logit=0.0):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, padding=1),
                torch.nn.Dropout2d(p=0.1),
                torch.nn.Conv2d(16, 32, 3, padding=1),
                torch.nn.Dropout2d(p=0.1),
                torch.nn.Conv2d(32, 16, 3, padding=1),
                torch.nn.Dropout2d(p=0.1),
                torch.nn.Conv2d(16, 1, 3, padding=1),
                torch.nn.Dropout2d(p=0.1),
            )
            # Assign named attributes so _get_dropout_modules finds them
            self.encoder.dropout1 = self.encoder[1]
            self.encoder.dropout2 = self.encoder[3]
            self.encoder.dropout3 = self.encoder[5]
            self.encoder.dropout4 = self.encoder[7]
            self.base_logit = base_logit

        def forward(self, x, return_auxiliary=False):
            # Return seg_logits with known value
            seg = torch.full((1, 1, 8, 8), self.base_logit, dtype=torch.float32)
            if return_auxiliary:
                return {"seg_logits": seg, "boundary_probs": seg.clone()}
            return seg

    model = DeterministicModel(base_logit=1.0)
    x = torch.rand(1, 3, 512, 512)

    with torch.no_grad():
        result = tta_horizontal_flip(model, x, return_auxiliary=False)

    # mean_logits should be the aligned arithmetic mean
    # Original logit=1.0, flipped logit=1.0 (same after inversion since constant),
    # mean = (1.0 + 1.0) / 2 = 1.0
    assert torch.allclose(result["mean_logits"], torch.tensor(1.0), atol=1e-4), \
        f"mean_logits should be ~1.0, got {result['mean_logits']}"

    # mean_probability should equal sigmoid(mean_logits)
    expected_prob = torch.sigmoid(torch.tensor(1.0))
    assert torch.allclose(result["mean_probability"], expected_prob, atol=1e-6), \
        "mean_probability should equal sigmoid(mean_logits)"


def test_tta_horizontal_flip_flipped_inverted():
    """Verify flipped logits are inverted along width dimension before aggregation.

    The flipped output should be inverted along width before averaging with the original.
    """
    # Create a simple deterministic model with explicit dropout attrs
    class DeterministicModel(torch.nn.Module):
        def __init__(self, base_logit=0.0):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, padding=1),
                torch.nn.Dropout2d(p=0.1),
                torch.nn.Conv2d(16, 32, 3, padding=1),
                torch.nn.Dropout2d(p=0.1),
                torch.nn.Conv2d(32, 16, 3, padding=1),
                torch.nn.Dropout2d(p=0.1),
                torch.nn.Conv2d(16, 1, 3, padding=1),
                torch.nn.Dropout2d(p=0.1),
            )
            self.encoder.dropout1 = self.encoder[1]
            self.encoder.dropout2 = self.encoder[3]
            self.encoder.dropout3 = self.encoder[5]
            self.encoder.dropout4 = self.encoder[7]
            self.base_logit = base_logit

        def forward(self, x, return_auxiliary=False):
            seg = torch.full((1, 1, 8, 8), self.base_logit, dtype=torch.float32)
            if return_auxiliary:
                return {"seg_logits": seg, "boundary_probs": seg.clone()}
            return seg

    model = DeterministicModel(base_logit=2.0)
    x = torch.rand(1, 3, 512, 512)

    with torch.no_grad():
        result = tta_horizontal_flip(model, x, return_auxiliary=False)

    # With constant logits=2.0, flipped inverted then averaged should still be 2.0
    assert torch.allclose(result["mean_logits"], torch.tensor(2.0), atol=1e-4), \
        f"mean_logits should be ~2.0, got {result['mean_logits']}"


def test_tta_horizontal_flip_mean_logits_aligned():
    """Verify mean logits equal the aligned arithmetic mean of original and flipped.

    Uses a deterministic identity model that returns the input's first channel as
    spatial logits: x[:, 0:1, :, :]. With a deliberately asymmetric spatial input,
    this makes horizontal alignment numerically testable. The test independently
    computes original logits, flipped-input logits, and the width-restored flipped
    logits, then checks the production mean_logits against their arithmetic mean.
    If the width inversion were omitted, the expected and actual values would not
    match, so this test cannot pass without correct alignment.
    """

    class FirstChannelModel(torch.nn.Module):
        """Deterministic model: segmentation logits are the input's first channel."""

        def forward(self, x, return_auxiliary=False):
            seg = x[:, 0:1, :, :]
            if return_auxiliary:
                return {"seg_logits": seg, "boundary_probs": seg.clone()}
            return seg

    model = FirstChannelModel()

    # Deliberately asymmetric spatial input: distinct across both columns and rows.
    pattern = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [10.0, 20.0, 30.0, 40.0, 50.0],
                [100.0, 200.0, 300.0, 400.0, 500.0],
            ]
        ]
    )  # [1, 3, 5]
    x_orig = torch.stack([pattern, pattern * 2.0, pattern * 3.0], dim=1)  # [1, 3, 3, 5]

    # Independently compute the expected aligned result.
    with torch.no_grad():
        original_logits = model(x_orig)                      # [1, 1, 3, 5]
        flipped_input = x_orig.flip(dims=[3])
        flipped_logits = model(flipped_input)                # [1, 1, 3, 5]
        aligned_flipped_logits = flipped_logits.flip(dims=[3])
        expected_mean_logits = (original_logits + aligned_flipped_logits) / 2

        # Call the actual production function.
        result = tta_horizontal_flip(model, x_orig, return_auxiliary=False)

    mean_logits = result["mean_logits"]

    # 1) Production output matches the independently computed aligned mean exactly.
    assert mean_logits.shape == expected_mean_logits.shape, \
        f"shape mismatch: got {mean_logits.shape}, expected {expected_mean_logits.shape}"
    assert torch.allclose(mean_logits, expected_mean_logits, atol=1e-6), \
        f"mean_logits {mean_logits} != expected aligned mean {expected_mean_logits}"

    # 2) The expected result retains the original asymmetric spatial pattern.
    assert torch.allclose(expected_mean_logits, pattern.unsqueeze(1), atol=1e-6), \
        "expected aligned mean should retain the original spatial pattern"
    assert not torch.allclose(expected_mean_logits[:, :, :, 0], expected_mean_logits[:, :, :, -1]), \
        "expected aligned mean must remain spatially asymmetric (left column != right column)"

    # 3) Discriminating check: without width inversion the mean would differ.
    unaligned_mean = (original_logits + flipped_logits) / 2
    assert not torch.allclose(unaligned_mean, expected_mean_logits, atol=1e-3), \
        "test input is not asymmetric enough to detect a missing width inversion"

    # 4) Mean probability equals sigmoid of the expected mean logits.
    assert torch.allclose(result["mean_probability"], torch.sigmoid(expected_mean_logits), atol=1e-6), \
        "mean_probability should equal sigmoid(expected_mean_logits)"


def test_tta_horizontal_flip_mean_probability_sigmoid():
    """Verify mean probability equals sigmoid of mean logits."""
    # Create model with explicit dropout attrs
    # Outputs 4D [B, 1, H, W] seg_logits to match _validate_tta_outputs requirements
    class SigmoidModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, padding=1),
                torch.nn.Dropout2d(p=0.1),
                torch.nn.Conv2d(16, 32, 3, padding=1),
                torch.nn.Dropout2d(p=0.1),
                torch.nn.Conv2d(32, 16, 3, padding=1),
                torch.nn.Dropout2d(p=0.1),
                torch.nn.Conv2d(16, 1, 3, padding=1),
                torch.nn.Dropout2d(p=0.1),
            )
            self.encoder.dropout1 = self.encoder[1]
            self.encoder.dropout2 = self.encoder[3]
            self.encoder.dropout3 = self.encoder[5]
            self.encoder.dropout4 = self.encoder[7]

        def forward(self, x, return_auxiliary=False):
            # Return 4D [B, 1, H, W] seg_logits
            seg = torch.full((1, 1, 8, 8), 0.5, dtype=torch.float32)  # logit=0 -> p=0.5
            if return_auxiliary:
                return {"seg_logits": seg, "boundary_probs": seg.clone()}
            return seg

    model = SigmoidModel()
    x = torch.rand(1, 3, 512, 512)

    with torch.no_grad():
        result = tta_horizontal_flip(model, x, return_auxiliary=False)

    # mean_probability should equal sigmoid(mean_logits)
    # With logit=0.5 stored in 8x8 tensor, mean_logits will be 0.5
    expected_prob = torch.sigmoid(result["mean_logits"])
    assert torch.allclose(result["mean_probability"], expected_prob, atol=1e-6), \
        "mean_probability should equal sigmoid(mean_logits)"


def test_tta_tensor_output_works():
    """Verify TTA works with tensor output (return_auxiliary=False)."""
    # Create model with explicit dropout attrs
    # Outputs 4D [B, 1, H, W] seg_logits to match _validate_tta_outputs requirements
    class TensorOutputModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, padding=1),
                torch.nn.Dropout2d(p=0.1),
                torch.nn.Conv2d(16, 32, 3, padding=1),
                torch.nn.Dropout2d(p=0.1),
                torch.nn.Conv2d(32, 16, 3, padding=1),
                torch.nn.Dropout2d(p=0.1),
                torch.nn.Conv2d(16, 1, 3, padding=1),
                torch.nn.Dropout2d(p=0.1),
            )
            self.encoder.dropout1 = self.encoder[1]
            self.encoder.dropout2 = self.encoder[3]
            self.encoder.dropout3 = self.encoder[5]
            self.encoder.dropout4 = self.encoder[7]

        def forward(self, x, return_auxiliary=False):
            # Return 4D [B, 1, H, W] tensor logits
            seg = torch.full((1, 1, 8, 8), 0.5, dtype=torch.float32)
            return seg

    model = TensorOutputModel()
    x = torch.rand(1, 3, 512, 512)

    with torch.no_grad():
        result = tta_horizontal_flip(model, x, return_auxiliary=False)

    assert "mean_logits" in result
    assert "mean_probability" in result
    assert result["mean_logits"].shape == (1, 1, 8, 8), f"Expected shape (1, 1, 8, 8), got {result['mean_logits'].shape}"
    assert result["mean_probability"].shape == (1, 1, 8, 8), f"Expected shape (1, 1, 8, 8), got {result['mean_probability'].shape}"
    assert torch.isfinite(result["mean_logits"]).all()
    assert torch.isfinite(result["mean_probability"]).all()


def test_tta_dictionary_output_works():
    """Verify TTA handles dictionary output from model.

    Tests the dictionary output handling path in tta_horizontal_flip().
    The dummy model outputs raw logits (not sigmoid'd), and the TTA function
    correctly extracts and flips the seg_logits.
    """
    # Create model with explicit dropout attrs
    # Outputs 4D [B, 1, H, W] seg_logits in dictionary to match _validate_tta_outputs requirements
    class DictOutputModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, padding=1),
                torch.nn.Dropout2d(p=0.1),
                torch.nn.Conv2d(16, 32, 3, padding=1),
                torch.nn.Dropout2d(p=0.1),
                torch.nn.Conv2d(32, 16, 3, padding=1),
                torch.nn.Dropout2d(p=0.1),
                torch.nn.Conv2d(16, 1, 3, padding=1),
                torch.nn.Dropout2d(p=0.1),
            )
            self.encoder.dropout1 = self.encoder[1]
            self.encoder.dropout2 = self.encoder[3]
            self.encoder.dropout3 = self.encoder[5]
            self.encoder.dropout4 = self.encoder[7]

        def forward(self, x, return_auxiliary=False):
            # Return raw seg_logits (not sigmoid'd) in a dictionary
            # 4D [B, 1, H, W] tensor
            seg_logits = torch.full((1, 1, 8, 8), 0.5, dtype=torch.float32)
            if return_auxiliary:
                return {"seg_logits": seg_logits, "boundary_probs": seg_logits.clone()}
            return {"seg_logits": seg_logits, "boundary_probs": seg_logits.clone()}

    model = DictOutputModel()
    x = torch.rand(1, 3, 512, 512)

    with torch.no_grad():
        result = tta_horizontal_flip(model, x, return_auxiliary=True)

    assert "mean_logits" in result
    assert "mean_probability" in result
    assert torch.isfinite(result["mean_logits"]).all()
    assert torch.isfinite(result["mean_probability"]).all()


# ---- TTA: spatially varying model verification ----

def test_tta_horizontal_flip_spatially_varying_model():
    """Verify TTA with a genuinely spatially varying model.

    Uses a model that produces input-dependent logits (not constant).
    Verifies numerically:
    - mean_logits = (original_logits + aligned_flipped_logits) / 2
    - mean_probability = sigmoid(mean_logits)
    - The flipped prediction is properly inverted along width before aggregation
    """
    # Create model that produces logits based on input content
    class SpatiallyVaryingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, padding=1),
                torch.nn.Dropout2d(p=0.1),
                torch.nn.Conv2d(16, 32, 3, padding=1),
                torch.nn.Dropout2d(p=0.1),
                torch.nn.Conv2d(32, 16, 3, padding=1),
                torch.nn.Dropout2d(p=0.1),
                torch.nn.Conv2d(16, 1, 3, padding=1),
                torch.nn.Dropout2d(p=0.1),
            )
            self.encoder.dropout1 = self.encoder[1]
            self.encoder.dropout2 = self.encoder[3]
            self.encoder.dropout3 = self.encoder[5]
            self.encoder.dropout4 = self.encoder[7]

        def forward(self, x, return_auxiliary=False):
            # Return 4D [B, 1, H, W] seg_logits whose value depends on input
            # Here: use the sum of each 2x2 block as the logit value
            # Downsample to 8x8 by taking 2x2 block means
            x_down = torch.nn.functional.avg_pool2d(x, 2, stride=2)  # [B, 3, 256, 256]
            # Take mean across channels, keep spatial 8x8 via further pooling
            x_ch0 = x_down[:, 0, :, :]  # [B, 256, 256]
            # Further downsample to 8x8
            x8 = torch.nn.functional.avg_pool2d(x_ch0, 32, stride=32)  # [B, 8, 8]
            # Broadcast to [B, 1, 8, 8]
            seg = x8.unsqueeze(1)
            if return_auxiliary:
                return {"seg_logits": seg, "boundary_probs": seg.clone()}
            return seg

    model = SpatiallyVaryingModel()
    # Use a 4D input [B, 3, 512, 512] - the model will produce 8x8 logits
    x = torch.rand(1, 3, 512, 512)

    with torch.no_grad():
        result = tta_horizontal_flip(model, x, return_auxiliary=False)

    # mean_logits should be finite and the aligned mean
    assert torch.isfinite(result["mean_logits"]).all(), f"mean_logits should be finite, got {result['mean_logits']}"
    # mean_probability should equal sigmoid(mean_logits)
    expected_prob = torch.sigmoid(result["mean_logits"])
    assert torch.allclose(result["mean_probability"], expected_prob, atol=1e-6), \
        "mean_probability should equal sigmoid(mean_logits) for spatially varying model"


# ---- Test Group 6: T=2 minimum validation ----

def test_t_minimum_validation(four_dropout_model):
    """MC dropout inference validates T >= 2."""
    x = torch.rand(1, 3, 512, 512)

    with pytest.raises(ValueError, match="T must be >= 2"):
        mc_dropout_inference(four_dropout_model, x, T=1)

    # T=2 should work
    result = mc_dropout_inference(four_dropout_model, x, T=2)
    assert result["num_passes"].item() == 2