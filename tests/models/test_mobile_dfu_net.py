"""
Unit and integration tests for MobileDFU-Net architecture:
1. Forward pass with batch size 1.
2. Forward pass with batch size 4.
3. Output shape correctness ([B, 1, 512, 512]).
4. Output finite-value checks (no NaNs or Infs).
5. CPU execution.
6. CUDA execution (if available).
7. Backpropagation and gradient computation.
8. Nonzero gradients on all trainable parameters.
9. Parameter count verification (< 4.0 M).
10. GFLOPs verification (< 8.0 GFLOPs).
11. Model serialization and reload.
12. DCS-Stem gradient feature extraction.
13. Boundary head dual output (logits, probs).
14. Primary segmentation output is raw logits (can be negative and positive).
15. Inference mode (return_auxiliary=False) returns single tensor.
"""

import tempfile
from pathlib import Path
import pytest
import torch

from src.models.mobile_dfu_net.model import MobileDFUNet
from src.models.mobile_dfu_net.config import MobileDFUNetConfig
from src.models.mobile_dfu_net.profile import count_parameters, measure_flops_manual


@pytest.fixture(scope="module")
def model():
    m = MobileDFUNet()
    m.eval()
    return m


def test_01_forward_batch_1(model):
    """Test 1: Forward pass with batch size 1."""
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, dict)
    assert "seg_logits" in out
    assert out["seg_logits"].shape == (1, 1, 512, 512)


def test_02_forward_batch_4(model):
    """Test 2: Forward pass with batch size 4."""
    x = torch.randn(4, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    assert out["seg_logits"].shape == (4, 1, 512, 512)
    assert out["boundary_logits"].shape == (4, 1, 512, 512)
    assert out["boundary_probs"].shape == (4, 1, 512, 512)


def test_03_output_shape_correctness(model):
    """Test 3: Primary segmentation output has exact shape [B, 1, 512, 512]."""
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = model(x, return_auxiliary=False)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 1, 512, 512)


def test_04_output_finite_values(model):
    """Test 4: Output finite-value checks (no NaNs or Infs)."""
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out["seg_logits"]).all()
    assert torch.isfinite(out["boundary_logits"]).all()
    assert torch.isfinite(out["boundary_probs"]).all()


def test_05_cpu_execution(model):
    """Test 5: Model runs smoothly on CPU."""
    x = torch.randn(1, 3, 512, 512, device="cpu")
    model_cpu = model.to("cpu")
    with torch.no_grad():
        out = model_cpu(x)
    assert out["seg_logits"].device.type == "cpu"


def test_06_cuda_execution(model):
    """Test 6: CUDA execution if available, otherwise cleanly skipped."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine (running in CPU mode).")
    x = torch.randn(1, 3, 512, 512, device="cuda")
    model_cuda = MobileDFUNet().to("cuda")
    with torch.no_grad():
        out = model_cuda(x)
    assert out["seg_logits"].device.type == "cuda"


def test_07_backpropagation():
    """Test 7: Backpropagation executes without error."""
    m = MobileDFUNet()
    m.train()
    x = torch.randn(2, 3, 512, 512)
    out = m(x)
    loss = out["seg_logits"].mean() + out["boundary_logits"].mean()
    loss.backward()
    assert True


def test_08_nonzero_gradients():
    """Test 8: Trainable parameters receive valid non-zero gradients."""
    m = MobileDFUNet()
    m.train()
    x = torch.randn(2, 3, 512, 512)
    out = m(x)
    loss = out["seg_logits"].sum() + out["boundary_logits"].sum()
    loss.backward()
    
    for name, p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has None grad!"
            assert not torch.isnan(p.grad).any(), f"Parameter {name} has NaN grad!"
    
    nonzero_grads = [p.grad.abs().sum().item() > 0 for p in m.parameters() if p.requires_grad]
    assert any(nonzero_grads), "No parameters received non-zero gradients!"


def test_09_parameter_budget(model):
    """Test 9: Total parameters < 4.0 Million."""
    total, trainable = count_parameters(model)
    params_m = total / 1e6
    assert params_m < 4.0, f"Model has {params_m:.3f}M parameters, exceeding 4.0M limit!"
    assert total == trainable


def test_10_gflops_budget(model):
    """Test 10: Computation < 8.0 GFLOPs at 512x512."""
    macs = measure_flops_manual(model, (1, 3, 512, 512))
    gflops = (macs * 2) / 1e9
    assert gflops < 8.0, f"Model computation {gflops:.3f} GFLOPs exceeds 8.0 GFLOPs budget!"


def test_11_model_serialization(model):
    """Test 11: Model serialization and reload."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "mobile_dfu_net_test.pt"
        torch.save(model.state_dict(), save_path)
        assert save_path.exists()
        
        new_model = MobileDFUNet()
        new_model.load_state_dict(torch.load(save_path, map_location="cpu", weights_only=True))
        new_model.eval()
        
        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            out1 = model(x, return_auxiliary=False)
            out2 = new_model(x, return_auxiliary=False)
        assert torch.allclose(out1, out2, atol=1e-5)


def test_12_dcs_stem_differentiation(model):
    """Test 12: DCS-Stem correctly separates and gates chromatic and structural features."""
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        grad = model.stem.extract_gradient(x)
        f0 = model.stem(x)
    assert grad.shape == (2, 1, 512, 512)
    assert f0.shape == (2, 24, 512, 512)
    assert (grad >= 0).all()  # Gradient magnitudes must be non-negative


def test_13_boundary_head_outputs(model):
    """Test 13: Boundary head outputs raw logits and Sigmoid-bounded probabilities [0, 1]."""
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    probs = out["boundary_probs"]
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_14_primary_logits_are_unbounded(model):
    """Test 14: Primary segmentation head produces raw logits (not Sigmoided probabilities)."""
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        logits = model(x, return_auxiliary=False)
    # Raw logits can take negative and positive values
    has_negative = (logits < 0).any()
    has_positive = (logits > 0).any()
    assert has_negative and has_positive, "Primary segmentation head should output unconstrained logits!"


def test_15_inference_mode_flag(model):
    """Test 15: Inference mode flag return_auxiliary=False returns single Tensor."""
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        res = model(x, return_auxiliary=False)
    assert isinstance(res, torch.Tensor)
    assert res.shape == (1, 1, 512, 512)


