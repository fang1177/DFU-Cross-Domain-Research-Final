"""
Focused tests for the DCS-Stem raw-RGB colour-processing contract:
1. Raw-RGB luminance matches BT.601 coefficients.
2. Structural Sobel gradient is computed from raw RGB, not normalized RGB.
3. Chromatic normalization matches (x - mean) / std.
4. ImageNet mean/std are non-persistent buffers that follow the module device.
5. Output shapes and channel counts are unchanged.
6. Forward pass produces finite outputs.
7. Gradients flow through both the chromatic and structural branches.
8. Fixed luminance/Sobel kernels remain non-trainable buffers.
9. Legacy strict state-dictionary loading is unaffected by the new buffers.
"""

import torch
import torch.nn.functional as F

from src.models.mobile_dfu_net.config import DCSStemConfig
from src.models.mobile_dfu_net.dcs_stream import DCSStem
from src.models.mobile_dfu_net.model import MobileDFUNet


def test_01_luminance_matches_bt601():
    """Test 1: compute_luminance uses fixed BT.601 coefficients on raw RGB."""
    stem = DCSStem()
    x = torch.stack([
        torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        torch.tensor([[0.5, 0.6], [0.7, 0.8]]),
        torch.tensor([[0.9, 1.0], [0.0, 0.25]]),
    ]).unsqueeze(0)  # [1, 3, 2, 2]

    expected = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
    lum = stem.compute_luminance(x)

    assert lum.shape == (1, 1, 2, 2)
    assert torch.allclose(lum, expected, atol=1e-6)


def test_02_structural_gradient_uses_raw_rgb():
    """Test 2: Sobel gradient is derived from raw RGB, not normalized RGB."""
    stem = DCSStem()
    x = torch.rand(2, 3, 8, 8)

    raw_lum = stem.compute_luminance(x)
    gx = F.conv2d(raw_lum, stem.sobel_x, padding=1)
    gy = F.conv2d(raw_lum, stem.sobel_y, padding=1)
    expected = torch.sqrt(gx * gx + gy * gy + 1e-8)

    grad = stem.extract_gradient(x)
    assert torch.allclose(grad, expected, atol=1e-6)

    # If normalization had leaked into the structural path, these would match.
    x_norm = stem.normalize_chromatic(x)
    assert not torch.allclose(grad, stem.extract_gradient(x_norm), atol=1e-4)

    # A constant raw-RGB image has no interior structural gradient
    # (border responses come from zero padding and are expected).
    flat = torch.full((1, 3, 6, 6), 0.5)
    flat_grad = stem.extract_gradient(flat)
    assert torch.allclose(flat_grad[..., 1:-1, 1:-1], torch.zeros(1, 1, 4, 4), atol=1e-3)


def test_03_chromatic_normalization_matches_imagenet():
    """Test 3: chromatic normalization is exactly (x - mean) / std."""
    config = DCSStemConfig()
    stem = DCSStem(config)
    x = torch.rand(2, 3, 6, 5)

    mean = torch.tensor(config.image_mean).view(1, 3, 1, 1)
    std = torch.tensor(config.image_std).view(1, 3, 1, 1)
    expected = (x - mean) / std

    out = stem.normalize_chromatic(x)
    assert out.shape == x.shape
    assert torch.allclose(out, expected, atol=1e-6)
    assert torch.allclose(stem.image_mean.squeeze(), torch.tensor(config.image_mean), atol=1e-7)
    assert torch.allclose(stem.image_std.squeeze(), torch.tensor(config.image_std), atol=1e-7)


def test_04_mean_std_buffers_follow_device_and_are_non_persistent():
    """Test 4: mean/std are non-persistent buffers moved by .to(device)."""
    stem = DCSStem()
    buffers = dict(stem.named_buffers())

    for name in ("image_mean", "image_std"):
        assert name in buffers
        assert not buffers[name].requires_grad

    # Non-persistent: excluded from the persistent state dictionary.
    state_keys = set(stem.state_dict().keys())
    assert "image_mean" not in state_keys
    assert "image_std" not in state_keys

    # Buffers must follow a device move (meta avoids requiring CUDA hardware).
    target = torch.device("cuda") if torch.cuda.is_available() else torch.device("meta")
    moved = DCSStem().to(target)
    moved_buffers = dict(moved.named_buffers())
    assert moved_buffers["image_mean"].device.type == target.type
    assert moved_buffers["image_std"].device.type == target.type


def test_05_output_shapes_and_channels_unchanged():
    """Test 5: DCS-Stem output is [B, 24, H, W]; branches keep 16/8 channels."""
    stem = DCSStem()
    stem.eval()
    x = torch.rand(2, 3, 24, 32)

    with torch.no_grad():
        f_color = stem.color_stem(stem.normalize_chromatic(x))
        f_struct = stem.struct_stem(stem.extract_gradient(x))
        f0 = stem(x)

    assert f_color.shape == (2, 16, 24, 32)
    assert f_struct.shape == (2, 8, 24, 32)
    assert f0.shape == (2, 24, 24, 32)


def test_06_forward_produces_finite_outputs():
    """Test 6: forward pass on uniform [0,1] RGB produces finite output."""
    stem = DCSStem()
    stem.eval()
    x = torch.rand(1, 3, 16, 16)

    with torch.no_grad():
        out = stem(x)

    assert out.shape == (1, 24, 16, 16)
    assert torch.isfinite(out).all()


def test_07_gradients_flow_through_both_branches():
    """Test 7: chromatic and structural trainable parameters receive gradients."""
    torch.manual_seed(0)
    stem = DCSStem()
    stem.train()
    x = torch.rand(2, 3, 16, 16)

    out = stem(x)
    out.sum().backward()

    for name, param in stem.color_stem.named_parameters():
        assert param.grad is not None, f"color_stem.{name} has no gradient"
        assert torch.isfinite(param.grad).all()

    for name, param in stem.struct_stem.named_parameters():
        assert param.grad is not None, f"struct_stem.{name} has no gradient"
        assert torch.isfinite(param.grad).all()

    color_grad = sum(p.grad.abs().sum() for p in stem.color_stem.parameters())
    struct_grad = sum(p.grad.abs().sum() for p in stem.struct_stem.parameters())
    assert color_grad.item() > 0.0
    assert struct_grad.item() > 0.0


def test_08_fixed_kernels_are_non_trainable_buffers():
    """Test 8: luminance/Sobel kernels are buffers, not trainable parameters."""
    stem = DCSStem()
    bufs = dict(stem.named_buffers())
    params = dict(stem.named_parameters())

    for name in ("luminance_kernel", "sobel_x", "sobel_y"):
        assert name in bufs
        assert name not in params
        assert not bufs[name].requires_grad

    assert bufs["luminance_kernel"].shape == (1, 3, 1, 1)
    assert bufs["sobel_x"].shape == (1, 1, 3, 3)
    assert bufs["sobel_y"].shape == (1, 1, 3, 3)


def test_09_legacy_state_dict_loads_strictly():
    """Test 9: old-format state dict (without new norm buffers) loads strictly."""
    model = MobileDFUNet()
    state = model.state_dict()

    # New normalization buffers are non-persistent, so they are absent.
    assert not any("image_mean" in k or "image_std" in k for k in state)
    # Pre-existing fixed buffers are still part of the (legacy) checkpoint.
    assert "stem.luminance_kernel" in state
    assert "stem.sobel_x" in state
    assert "stem.sobel_y" in state

    legacy_state = {k: v.clone() for k, v in state.items()}
    new_model = MobileDFUNet()
    result = new_model.load_state_dict(legacy_state, strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []