"""
Model profiling utilities for MobileDFU-Net:
Calculates:
- Total parameters
- Trainable parameters
- Model size (MB)
- FLOPs and MACs (using fvcore or ptflops or exact analytical calculation)
- CPU / GPU inference latency and FPS with warm-up iterations
"""

import time
import platform
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn

from src.models.mobile_dfu_net.model import MobileDFUNet
from src.models.mobile_dfu_net.config import MobileDFUNetConfig


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Returns (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_model_size_mb(model: nn.Module) -> float:
    """Calculates model size in Megabytes (FP32)."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def measure_flops_manual(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 512, 512)) -> int:
    """
    Measures Multiply-Accumulate Operations (MACs) using PyTorch forward hooks.
    Convention: 1 MAC = 2 FLOPs.
    """
    total_macs = 0

    def conv_hook(module, input, output):
        nonlocal total_macs
        # Output shape: [B, C_out, H_out, W_out]
        out = output[0] if isinstance(output, tuple) else output
        b, c_out, h_out, w_out = out.shape
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        macs_per_pos = kernel_ops
        total_macs += b * c_out * h_out * w_out * macs_per_pos

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))

    device = next(model.parameters()).device
    # Model input contract: raw RGB float32 in [0, 1].
    dummy_input = torch.rand(*input_size, device=device)
    with torch.no_grad():
        model(dummy_input, return_auxiliary=False)

    for h in hooks:
        h.remove()

    return total_macs


def benchmark_latency(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 512, 512),
    device: str = "cpu",
    warmup_iters: int = 10,
    test_iters: int = 50
) -> Dict[str, float]:
    """
    Benchmarks inference latency and throughput (FPS).
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)
    
    # Model input contract: raw RGB float32 in [0, 1].
    dummy_input = torch.rand(*input_size, device=dev)

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(dummy_input, return_auxiliary=False)
            if dev.type == "cuda":
                torch.cuda.synchronize()

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(test_iters):
            t0 = time.perf_counter()
            _ = model(dummy_input, return_auxiliary=False)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    mean_latency = sum(latencies) / len(latencies)
    fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0

    return {
        "mean_latency_ms": round(mean_latency, 2),
        "min_latency_ms": round(min(latencies), 2),
        "max_latency_ms": round(max(latencies), 2),
        "fps": round(fps, 2)
    }


def profile_mobile_dfu_net(config: Optional[MobileDFUNetConfig] = None) -> Dict[str, Any]:
    """
    Runs full architectural profile of MobileDFU-Net.
    """
    model = MobileDFUNet(config)
    total_params, trainable_params = count_parameters(model)
    model_size_mb = compute_model_size_mb(model)
    macs = measure_flops_manual(model, (1, 3, 512, 512))
    gmacs = macs / 1e9
    gflops = (macs * 2) / 1e9  # Standard definition: 1 MAC = 2 FLOPs

    # Benchmark CPU latency
    cpu_bench = benchmark_latency(model, (1, 3, 512, 512), device="cpu", warmup_iters=5, test_iters=15)

    profile_data = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "total_parameters_million": round(total_params / 1e6, 3),
        "model_size_mb_fp32": round(model_size_mb, 2),
        "macs": macs,
        "gmacs": round(gmacs, 3),
        "gflops": round(gflops, 3),
        "cpu_latency": cpu_bench,
        "hardware_info": {
            "processor": platform.processor(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available()
        }
    }

    if torch.cuda.is_available():
        gpu_bench = benchmark_latency(model, (1, 3, 512, 512), device="cuda", warmup_iters=10, test_iters=50)
        profile_data["gpu_latency"] = gpu_bench
        profile_data["hardware_info"]["gpu_name"] = torch.cuda.get_device_name(0)

    return profile_data


if __name__ == "__main__":
    prof = profile_mobile_dfu_net()
    print("=== MobileDFU-Net Complexity Profile ===")
    print(f"Parameters: {prof['total_parameters_million']} M")
    print(f"Size (FP32): {prof['model_size_mb_fp32']} MB")
    print(f"GMACs: {prof['gmacs']}")
    print(f"GFLOPs: {prof['gflops']}")
    print(f"CPU Latency: {prof['cpu_latency']['mean_latency_ms']} ms ({prof['cpu_latency']['fps']} FPS)")
