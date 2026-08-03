"""Shared benchmarking helpers — same metrics/format for every task.

Place at: simulations/accelerator/benchmark_utils.py
Every task's run.py imports from here so all results.csv files share columns.
"""
import time
import csv
import os
import statistics

import torch


def get_device(pref="auto"):
    """pref: 'cpu' | 'cuda' | 'auto'. (TPU handled separately via torch_xla.)"""
    if pref == "cpu":
        return torch.device("cpu")
    if pref in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def benchmark(run_once, n_warmup=10, n_iter=50, sync_cuda=True):
    """run_once: zero-arg callable doing ONE inference. Returns metrics dict."""
    for _ in range(n_warmup):          # warm up (caches, kernels, autotune)
        run_once()
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        run_once()
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()   # GPU is async — must sync before stopping timer
        times.append(time.perf_counter() - t0)

    mean = statistics.mean(times)
    return {
        "latency_ms": round(mean * 1000, 3),
        "p50_ms": round(statistics.median(times) * 1000, 3),
        "throughput_ips": round(1.0 / mean, 2),   # inferences/sec (batch=1)
        "n_iter": n_iter,
    }


def gpu_power_w():
    """Instantaneous GPU power draw (W), or None if unavailable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0, 1)
    except Exception:
        return None


def log_result(csv_path, row):
    """Append one row; write header if file is new."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"logged: {row}")
