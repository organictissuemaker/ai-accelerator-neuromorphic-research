"""Shared benchmarking helpers — same metrics/format for every task.

Place at: simulations/accelerator/benchmarks/benchmark_utils.py
Every task's run.py imports from here so all results.csv files share columns.

Metrics captured per run:
  latency_ms, p50_ms, throughput_ips, n_iter,
  cpu_util_pct  - average CPU load during the run (can exceed 100% = multi-core),
  ram_mb        - peak process RAM during the run,
  gpu_mem_mb    - peak GPU memory (CUDA runs only).
Requires: pip install psutil
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
    on_cuda = sync_cuda and torch.cuda.is_available()

    # optional resource monitoring (psutil); degrade gracefully if absent
    try:
        import psutil
        proc = psutil.Process()
    except Exception:
        proc = None

    for _ in range(n_warmup):          # warm up (caches, kernels, autotune)
        run_once()
    if on_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    if proc is not None:
        proc.cpu_percent(None)         # prime the CPU% counter
        peak_ram = proc.memory_info().rss
    else:
        peak_ram = None

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        run_once()
        if on_cuda:
            torch.cuda.synchronize()   # GPU is async — sync before stopping timer
        times.append(time.perf_counter() - t0)
        if proc is not None:
            peak_ram = max(peak_ram, proc.memory_info().rss)

    mean = statistics.mean(times)
    out = {
        "latency_ms": round(mean * 1000, 3),
        "p50_ms": round(statistics.median(times) * 1000, 3),
        "throughput_ips": round(1.0 / mean, 2),   # inferences/sec (batch=1)
        "n_iter": n_iter,
        "cpu_util_pct": round(proc.cpu_percent(None), 1) if proc is not None else None,
        "ram_mb": round(peak_ram / 1e6, 1) if peak_ram is not None else None,
        "gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if on_cuda else None,
    }
    return out


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
