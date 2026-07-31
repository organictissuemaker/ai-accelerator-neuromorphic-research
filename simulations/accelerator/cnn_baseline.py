#!/usr/bin/env python3
"""
Route A - Step 1: Conventional CNN baseline + CPU/GPU inference benchmark.

Trains a LeNet-style CNN on MNIST, then benchmarks inference on the available
device(s): CPU, CUDA GPU (e.g. Colab), and Apple Silicon GPU (MPS). Logs
accuracy, latency-per-inference, throughput, MAC count, and parameter count.
Results are appended to a CSV so they feed the Step 4 comparison table and
Step 5 headline figure.

This is the "conventional accelerator" arm of the same-task comparison against
the SNN. Keep the test set and batch size fixed across ALL runs (CNN, SNN) so
the numbers are comparable.

Usage
-----
    python cnn_baseline.py                 # train + benchmark on all devices
    python cnn_baseline.py --epochs 3      # fewer epochs
    python cnn_baseline.py --device cpu    # restrict to one device (cpu/cuda/mps)
    python cnn_baseline.py --weights lenet_mnist.pt   # reuse saved weights

Dependencies: torch, torchvision, numpy. (No thop/ptflops needed - MACs are
counted analytically via forward hooks.)

Repo placement: simulations/accelerator/cnn_baseline.py
"""

import argparse
import csv
import os
import platform
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# --------------------------------------------------------------------------- #
# Device helpers
# --------------------------------------------------------------------------- #
def mps_available() -> bool:
    return getattr(torch.backends, "mps", None) is not None and \
        torch.backends.mps.is_available()


def sync(device):
    """Block until the device finishes queued work (needed for honest timing)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def best_train_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class LeNet(nn.Module):
    """LeNet-5 adapted for 28x28 MNIST (padding=2 on conv1 keeps 28x28)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)   # -> 6x28x28
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)             # -> 16x10x10
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # -> 6x14x14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # -> 16x5x5
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def get_loaders(data_dir: str, batch_size: int):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    # pin_memory only helps CUDA; disable it elsewhere (avoids an MPS warning).
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=pin)
    # Fixed test set, no shuffle -> reproducible across runs and vs. the SNN.
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=pin)
    return train_loader, test_loader, len(test_set)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(model, loader, device, epochs, lr):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        running, seen = 0.0, 0
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(images), targets)
            loss.backward()
            opt.step()
            running += loss.item() * images.size(0)
            seen += images.size(0)
        print(f"  epoch {epoch}/{epochs}  train_loss={running / seen:.4f}")
    return model


# --------------------------------------------------------------------------- #
# MAC counting (analytical, via forward hooks)
# --------------------------------------------------------------------------- #
def count_macs(model, device):
    """Count multiply-accumulate ops for ONE 28x28 sample.

    Derived from actual output tensor shapes, so it stays correct if the
    architecture changes.
        Conv2d MAC = out_H * out_W * out_C * (in_C * kH * kW)
        Linear MAC = in_features * out_features
    Pooling / ReLU are not MACs and are ignored (standard convention).
    """
    total = {"macs": 0}
    handles = []

    def conv_hook(module, inp, out):
        out_elems = out.shape[2] * out.shape[3] * out.shape[1]  # H*W*Cout
        per_out = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
        total["macs"] += out_elems * per_out

    def linear_hook(module, inp, out):
        total["macs"] += module.in_features * module.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, 28, 28, device=device))
    for h in handles:
        h.remove()
    if was_training:
        model.train()
    return total["macs"]


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
def benchmark(model, loader, device, n_samples, warmup_batches=3):
    """Run inference and measure accuracy + timing on `device`."""
    model.to(device).eval()

    # Warmup (lazy backend init, cuDNN autotune, kernel/shader caches).
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            model(images.to(device))
            if i + 1 >= warmup_batches:
                break
    sync(device)

    correct = 0
    total_time = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            sync(device)
            t0 = time.perf_counter()
            logits = model(images)
            sync(device)
            total_time += time.perf_counter() - t0
            correct += (logits.argmax(1) == targets).sum().item()

    accuracy = correct / n_samples
    latency_ms = (total_time / n_samples) * 1e3          # ms per inference
    throughput = n_samples / total_time                   # inferences / sec
    return accuracy, latency_ms, throughput


def device_label(device):
    if device.type == "cuda":
        return f"GPU ({torch.cuda.get_device_name(device)})"
    if device.type == "mps":
        return "GPU (Apple MPS)"
    return f"CPU ({platform.processor() or platform.machine()})"


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
CSV_FIELDS = [
    "timestamp", "system", "device_type", "device_name", "model",
    "dataset", "batch_size", "test_samples", "accuracy",
    "latency_ms_per_inf", "throughput_inf_per_s",
    "workload_metric", "workload_per_inf", "params",
]


def log_row(csv_path, row):
    new_file = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="CNN baseline + CPU/GPU benchmark")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128,
                   help="Keep IDENTICAL across CNN and SNN runs.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--weights", default=None,
                   help="Path to load/save model weights (.pt).")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"],
                   default="auto",
                   help="'auto' benchmarks CPU + any GPU (CUDA or Apple MPS).")
    p.add_argument("--csv", default="../../results/metrics.csv",
                   help="CSV to append results to.")
    args = p.parse_args()

    # Reproducibility.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader, test_loader, n_test = get_loaders(args.data_dir, args.batch_size)

    # Model: load weights if provided & present, else train.
    model = LeNet()
    train_device = best_train_device()
    if args.weights and os.path.exists(args.weights):
        print(f"Loading weights from {args.weights}")
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    else:
        print(f"Training LeNet on {train_device.type} for {args.epochs} epoch(s)...")
        train(model, train_loader, train_device, args.epochs, args.lr)
        if args.weights:
            model.to("cpu")
            torch.save(model.state_dict(), args.weights)
            print(f"Saved weights to {args.weights}")
    model.to("cpu")

    # Static model stats (device-independent).
    params = sum(p.numel() for p in model.parameters())
    macs = count_macs(model, torch.device("cpu"))
    print(f"\nModel: LeNet  |  params={params:,}  |  MACs/inference={macs:,}")

    # Which devices to benchmark.
    if args.device == "auto":
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        if mps_available():
            devices.append(torch.device("mps"))
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available.")
        if args.device == "mps" and not mps_available():
            raise SystemExit("MPS requested but not available.")
        devices = [torch.device(args.device)]

    print(f"\nBenchmarking on: {[d.type for d in devices]}")
    print(f"Test set: {n_test} samples, batch_size={args.batch_size}\n")

    results = []
    for dev in devices:
        acc, lat, thr = benchmark(model, test_loader, dev, n_test)
        label = device_label(dev)
        print(f"[{dev.type.upper():4}] acc={acc*100:5.2f}%  "
              f"latency={lat:.4f} ms/inf  throughput={thr:,.0f} inf/s")
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "system": platform.platform(),
            "device_type": dev.type,
            "device_name": label,
            "model": "LeNet",
            "dataset": "MNIST",
            "batch_size": args.batch_size,
            "test_samples": n_test,
            "accuracy": round(acc, 4),
            "latency_ms_per_inf": round(lat, 4),
            "throughput_inf_per_s": round(thr, 1),
            "workload_metric": "MACs",       # SNN runs will use "spikes"
            "workload_per_inf": macs,
            "params": params,
        }
        log_row(args.csv, row)
        results.append(row)

    print(f"\nAppended {len(results)} row(s) to {args.csv}")
    print("Note: workload_metric=MACs here; the SNN script logs spikes-per-inference\n"
          "into the same CSV so Step 4 can compare workload directly.")


if __name__ == "__main__":
    main()