#!/usr/bin/env python3
"""
Route A - Step 2: Spiking neural network (SNN) baseline + spike-count proxy.

Trains a spiking LeNet (Leaky Integrate-and-Fire neurons) on MNIST with
surrogate-gradient backprop-through-time, then benchmarks inference and counts
the TOTAL number of spikes emitted across the network. The neuromorphic energy
proxy is spikes-per-inference = total spikes / test samples (Step 3).

Logs into the SAME CSV schema as the CNN baseline (cnn_baseline.py), but with
workload_metric="spikes" instead of "MACs", so Step 4 can build the comparison
table and Step 5 the spikes-per-inference figure directly from one file.

Keep the test set and batch size IDENTICAL to the CNN run so the two arms are
comparable (defaults match cnn_baseline.py: MNIST test set, batch_size=128).

Usage
-----
    python snn_baseline.py                       # train + benchmark, T=25
    python snn_baseline.py --timesteps 10        # fewer timesteps
    python snn_baseline.py --device cpu          # one device (cpu/cuda/mps)
    python snn_baseline.py --weights snn_lenet.pt  # reuse saved weights

Dependencies: torch, torchvision, snntorch, numpy.
Repo placement: simulations/snn/snn_baseline.py
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

import snntorch as snn
from snntorch import surrogate


# --------------------------------------------------------------------------- #
# Device helpers (mirror cnn_baseline.py)
# --------------------------------------------------------------------------- #
def mps_available() -> bool:
    return getattr(torch.backends, "mps", None) is not None and \
        torch.backends.mps.is_available()


def sync(device):
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
# Model: spiking LeNet
# --------------------------------------------------------------------------- #
class SpikingLeNet(nn.Module):
    """LeNet topology with LIF neurons after each layer.

    Input encoding is 'direct': the static (normalized) image is presented as
    constant input current at every timestep, and the first conv+LIF layer acts
    as the learned encoder. This is the snnTorch-recommended scheme and gives
    higher accuracy at low T than Poisson rate coding.
    """

    def __init__(self, beta=0.95, spike_grad=None, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)   # -> 6x28x28
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)             # -> 16x10x10
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(120, 84)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc3 = nn.Linear(84, num_classes)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x, num_steps):
        """Run T timesteps.

        Returns
        -------
        spk5_rec : (T, B, num_classes) stacked output spikes (for loss/accuracy)
        total_spikes : scalar tensor, summed spikes over ALL LIF layers,
                       all timesteps, all samples in the batch (energy proxy)
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()

        spk5_rec = []
        total_spikes = torch.zeros((), device=x.device)

        for _ in range(num_steps):
            cur1 = F.max_pool2d(self.conv1(x), 2)          # 6x14x14
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = F.max_pool2d(self.conv2(spk1), 2)       # 16x5x5
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc1(spk2.flatten(1))
            spk3, mem3 = self.lif3(cur3, mem3)
            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)
            cur5 = self.fc3(spk4)
            spk5, mem5 = self.lif5(cur5, mem5)

            spk5_rec.append(spk5)
            # Count every binary spike event across the network (energy proxy).
            total_spikes = total_spikes + spk1.sum() + spk2.sum() + \
                spk3.sum() + spk4.sum() + spk5.sum()

        return torch.stack(spk5_rec), total_spikes


# --------------------------------------------------------------------------- #
# Data (identical preprocessing to the CNN baseline)
# --------------------------------------------------------------------------- #
def get_loaders(data_dir: str, batch_size: int):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=pin)
    return train_loader, test_loader, len(test_set)


# --------------------------------------------------------------------------- #
# Training (BPTT with surrogate gradients)
# --------------------------------------------------------------------------- #
def train(model, loader, device, epochs, lr, num_steps):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        running, seen = 0.0, 0
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            opt.zero_grad()
            spk_rec, _ = model(images, num_steps)
            # Sum output spikes over time -> use as class logits for CE loss.
            logits = spk_rec.sum(dim=0)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            opt.step()
            running += loss.item() * images.size(0)
            seen += images.size(0)
        print(f"  epoch {epoch}/{epochs}  train_loss={running / seen:.4f}")
    return model


# --------------------------------------------------------------------------- #
# Benchmark (accuracy + timing + total spikes)
# --------------------------------------------------------------------------- #
def benchmark(model, loader, device, n_samples, num_steps, warmup_batches=3):
    model.to(device).eval()

    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            model(images.to(device), num_steps)
            if i + 1 >= warmup_batches:
                break
    sync(device)

    correct = 0
    total_time = 0.0
    spike_total = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            sync(device)
            t0 = time.perf_counter()
            spk_rec, batch_spikes = model(images, num_steps)
            sync(device)
            total_time += time.perf_counter() - t0
            preds = spk_rec.sum(dim=0).argmax(1)
            correct += (preds == targets).sum().item()
            spike_total += batch_spikes.item()

    accuracy = correct / n_samples
    latency_ms = (total_time / n_samples) * 1e3
    throughput = n_samples / total_time
    spikes_per_inf = spike_total / n_samples
    return accuracy, latency_ms, throughput, spikes_per_inf


def device_label(device):
    if device.type == "cuda":
        return f"GPU ({torch.cuda.get_device_name(device)})"
    if device.type == "mps":
        return "GPU (Apple MPS)"
    return f"CPU ({platform.processor() or platform.machine()})"


# --------------------------------------------------------------------------- #
# Logging (same schema/file as cnn_baseline.py)
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
    p = argparse.ArgumentParser(description="Spiking LeNet + spike-count proxy")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128,
                   help="Keep IDENTICAL to the CNN run.")
    p.add_argument("--timesteps", type=int, default=25,
                   help="T: number of simulation timesteps per inference.")
    p.add_argument("--beta", type=float, default=0.95,
                   help="LIF membrane decay rate.")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--weights", default=None,
                   help="Path to load/save model weights (.pt).")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"],
                   default="auto")
    p.add_argument("--csv", default="../../results/metrics.csv")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader, test_loader, n_test = get_loaders(args.data_dir, args.batch_size)

    spike_grad = surrogate.fast_sigmoid()
    model = SpikingLeNet(beta=args.beta, spike_grad=spike_grad)

    train_device = best_train_device()
    if args.weights and os.path.exists(args.weights):
        print(f"Loading weights from {args.weights}")
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    else:
        print(f"Training SpikingLeNet on {train_device.type} "
              f"for {args.epochs} epoch(s), T={args.timesteps}...")
        train(model, train_loader, train_device, args.epochs, args.lr,
              args.timesteps)
        if args.weights:
            model.to("cpu")
            torch.save(model.state_dict(), args.weights)
            print(f"Saved weights to {args.weights}")
    model.to("cpu")

    params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: SpikingLeNet(T={args.timesteps})  |  params={params:,}")

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
    print(f"Test set: {n_test} samples, batch_size={args.batch_size}, "
          f"T={args.timesteps}\n")

    results = []
    for dev in devices:
        acc, lat, thr, spk = benchmark(model, test_loader, dev, n_test,
                                       args.timesteps)
        label = device_label(dev)
        print(f"[{dev.type.upper():4}] acc={acc*100:5.2f}%  "
              f"latency={lat:.4f} ms/inf  throughput={thr:,.0f} inf/s  "
              f"spikes/inf={spk:,.1f}")
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "system": platform.platform(),
            "device_type": dev.type,
            "device_name": label,
            "model": f"SpikingLeNet(T={args.timesteps})",
            "dataset": "MNIST",
            "batch_size": args.batch_size,
            "test_samples": n_test,
            "accuracy": round(acc, 4),
            "latency_ms_per_inf": round(lat, 4),
            "throughput_inf_per_s": round(thr, 1),
            "workload_metric": "spikes",
            "workload_per_inf": round(spk, 1),
            "params": params,
        }
        log_row(args.csv, row)
        results.append(row)

    print(f"\nAppended {len(results)} row(s) to {args.csv}")
    print("spikes-per-inference is the neuromorphic energy proxy (Step 3).\n"
          "It is device-independent, so all rows share the same value; the\n"
          "per-device latency/throughput differ.")


if __name__ == "__main__":
    main()
