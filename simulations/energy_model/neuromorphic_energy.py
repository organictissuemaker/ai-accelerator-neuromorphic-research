#!/usr/bin/env python3
"""Route B - Phase 3: neuromorphic (SNN) vs conventional (CNN) ENERGY comparison.

Trains a LeNet CNN and a matched Spiking LeNet on MNIST, then compares energy per
inference using published per-operation energy figures. The CNN cost is MACs x E_MAC.
The SNN cost is synaptic operations (SOPs) x E_AC, where

    SOPs(layer) = (avg spikes per presynaptic neuron) x (that layer's MAC count)

which is the standard identity linking event-driven SOPs to the equivalent ANN.
The first conv ("encoding") layer is driven by dense input current every timestep,
so it is NOT event-driven; we report SNN energy BOTH with and without that dense
encoding cost.

Per-operation energy (Horowitz, ISSCC 2014; 45 nm, 0.9 V):
    E_MAC = 4.6 pJ   (32-bit multiply-accumulate)
    E_AC  = 0.9 pJ   (32-bit add; an SNN synaptic op is accumulate-only)
Alt hardware anchor: Loihi ~23.6 pJ/SOP (Davies 2018).
NODE CAVEAT: 45 nm figures; absolute joules are approximate, the RATIO is the result.

Repo placement: energy_model/neuromorphic_energy.py
Run (Colab or local):  python neuromorphic_energy.py --epochs 5 --timesteps 25
Dependencies: torch, torchvision, snntorch, numpy, matplotlib.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import snntorch as snn
from snntorch import surrogate

E_MAC = 4.6e-12          # J per 32-bit MAC   (Horowitz 2014)
E_AC = 0.9e-12           # J per 32-bit add   (SNN synaptic accumulate)


# ---------------- data ----------------
def get_loaders(data_dir, batch):
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    te = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    return (DataLoader(tr, batch_size=batch, shuffle=True, num_workers=2),
            DataLoader(te, batch_size=256, shuffle=False, num_workers=2), len(te))


# ---------------- conventional CNN ----------------
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# per-layer MACs for the shared LeNet topology (one 28x28 sample)
def layer_macs():
    return {
        "conv1": 28 * 28 * 6 * (1 * 5 * 5),     # 117,600  (encoding layer)
        "conv2": 10 * 10 * 16 * (6 * 5 * 5),    # 240,000
        "fc1": 400 * 120,                       # 48,000
        "fc2": 120 * 84,                        # 10,080
        "fc3": 84 * 10,                         # 840
    }


# presynaptic neuron counts feeding each connection layer
N_PRESYN = {"conv2": 6 * 14 * 14, "fc1": 16 * 5 * 5, "fc2": 120, "fc3": 84}


# ---------------- spiking LeNet with per-layer spike counters ----------------
class SpikingLeNet(nn.Module):
    def __init__(self, beta=0.95, spike_grad=None):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(120, 84)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc3 = nn.Linear(84, 10)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x, T):
        m1 = self.lif1.init_leaky(); m2 = self.lif2.init_leaky()
        m3 = self.lif3.init_leaky(); m4 = self.lif4.init_leaky()
        m5 = self.lif5.init_leaky()
        rec = []
        # per-layer spike totals for THIS batch (summed over neurons + timesteps)
        s = {"spk1": 0.0, "spk2": 0.0, "spk3": 0.0, "spk4": 0.0}
        for _ in range(T):
            c1 = F.max_pool2d(self.conv1(x), 2)
            spk1, m1 = self.lif1(c1, m1)
            c2 = F.max_pool2d(self.conv2(spk1), 2)
            spk2, m2 = self.lif2(c2, m2)
            c3 = self.fc1(spk2.flatten(1))
            spk3, m3 = self.lif3(c3, m3)
            c4 = self.fc2(spk3)
            spk4, m4 = self.lif4(c4, m4)
            c5 = self.fc3(spk4)
            spk5, m5 = self.lif5(c5, m5)
            rec.append(spk5)
            s["spk1"] += spk1.sum().item(); s["spk2"] += spk2.sum().item()
            s["spk3"] += spk3.sum().item(); s["spk4"] += spk4.sum().item()
        return torch.stack(rec), s


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_cnn(model, loader, dev, epochs, lr):
    model.to(dev).train(); opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); F.cross_entropy(model(xb), yb).backward(); opt.step()
    return model


def train_snn(model, loader, dev, epochs, lr, T):
    model.to(dev).train(); opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            rec, _ = model(xb, T)
            F.cross_entropy(rec.sum(0), yb).backward(); opt.step()
    return model


@torch.no_grad()
def eval_cnn(model, loader, dev, n):
    model.eval(); correct = 0
    for xb, yb in loader:
        correct += (model(xb.to(dev)).argmax(1) == yb.to(dev)).sum().item()
    return correct / n


@torch.no_grad()
def eval_snn(model, loader, dev, n, T):
    model.eval(); correct = 0
    tot = {"spk1": 0.0, "spk2": 0.0, "spk3": 0.0, "spk4": 0.0}
    for xb, yb in loader:
        rec, s = model(xb.to(dev), T)
        correct += (rec.sum(0).argmax(1) == yb.to(dev)).sum().item()
        for k in tot:
            tot[k] += s[k]
    spikes_per_inf = {k: tot[k] / n for k in tot}
    return correct / n, spikes_per_inf


def snn_energy(spikes_per_inf, T):
    macs = layer_macs()
    # event-driven SOPs = avg spikes per presynaptic neuron x layer MACs
    sops = {
        "conv2": spikes_per_inf["spk1"] / N_PRESYN["conv2"] * macs["conv2"],
        "fc1": spikes_per_inf["spk2"] / N_PRESYN["fc1"] * macs["fc1"],
        "fc2": spikes_per_inf["spk3"] / N_PRESYN["fc2"] * macs["fc2"],
        "fc3": spikes_per_inf["spk4"] / N_PRESYN["fc3"] * macs["fc3"],
    }
    total_sops = sum(sops.values())
    e_event = total_sops * E_AC
    e_encode = macs["conv1"] * T * E_MAC          # dense conv1, every timestep
    return total_sops, e_event, e_encode, sops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--timesteps", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out", default="../results/figures")
    args = ap.parse_args()

    torch.manual_seed(42); np.random.seed(42)
    dev = device()
    tr, te, n = get_loaders(args.data_dir, 128)
    print(f"device={dev} | MNIST test n={n} | T={args.timesteps}\n")

    # CNN
    cnn = train_cnn(LeNet(), tr, dev, args.epochs, args.lr)
    cnn_acc = eval_cnn(cnn, te, dev, n)
    cnn_macs = sum(layer_macs().values())
    cnn_E = cnn_macs * E_MAC

    # SNN
    sg = surrogate.fast_sigmoid()
    snn_model = train_snn(SpikingLeNet(spike_grad=sg), tr, dev, args.epochs, args.lr, args.timesteps)
    snn_acc, spk = eval_snn(snn_model, te, dev, n, args.timesteps)
    total_sops, e_event, e_encode, sops = snn_energy(spk, args.timesteps)
    snn_E_event = e_event
    snn_E_full = e_event + e_encode

    print("=" * 66)
    print(" ENERGY PER INFERENCE  (MNIST, LeNet vs Spiking LeNet)")
    print("=" * 66)
    print(f" CNN  : acc {cnn_acc*100:5.2f}% | {cnn_macs:,} MACs | {cnn_E*1e9:8.2f} nJ")
    print(f" SNN  : acc {snn_acc*100:5.2f}% | {total_sops:,.0f} SOPs")
    print(f"        spikes/inf: {', '.join(f'{k}={v:.0f}' for k,v in spk.items())}")
    print(f"        event-only        : {snn_E_event*1e9:8.2f} nJ  "
          f"({cnn_E/snn_E_event:5.1f}x vs CNN)")
    print(f"        + dense encoding  : {snn_E_full*1e9:8.2f} nJ  "
          f"({cnn_E/snn_E_full:5.1f}x vs CNN)")
    print(f"        (encoding cost alone: {e_encode*1e9:.2f} nJ = conv1 MACs x T)")

    # figure + csv
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.out, exist_ok=True)
        labels = ["CNN\n(dense MACs)", "SNN\n(event only)", "SNN\n(+ encoding)"]
        vals = [cnn_E * 1e9, snn_E_event * 1e9, snn_E_full * 1e9]
        colors = ["#4C72B0", "#55A868", "#DD8452"]
        plt.figure(figsize=(7, 5))
        b = plt.bar(labels, vals, color=colors)
        for bar, v in zip(b, vals):
            plt.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f} nJ",
                     ha="center", va="bottom", fontsize=10)
        plt.ylabel("Energy per inference (nJ, lower = better)")
        plt.title(f"MNIST Energy per Inference: CNN vs SNN (T={args.timesteps})")
        plt.grid(True, axis="y", alpha=0.3); plt.margins(y=0.15)
        f = os.path.join(args.out, "neuromorphic_energy.png")
        plt.tight_layout(); plt.savefig(f, dpi=150)
        print(f"\n saved figure: {f}")
    except Exception as e:
        print(f" (figure skipped: {e})")


if __name__ == "__main__":
    main()
