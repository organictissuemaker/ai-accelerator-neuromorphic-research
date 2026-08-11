#!/usr/bin/env python3
"""Route B - Design-space exploration: SNN timesteps (T) vs energy and accuracy.

The neuromorphic energy analysis found the spiking network's energy to be dominated
by its dense encoding layer, whose cost scales as (conv1 MACs x T). T (the number of
simulation timesteps) is therefore a DESIGN KNOB. This script sweeps T, and for each
value trains a spiking LeNet on MNIST and records accuracy and energy per inference
(event-driven core only, and including the dense encoding). A fixed CNN is the
reference. The result is the energy-accuracy design trade-off and a recommended T.

Repo placement: energy_model/snn_timestep_sweep.py
Run (Colab, GPU):  python snn_timestep_sweep.py --epochs 3 --timesteps 5 10 15 20 25
Dependencies: torch, torchvision, snntorch, numpy, matplotlib.
"""
import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import snntorch as snn
from snntorch import surrogate

E_MAC = 4.6e-12          # J per 32-bit MAC (Horowitz 2014)
E_AC = 0.9e-12           # J per 32-bit add (SNN synaptic accumulate)


def get_loaders(data_dir, batch):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    te = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    return (DataLoader(tr, batch_size=batch, shuffle=True, num_workers=2),
            DataLoader(te, batch_size=256, shuffle=False, num_workers=2), len(te))


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2); self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120); self.fc2 = nn.Linear(120, 84); self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2); x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1); x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x)); return self.fc3(x)


def layer_macs():
    return {"conv1": 28*28*6*25, "conv2": 10*10*16*150, "fc1": 400*120, "fc2": 120*84, "fc3": 84*10}


N_PRESYN = {"conv2": 6*14*14, "fc1": 16*5*5, "fc2": 120, "fc3": 84}


class SpikingLeNet(nn.Module):
    def __init__(self, beta=0.95, spike_grad=None):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2); self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(6, 16, 5); self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc1 = nn.Linear(16*5*5, 120); self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(120, 84); self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc3 = nn.Linear(84, 10); self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x, T):
        m1 = self.lif1.init_leaky(); m2 = self.lif2.init_leaky(); m3 = self.lif3.init_leaky()
        m4 = self.lif4.init_leaky(); m5 = self.lif5.init_leaky()
        rec = []; s = {"spk1": 0.0, "spk2": 0.0, "spk3": 0.0, "spk4": 0.0}
        for _ in range(T):
            c1 = F.max_pool2d(self.conv1(x), 2); spk1, m1 = self.lif1(c1, m1)
            c2 = F.max_pool2d(self.conv2(spk1), 2); spk2, m2 = self.lif2(c2, m2)
            c3 = self.fc1(spk2.flatten(1)); spk3, m3 = self.lif3(c3, m3)
            c4 = self.fc2(spk3); spk4, m4 = self.lif4(c4, m4)
            c5 = self.fc3(spk4); spk5, m5 = self.lif5(c5, m5)
            rec.append(spk5)
            s["spk1"] += spk1.sum().item(); s["spk2"] += spk2.sum().item()
            s["spk3"] += spk3.sum().item(); s["spk4"] += spk4.sum().item()
        return torch.stack(rec), s


def dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_cnn(m, loader, d, epochs, lr):
    m.to(d).train(); opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(d), yb.to(d); opt.zero_grad(); F.cross_entropy(m(xb), yb).backward(); opt.step()
    return m


def train_snn(m, loader, d, epochs, lr, T):
    m.to(d).train(); opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(d), yb.to(d); opt.zero_grad()
            rec, _ = m(xb, T); F.cross_entropy(rec.sum(0), yb).backward(); opt.step()
    return m


@torch.no_grad()
def eval_cnn(m, loader, d, n):
    m.eval(); c = 0
    for xb, yb in loader: c += (m(xb.to(d)).argmax(1) == yb.to(d)).sum().item()
    return c / n


@torch.no_grad()
def eval_snn(m, loader, d, n, T):
    m.eval(); c = 0; tot = {"spk1": 0.0, "spk2": 0.0, "spk3": 0.0, "spk4": 0.0}
    for xb, yb in loader:
        rec, s = m(xb.to(d), T); c += (rec.sum(0).argmax(1) == yb.to(d)).sum().item()
        for k in tot: tot[k] += s[k]
    return c / n, {k: tot[k] / n for k in tot}


def snn_energy(spk, T):
    mac = layer_macs()
    sops = (spk["spk1"]/N_PRESYN["conv2"]*mac["conv2"] + spk["spk2"]/N_PRESYN["fc1"]*mac["fc1"]
            + spk["spk3"]/N_PRESYN["fc2"]*mac["fc2"] + spk["spk4"]/N_PRESYN["fc3"]*mac["fc3"])
    e_event = sops * E_AC
    e_encode = mac["conv1"] * T * E_MAC
    return e_event, e_encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--timesteps", type=int, nargs="+", default=[5, 10, 15, 20, 25])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out", default="snn_sweep_results.csv")
    args = ap.parse_args()

    torch.manual_seed(42); np.random.seed(42)
    d = dev(); tr, te, n = get_loaders(args.data_dir, 128)
    print(f"device={d} | epochs={args.epochs} | T sweep={args.timesteps}\n")

    # CNN reference
    cnn = train_cnn(LeNet(), tr, d, args.epochs, args.lr)
    cnn_acc = eval_cnn(cnn, te, d, n)
    cnn_E = sum(layer_macs().values()) * E_MAC
    print(f"CNN reference: acc {cnn_acc*100:.2f}% | energy {cnn_E*1e9:.1f} nJ\n")

    sg = surrogate.fast_sigmoid()
    rows = []
    print(f"{'T':>4}{'acc %':>9}{'event nJ':>12}{'+encode nJ':>13}")
    for T in args.timesteps:
        m = train_snn(SpikingLeNet(spike_grad=sg), tr, d, args.epochs, args.lr, T)
        acc, spk = eval_snn(m, te, d, n, T)
        e_ev, e_en = snn_energy(spk, T)
        e_full = e_ev + e_en
        rows.append({"T": T, "snn_acc": round(acc, 4), "event_nJ": round(e_ev*1e9, 1),
                     "encode_nJ": round(e_en*1e9, 1), "full_nJ": round(e_full*1e9, 1)})
        print(f"{T:>4}{acc*100:>8.2f}{e_ev*1e9:>12.1f}{e_full*1e9:>13.1f}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["T", "snn_acc", "event_nJ", "encode_nJ", "full_nJ"])
        w.writeheader(); w.writerows(rows)
    print(f"\nCNN reference (for the report): acc={cnn_acc:.4f}, energy_nJ={cnn_E*1e9:.1f}")
    print(f"saved: {args.out}")

    # figure: energy + accuracy vs T
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        Ts = [r["T"] for r in rows]
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(Ts, [r["event_nJ"] for r in rows], "-o", color="#55A868", label="SNN energy (event core)")
        ax1.plot(Ts, [r["full_nJ"] for r in rows], "-s", color="#DD8452", label="SNN energy (+ encoding)")
        ax1.axhline(cnn_E*1e9, ls="--", color="#4C72B0", label="CNN energy (ref)")
        ax1.set_xlabel("Timesteps T (design knob)"); ax1.set_ylabel("Energy per inference (nJ)"); ax1.set_yscale("log")
        ax2 = ax1.twinx()
        ax2.plot(Ts, [r["snn_acc"]*100 for r in rows], "-^", color="#8172B3", label="SNN accuracy")
        ax2.set_ylabel("Accuracy (%)")
        ax1.set_title("SNN Design Trade-off: Timesteps vs Energy and Accuracy")
        l1, la1 = ax1.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
        ax1.legend(l1+l2, la1+la2, loc="center right", fontsize=9)
        ax1.grid(True, which="both", alpha=0.3); plt.tight_layout()
        plt.savefig("snn_timestep_tradeoff.png", dpi=150)
        print("saved figure: snn_timestep_tradeoff.png")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
