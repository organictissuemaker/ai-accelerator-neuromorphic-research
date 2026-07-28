"""
Train a small SNN on MNIST (Part 2 — the framework baseline)
------------------------------------------------------------
Trains a 784 -> HIDDEN -> 10 leaky-integrate-and-fire network with snnTorch,
using SURROGATE-GRADIENT backpropagation-through-time (never hand-coded — that
is the whole reason to use a framework here). Locks the baseline accuracy that
BOTH threads need:
  * Thread A  (AI-vs-CPU benchmark) — same trained model, two execution models.
  * Thread B  (Part 3 variability sweep) — perturb THESE weights: W -> W*(1+N(0,sigma)).

Expected: ~96% test accuracy. Also reports average spikes/inference per layer,
the raw material for the Thread-A event-synop count.

Run:  python train_snn_mnist.py            # defaults below
      python train_snn_mnist.py --epochs 3 --hidden 512   # faster
"""

import os
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import snntorch as snn
from snntorch import surrogate

# ── Paths (mirror lif_neuron.py's convention) ───────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR  = os.path.normpath(os.path.join(SCRIPT_DIR, "../../results/figures"))
RESULTS_DIR  = os.path.normpath(os.path.join(SCRIPT_DIR, "../../results"))
DATA_DIR     = os.path.normpath(os.path.join(SCRIPT_DIR, "../../data"))
for d in (FIGURES_DIR, RESULTS_DIR, DATA_DIR):
    os.makedirs(d, exist_ok=True)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")          # Apple Silicon
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SNN(nn.Module):
    """784 -> hidden -> 10 LIF network. Static image injected each of num_steps."""
    def __init__(self, hidden=1000, beta=0.95, num_steps=25):
        super().__init__()
        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid()          # surrogate gradient
        self.fc1  = nn.Linear(28 * 28, hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2  = nn.Linear(hidden, 10)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        """x: (B, 1, 28, 28). Returns summed output spikes and per-layer spike totals."""
        x = x.view(x.size(0), -1)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        out_sum = torch.zeros(x.size(0), 10, device=x.device)
        n_hidden_spk = torch.zeros(x.size(0), device=x.device)
        n_out_spk    = torch.zeros(x.size(0), device=x.device)

        for _ in range(self.num_steps):
            cur1 = self.fc1(x)                 # direct/current input coding
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            out_sum += spk2
            n_hidden_spk += spk1.sum(dim=1)
            n_out_spk    += spk2.sum(dim=1)
        return out_sum, n_hidden_spk, n_out_spk


def loaders(batch=128):
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=tf)
    te = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    return (DataLoader(tr, batch_size=batch, shuffle=True,  drop_last=True),
            DataLoader(te, batch_size=256, shuffle=False))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    hidden_spk = out_spk = 0.0
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        out, nh, no = model(data)
        correct += (out.argmax(1) == targets).sum().item()
        total   += targets.size(0)
        hidden_spk += nh.sum().item()
        out_spk    += no.sum().item()
    acc = 100.0 * correct / total
    return acc, hidden_spk / total, out_spk / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=1000)
    ap.add_argument("--num_steps", type=int, default=25)
    ap.add_argument("--beta", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = get_device()
    print("=" * 60)
    print(" Train SNN on MNIST  (Part 2 baseline)")
    print("=" * 60)
    print(f" torch {torch.__version__} | snntorch {snn.__version__} | device: {device}")
    print(f" arch 784->{args.hidden}->10 | T={args.num_steps} | beta={args.beta} "
          f"| epochs={args.epochs}\n")

    train_loader, test_loader = loaders(args.batch)
    model = SNN(args.hidden, args.beta, args.num_steps).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    loss_fn = nn.CrossEntropyLoss()            # applied to summed output spikes

    hist = []
    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for i, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            out, _, _ = model(data)
            loss = loss_fn(out, targets)       # spike-count logits
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        acc, h_spk, o_spk = evaluate(model, test_loader, device)
        hist.append((ep, running / len(train_loader), acc))
        print(f" epoch {ep:2d} | loss {running/len(train_loader):.4f} "
              f"| test acc {acc:.2f}% | spikes/inf: hidden {h_spk:.1f}, out {o_spk:.1f}")

    acc, h_spk, o_spk = evaluate(model, test_loader, device)

    # ── Thread-A raw material: event-synop accounting ───────────────────────
    # fc2 (hidden->out) is genuinely event-driven: only hidden spikes drive it.
    #   event synops (fc2) = avg_hidden_spikes * 10 output targets
    # fc1 is driven by the static image as current each step (encoding layer),
    #   so it is dense: 784 * hidden MACs * num_steps. Flagged for the prof.
    ev_synops_fc2 = h_spk * 10
    dense_fc2     = args.hidden * 10 * args.num_steps
    dense_fc1     = 28 * 28 * args.hidden * args.num_steps

    print("\n" + "=" * 60)
    print(" BASELINE — record these numbers")
    print("=" * 60)
    print(f" Test accuracy               : {acc:.2f}%")
    print(f" Architecture                : 784 -> {args.hidden} -> 10")
    print(f" Timesteps (T)               : {args.num_steps}")
    print(f" Avg spikes/inference        : hidden {h_spk:.1f} | output {o_spk:.1f}")
    print(f" fc2 event synops/inference  : {ev_synops_fc2:,.0f}  "
          f"(vs {dense_fc2:,} dense clocked  = {dense_fc2/max(ev_synops_fc2,1):.0f}x)")
    print(f" fc1 (encoding) dense MACs   : {dense_fc1:,}  "
          f"(analog-current input — NOT event-driven; confirm treatment w/ prof)")

    # Save weights (the object Part 3 perturbs) + training curve.
    ckpt = os.path.join(RESULTS_DIR, "snn_mnist_model.pt")
    torch.save({"state_dict": model.state_dict(),
                "hidden": args.hidden, "beta": args.beta,
                "num_steps": args.num_steps, "test_acc": acc}, ckpt)
    print(f"\n Saved model : {ckpt}")

    try:
        import matplotlib.pyplot as plt
        ep_a = [h[0] for h in hist]; ls = [h[1] for h in hist]; ac = [h[2] for h in hist]
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        ax2 = ax1.twinx()
        ax1.plot(ep_a, ls, "-o", color="#2F5496", label="train loss")
        ax2.plot(ep_a, ac, "-o", color="#F2A900", label="test acc")
        ax1.set_xlabel("Epoch", fontfamily="Georgia", fontweight="bold")
        ax1.set_ylabel("Train loss", fontfamily="Georgia", fontweight="bold", color="#2F5496")
        ax2.set_ylabel("Test accuracy (%)", fontfamily="Georgia", fontweight="bold", color="#B8860B")
        ax1.set_title(f"SNN on MNIST — baseline {acc:.2f}%  (784->{args.hidden}->10, T={args.num_steps})",
                      fontfamily="Georgia", fontweight="bold", fontsize=12)
        for ax in (ax1, ax2):
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontfamily("Georgia"); lbl.set_fontweight("bold"); lbl.set_fontsize(8)
        plt.tight_layout()
        out = os.path.join(FIGURES_DIR, "snn_mnist_training.png")
        plt.savefig(out, dpi=150)
        print(f" Saved figure: {out}")
    except Exception as e:
        print(f" (figure skipped: {e})")


if __name__ == "__main__":
    main()
