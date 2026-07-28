"""
Validate snnTorch against the hand-built LIF neuron
---------------------------------------------------
Part 2 gate. Before trusting any framework-trained SNN, confirm that snnTorch's
LIF neuron reproduces the hand-built neuron from `lif_neuron.py` on a controlled
single-neuron test: constant 3 nA drive, no refractory (the "Neuron 1" model).

TARGET (from the hand-built model): steady inter-spike interval = 22 ms, i.e.
45.5 Hz on the clean interval. snnTorch should match spike-for-spike to within
one simulation timestep (DT = 0.1 ms).

THE BRIDGE (physical units -> snnTorch's normalized model)
    Hand-built forward-Euler update, shifted to u = V - V_REST:
        u[t+1] = (1 - DT/TAU_M) * u[t] + (DT/TAU_M) * R_M * I
    Mapping onto snn.Leaky(mem[t+1] = beta*mem[t] + cur):
        beta      = 1 - DT/TAU_M                 (Euler decay, NOT exp(-DT/TAU_M))
        threshold = V_THRESH - V_REST            (=20 mV)
        cur       = (DT/TAU_M) * R_M * I         (constant input per step)
        reset     = "zero"                       (hand-built resets V -> V_REST => u=0)
This mapping is exact on paper; this script confirms it against the real library.

Run:  python validate_snntorch_vs_handbuilt.py
"""

import os
import numpy as np

# Reuse the EXACT parameters of the hand-built neuron (single source of truth).
from lif_neuron import TAU_M, V_REST, V_THRESH, V_RESET, R_M, DT, T_SIM

import torch
import snntorch as snn

I_DRIVE = 3e-9          # 3 nA — the drive that gives the 22 ms / 45.5 Hz reference
TOL_STEPS = 1.0         # pass if spike-time mismatch <= 1 timestep

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../results/figures"))
os.makedirs(FIGURES_DIR, exist_ok=True)


def handbuilt_reference(I=I_DRIVE):
    """Bare physical LIF (forward Euler, no refractory) — the 'Neuron 1' model."""
    t = np.arange(0, T_SIM, DT)
    V = V_REST
    spikes = []
    for i in range(1, len(t)):
        V = V + (-(V - V_REST) + R_M * I) / TAU_M * DT
        if V >= V_THRESH:
            spikes.append(t[i])
            V = V_RESET
    return t, np.array(spikes)


def snntorch_neuron(I=I_DRIVE):
    """snn.Leaky driven by the mapped constant current; returns spike times (s)."""
    beta = 1.0 - DT / TAU_M                     # 0.995
    threshold = V_THRESH - V_REST               # 0.020 V
    cur = (DT / TAU_M) * R_M * I                # constant input per step

    lif = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism="zero")
    mem = torch.zeros(1)
    cur_t = torch.tensor([cur], dtype=torch.float32)

    t = np.arange(0, T_SIM, DT)
    spikes = []
    for i in range(1, len(t)):
        spk, mem = lif(cur_t, mem)
        if spk.item() > 0:
            spikes.append(t[i])
    return np.array(spikes), beta, threshold, cur


def rate_from_isi(spikes):
    """Rate on the CLEAN interval = 1 / mean ISI (ignores from-rest startup)."""
    if len(spikes) < 2:
        return float("nan"), float("nan")
    isi_ms = np.diff(spikes) * 1000.0
    return 1000.0 / isi_ms.mean(), isi_ms.mean()


def main():
    print("=" * 62)
    print(" Validate snnTorch  vs  hand-built LIF   (single neuron, 3 nA)")
    print("=" * 62)
    print(f" torch {torch.__version__} | snntorch {snn.__version__} | "
          f"DT={DT*1000:.2f} ms  T_SIM={T_SIM*1000:.0f} ms\n")

    t, ref = handbuilt_reference()
    snt, beta, thr, cur = snntorch_neuron()

    ref_rate, ref_isi = rate_from_isi(ref)
    snt_rate, snt_isi = rate_from_isi(snt)

    print(f" Mapping:  beta={beta:.4f}  threshold={thr*1000:.1f} mV  "
          f"cur/step={cur:.3e}\n")
    print(f" {'':22s}{'spikes':>8s}{'rate (Hz)':>12s}{'ISI (ms)':>11s}")
    print(f" {'hand-built ref':22s}{len(ref):>8d}{ref_rate:>12.2f}{ref_isi:>11.3f}")
    print(f" {'snnTorch':22s}{len(snt):>8d}{snt_rate:>12.2f}{snt_isi:>11.3f}")

    # Spike-time agreement over the overlap.
    n = min(len(ref), len(snt))
    if n >= 1:
        dmax_ms = np.abs(ref[:n] - snt[:n]).max() * 1000.0
    else:
        dmax_ms = float("nan")
    dmax_steps = dmax_ms / (DT * 1000.0)

    print(f"\n Max spike-time difference: {dmax_ms:.4f} ms "
          f"({dmax_steps:.2f} timesteps)")
    print(f" Spike-count difference   : {abs(len(ref) - len(snt))}")

    ok = (abs(len(ref) - len(snt)) <= 1) and (dmax_steps <= TOL_STEPS + 1e-9)
    print("\n " + ("PASS  " if ok else "FAIL  ") +
          f"(tolerance = {TOL_STEPS:.0f} timestep)  ->  "
          "snnTorch reproduces the hand-built neuron"
          if ok else
          "\n FAIL  -> mismatch exceeds one timestep; check beta / threshold / reset")

    # Optional overlay figure (house style).
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 3.2))
        ax.eventplot([ref * 1000, snt * 1000], lineoffsets=[1, 0],
                     linelengths=0.7, colors=["#2F5496", "#F2A900"])
        ax.set_yticks([1, 0]); ax.set_yticklabels(["hand-built", "snnTorch"])
        ax.set_xlabel("Time (ms)")
        ax.set_title(f"snnTorch vs hand-built LIF  —  ISI {ref_isi:.1f} ms "
                     f"({ref_rate:.1f} Hz), max diff {dmax_ms:.2f} ms",
                     fontfamily="Georgia", fontweight="bold", fontsize=12)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily("Georgia"); lbl.set_fontweight("bold"); lbl.set_fontsize(8)
        plt.tight_layout()
        out = os.path.join(FIGURES_DIR, "snntorch_validation.png")
        plt.savefig(out, dpi=150)
        print(f"\n Saved figure: {out}")
    except Exception as e:
        print(f" (figure skipped: {e})")

    return ok


if __name__ == "__main__":
    main()
