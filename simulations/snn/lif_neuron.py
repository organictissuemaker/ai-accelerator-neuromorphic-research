"""
Leaky Integrate-and-Fire (LIF) Neuron Simulator
------------------------------------------------
A basic SNN building block. Simulates how a biological neuron
integrates input current and fires a spike when voltage threshold is crossed.

Contents:
  1. simulate_two_neurons()      — 2-neuron feedforward chain (Week 3)
  2. simulate_feedforward_layer() — N_IN -> N_OUT layer (Week 4)
  3. layer_metrics()             — timestep-invariant sparsity / synop accounting

The weight matrix W built in (2) is the object that the Week-5+ variability
experiment perturbs: W -> W * (1 + N(0, sigma)) with sigma taken from measured
RRAM device statistics. Everything here is scaffolding for that.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ── Resolve output directory relative to this script's location ──────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../results/figures"))
os.makedirs(FIGURES_DIR, exist_ok=True)  # create if it doesn't exist


# ── LIF Neuron Parameters ────────────────────────────────────────────────────
TAU_M   = 20e-3   # membrane time constant (seconds)
V_REST  = -70e-3  # resting membrane potential (volts)
V_THRESH = -50e-3 # firing threshold (volts)
V_RESET  = -70e-3 # reset potential after spike (volts)
R_M     = 10e6    # membrane resistance (ohms)
DT      = 1e-4    # simulation time step (seconds)
T_SIM   = 0.5     # total simulation time (seconds)

TAU_SYN = 5e-3
W_SYN = 10e-9


# ── Two Connected Neurons ─────────────────────────────────────────────────────
def simulate_two_neurons(I_input):
    """
    Simulate two LIF neurons connected in series (feedforward SNN layer).

    Neuron 1: receives constant input current
    Neuron 2: receives no direct input — only synaptic current from Neuron 1's spikes.

    When Neuron 1 fires:
      - A current pulse of size W_SYN is added to I_syn
      - I_syn decays exponentially with time constant TAU_SYN
      - Neuron 2 integrates I_syn the same way it would integrate any input current

    Parameters:
        I_input (float): Constant input current into Neuron 1 (amperes)

    Returns:
        t       : time array
        V1      : Neuron 1 membrane voltage array
        spikes1 : Neuron 1 spike times
        V2      : Neuron 2 membrane voltage array
        spikes2 : Neuron 2 spike times
    """
    t = np.arange(0, T_SIM, DT)

    # Neuron 1 state
    V1 = np.full(len(t), V_REST)
    spikes1 = []

    # Neuron 2 state
    V2 = np.full(len(t), V_REST)
    spikes2 = []

    # Synaptic current flowing from Neuron 1 into Neuron 2
    I_syn = 0.0

    for i in range(1, len(t)):

        # ── Neuron 1: integrate constant input current ──
        dV1 = (-(V1[i-1] - V_REST) + R_M * I_input) / TAU_M * DT
        V1[i] = V1[i-1] + dV1

        if V1[i] >= V_THRESH:
            spikes1.append(t[i])
            V1[i] = V_RESET
            I_syn += W_SYN          # spike → inject current pulse into Neuron 2

        # ── Synaptic current decays exponentially each timestep ──
        # Like a capacitor discharging: the pulse fades with time constant TAU_SYN
        I_syn -= (I_syn / TAU_SYN) * DT

        # ── Neuron 2: integrate synaptic input from Neuron 1 ──
        dV2 = (-(V2[i-1] - V_REST) + R_M * I_syn) / TAU_M * DT
        V2[i] = V2[i-1] + dV2

        if V2[i] >= V_THRESH:
            spikes2.append(t[i])
            V2[i] = V_RESET

    return t, V1, spikes1, V2, spikes2


def plot_two_neurons(t, V1, spikes1, V2, spikes2, I_input):
    """Plot both neuron voltage traces stacked vertically."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # ── Neuron 1 ──
    ax1.plot(t * 1000, np.array(V1) * 1000, color="#2F5496", linewidth=1.5, label="Membrane Voltage")
    ax1.axhline(V_THRESH * 1000, color="red", linestyle="--", linewidth=1, label=f"Threshold ({V_THRESH*1000:.0f} mV)")
    ax1.axhline(V_REST * 1000, color="grey", linestyle=":", linewidth=1, label=f"Rest ({V_REST*1000:.0f} mV)")
    for s in spikes1:
        ax1.axvline(s * 1000, color="orange", alpha=0.5, linewidth=0.8)
    ax1.set_ylabel("Voltage (mV)")
    ax1.set_title(f"Neuron 1 — Constant Input: {I_input*1e9:.0f} nA | {len(spikes1)} spikes | {len(spikes1)/T_SIM:.1f} Hz")
    ax1.legend(loc="upper right", fontsize=8)

    # ── Neuron 2 ──
    ax2.plot(t * 1000, np.array(V2) * 1000, color="#375623", linewidth=1.5, label="Membrane Voltage")
    ax2.axhline(V_THRESH * 1000, color="red", linestyle="--", linewidth=1, label=f"Threshold ({V_THRESH*1000:.0f} mV)")
    ax2.axhline(V_REST * 1000, color="grey", linestyle=":", linewidth=1, label=f"Rest ({V_REST*1000:.0f} mV)")
    for s in spikes2:
        ax2.axvline(s * 1000, color="purple", alpha=0.5, linewidth=0.8)
    ax2.set_ylabel("Voltage (mV)")
    ax2.set_xlabel("Time (ms)")
    ax2.set_title(f"Neuron 2 — Driven by N1 spikes (W={W_SYN*1e9:.1f} nA, τ={TAU_SYN*1000:.0f} ms) | {len(spikes2)} spikes | {len(spikes2)/T_SIM:.1f} Hz")
    ax2.legend(loc="upper right", fontsize=8)

    plt.suptitle("Two Connected LIF Neurons — Feedforward SNN Layer", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "snn_two_neurons.png"), dpi=150)
    plt.show()

    print(f"\nNeuron 1: {len(spikes1)} spikes | {len(spikes1)/T_SIM:.1f} Hz")
    print(f"Neuron 2: {len(spikes2)} spikes | {len(spikes2)/T_SIM:.1f} Hz")
    if spikes2:
        print(f"First Neuron 2 spike at: {spikes2[0]*1000:.1f} ms")
    else:
        print("Neuron 2 never reached threshold — try increasing W_SYN")


# ─────────────────────────────────────────────────────────────────────────────
#  N-NEURON FEEDFORWARD LAYER
#  Guided by Eshraghian et al. 2021 (arXiv:2109.12894).
#
#  Architecture:  N_IN input neurons  →  [weight matrix W]  →  N_OUT output neurons
#    - Each input neuron gets its own constant drive (heterogeneous rates).
#    - Every input spike injects W[j, i] of synaptic current into output neuron j.
#    - I_syn per output neuron decays with TAU_SYN (same as the 2-neuron model).
#    - Fully vectorised: the per-neuron state is a NumPy vector, not a Python loop.
#
#  Purpose:  observe EVENT-DRIVEN SPARSITY — computation happens only when a
#  spike occurs. This is why event-driven hardware (Loihi) is efficient:
#  no spike → no synaptic operation → no energy.
# ─────────────────────────────────────────────────────────────────────────────

T_REF = 3e-3  # absolute refractory period (s) — neuron can't re-fire during this window

N_IN  = 10   # input-layer neurons
N_OUT = 10   # output-layer neurons  (the "10-neuron layer" success target)

P_CONNECT  = 0.6    # connection probability (fraction of the 100 synapses that exist)
W_FRAC_MIN = 0.15   # synaptic weights drawn uniformly from
W_FRAC_MAX = 0.45   #   [W_FRAC_MIN, W_FRAC_MAX] x W_SYN
# NOTE on weight scale: with weights ~1.33x larger than this, convergent excitation
# drives the output layer to a HIGHER mean rate than the input layer (43.6 Hz out vs
# 30.4 Hz in). That is physically legitimate, but it makes the layer an amplifier,
# which muddies the sparsity argument. This range holds output rate at or just below
# input rate, so the layer propagates activity without inflating it.


def simulate_feedforward_layer(seed=3):
    """
    Simulate a single feedforward SNN layer: N_IN → N_OUT LIF neurons.

    Returns:
        t          : (T,)          time array
        in_spikes  : (T, N_IN)  bool  input-layer spike raster
        out_spikes : (T, N_OUT) bool  output-layer spike raster
        V_out_rec  : (T, N_OUT) float output membrane voltages (for inspection)
        W          : (N_OUT, N_IN) synaptic weight matrix used
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, T_SIM, DT)
    T = len(t)

    # Threshold current: I such that R_M*I clears (V_THRESH - V_REST).
    #   I_th = (V_THRESH - V_REST)/R_M = 20 mV / 10 MΩ = 2 nA
    I_th = (V_THRESH - V_REST) / R_M
    # Heterogeneous drive: input neurons span 1.05x–1.6x threshold → a spread of rates
    # (roughly 14–44 Hz with the parameters above).
    I_in = np.linspace(1.05 * I_th, 1.6 * I_th, N_IN)

    # Sparse, random synaptic weights (post × pre).
    mask = rng.random((N_OUT, N_IN)) < P_CONNECT
    W = mask * rng.uniform(W_FRAC_MIN * W_SYN, W_FRAC_MAX * W_SYN, size=(N_OUT, N_IN))

    # State vectors
    V_in  = np.full(N_IN,  V_REST)
    V_out = np.full(N_OUT, V_REST)
    I_syn = np.zeros(N_OUT)   # synaptic current into each output neuron
    ref_in  = np.zeros(N_IN)   # refractory countdown (s) per input neuron
    ref_out = np.zeros(N_OUT)  # refractory countdown (s) per output neuron

    in_spikes  = np.zeros((T, N_IN),  dtype=bool)
    out_spikes = np.zeros((T, N_OUT), dtype=bool)
    V_out_rec  = np.zeros((T, N_OUT))
    V_out_rec[0] = V_out

    for i in range(1, T):
        # ── Input layer: integrate constant drive (vectorised LIF) ──
        active_in = ref_in <= 0                      # neurons not in refractory
        V_in[active_in] += (-(V_in[active_in] - V_REST) + R_M * I_in[active_in]) / TAU_M * DT
        ref_in = np.maximum(ref_in - DT, 0.0)
        fired_in = (V_in >= V_THRESH) & active_in
        V_in[fired_in] = V_RESET
        ref_in[fired_in] = T_REF
        in_spikes[i] = fired_in

        # ── Synapses: each input spike injects weighted current, then decay ──
        I_syn += W @ fired_in.astype(float)
        I_syn -= (I_syn / TAU_SYN) * DT

        # ── Output layer: integrate synaptic input (vectorised LIF) ──
        active_out = ref_out <= 0
        V_out[active_out] += (-(V_out[active_out] - V_REST) + R_M * I_syn[active_out]) / TAU_M * DT
        ref_out = np.maximum(ref_out - DT, 0.0)
        fired_out = (V_out >= V_THRESH) & active_out
        V_out[fired_out] = V_RESET
        ref_out[fired_out] = T_REF
        out_spikes[i] = fired_out
        V_out_rec[i] = V_out

    return t, in_spikes, out_spikes, V_out_rec, W


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS — deliberately timestep-invariant
#
#  WHY THIS MATTERS.  The obvious sparsity metric, "fraction of (neuron x
#  timestep) slots that contain no spike," is a trap.  It equals exactly
#  (mean firing rate x DT), so it reports the simulation timestep, not the
#  network.  Measured on this layer with everything else held fixed:
#
#       DT = 0.4 ms  ->  98.4% "sparse"
#       DT = 0.1 ms  ->  99.6% "sparse"
#       DT = 0.02 ms ->  99.9% "sparse"
#
#  Same network, same spikes (+-2% Euler drift), metric moves 1.5 points.
#  Halving the timestep must not make a network look sparser.
#
#  The metrics below depend only on spike counts and connectivity, both of
#  which are physical.  Verified constant to the digit across the DT sweep above.
# ─────────────────────────────────────────────────────────────────────────────

def layer_metrics(in_spikes, out_spikes, W, dt=None):
    """
    Timestep-invariant activity and synaptic-operation accounting.

    Returns a dict:
      in_rate_hz, out_rate_hz : mean firing rate per neuron  (DT-invariant)
      synops_event            : synaptic operations actually performed by an
                                event-driven fabric = sum over input spikes of
                                that neuron's fan-out                (DT-invariant)
      synops_dense            : operations a clocked dense datapath would perform
                                = (nonzero synapses) x (timesteps)   (clock-dependent)
      synop_ratio             : synops_dense / synops_event          (clock-dependent)
    """
    dt = DT if dt is None else dt
    T = out_spikes.shape[0]
    t_sim = T * dt

    fan_out = (W > 0).sum(axis=0)                       # (N_IN,) postsyn targets per input
    spikes_per_input = in_spikes.sum(axis=0)            # (N_IN,)
    synops_event = int((spikes_per_input * fan_out).sum())

    n_synapses = int((W > 0).sum())
    synops_dense = n_synapses * T

    return {
        "in_rate_hz":   in_spikes.sum()  / in_spikes.shape[1]  / t_sim,
        "out_rate_hz":  out_spikes.sum() / out_spikes.shape[1] / t_sim,
        "in_spikes":    int(in_spikes.sum()),
        "out_spikes":   int(out_spikes.sum()),
        "n_synapses":   n_synapses,
        "synops_event": synops_event,
        "synops_dense": synops_dense,
        "synop_ratio":  synops_dense / max(synops_event, 1),
    }


def plot_sparsity(t, in_spikes, out_spikes, W):
    """
    Three-panel figure showing the layer's transformation, not just its output:
      (top)    input-layer spike raster
      (middle) output-layer spike raster
      (bottom) population activity — % of output neurons active per 5 ms bin.

    The bottom panel is the headline: it is a physical rate, independent of DT.

    House style: Georgia bold titles/axis labels, 8 pt bold tick labels, dots s=50.
    """
    t_ms = t * 1000
    m = layer_metrics(in_spikes, out_spikes, W)

    # Population activity in 5 ms bins (fraction of neurons firing per bin).
    bin_ms    = 5.0
    bin_steps = int(bin_ms / (DT * 1000))
    n_bins    = len(t) // bin_steps
    trimmed   = out_spikes[: n_bins * bin_steps]
    binned    = trimmed.reshape(n_bins, bin_steps, out_spikes.shape[1])
    frac_active = binned.any(axis=1).mean(axis=1) * 100.0
    bin_centers = (np.arange(n_bins) + 0.5) * bin_ms

    fig, (ax_i, ax_r, ax_a) = plt.subplots(
        3, 1, figsize=(11, 8.5), sharex=True,
        gridspec_kw={"height_ratios": [2, 2, 1.2]}
    )

    font_title = {"family": "Georgia", "weight": "bold", "size": 12}
    font_label = {"family": "Georgia", "weight": "bold", "size": 11}

    # ── Input raster ──
    for n in range(in_spikes.shape[1]):
        st = t_ms[in_spikes[:, n]]
        ax_i.scatter(st, np.full_like(st, n), s=50, color="#7F7F7F",
                     marker="|", linewidths=1.5)
    ax_i.set_ylabel("Input neuron", fontdict=font_label)
    ax_i.set_yticks(range(in_spikes.shape[1]))
    ax_i.set_title(
        f"Input Layer — heterogeneous constant drive  "
        f"({m['in_spikes']} spikes, {m['in_rate_hz']:.1f} Hz/neuron)",
        fontdict=font_title
    )

    # ── Output raster ──
    for n in range(out_spikes.shape[1]):
        st = t_ms[out_spikes[:, n]]
        ax_r.scatter(st, np.full_like(st, n), s=50, color="#2F5496",
                     marker="|", linewidths=1.5)
    ax_r.set_ylabel("Output neuron", fontdict=font_label)
    ax_r.set_yticks(range(out_spikes.shape[1]))
    ax_r.set_title(
        f"Output Layer — driven through W ({m['n_synapses']}/{W.size} synapses)  "
        f"({m['out_spikes']} spikes, {m['out_rate_hz']:.1f} Hz/neuron)",
        fontdict=font_title
    )

    # ── Population activity ──
    ax_a.bar(bin_centers, frac_active, width=bin_ms * 0.9,
             color="#375623", alpha=0.85)
    ax_a.set_ylabel("% neurons\nactive / 5 ms", fontdict=font_label)
    ax_a.set_xlabel("Time (ms)", fontdict=font_label)
    ax_a.set_ylim(0, 100)
    ax_a.axhline(frac_active.mean(), color="orange", linestyle="--", linewidth=1.2,
                 label=f"mean = {frac_active.mean():.1f}%")
    ax_a.legend(loc="upper right", prop={"family": "Georgia", "size": 9})

    for ax in (ax_i, ax_r, ax_a):
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily("Georgia"); lbl.set_fontweight("bold"); lbl.set_fontsize(8)

    plt.suptitle(
        f"Feedforward SNN Layer — Event-Driven Activity  "
        f"({m['synops_event']} synaptic ops vs {m['synops_dense']:,} dense clocked)",
        fontfamily="Georgia", fontweight="bold", fontsize=13
    )
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "snn_feedforward_sparsity.png")
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved: {out_path}")


def report_metrics(in_spikes, out_spikes, W):
    """Print the timestep-invariant summary (the numbers that go on the slide)."""
    m = layer_metrics(in_spikes, out_spikes, W)
    print("\n=== Layer activity (timestep-invariant) ===")
    print(f"Input  : {m['in_spikes']:4d} spikes   {m['in_rate_hz']:5.1f} Hz/neuron")
    print(f"Output : {m['out_spikes']:4d} spikes   {m['out_rate_hz']:5.1f} Hz/neuron")
    print(f"Synapses present: {m['n_synapses']}/{W.size}")
    print("\n=== Synaptic-operation accounting ===")
    print(f"Event-driven  : {m['synops_event']:,} synops to process {T_SIM*1000:.0f} ms")
    print(f"  (= sum over input spikes of that neuron's fan-out; independent of DT)")
    print(f"Dense clocked : {m['synops_dense']:,} synops at DT = {DT*1000:.2f} ms")
    print(f"  (= {m['n_synapses']} synapses x {len(in_spikes)} timesteps)")
    print(f"Ratio         : {m['synop_ratio']:.0f}x fewer operations")
    print("  CAVEAT: the ratio scales with the dense datapath's clock rate, so quote")
    print("  it with the timestep stated. The event-driven count alone is the")
    print("  physical number — it does not move when DT changes.")


if __name__ == "__main__":
    I = 3e-9

    # ── Two connected neurons ──
    print("\n=== Two Connected Neurons ===")
    t, V1, spikes1, V2, spikes2 = simulate_two_neurons(I)
    plot_two_neurons(t, V1, spikes1, V2, spikes2, I)

    # ── N-neuron feedforward layer (Week 4) ──
    print("\n=== N-Neuron Feedforward Layer ===")
    t, in_spikes, out_spikes, V_out_rec, W = simulate_feedforward_layer(seed=3)
    print("Input  firing rates (Hz):", np.round(in_spikes.sum(axis=0)  / T_SIM, 1))
    print("Output firing rates (Hz):", np.round(out_spikes.sum(axis=0) / T_SIM, 1))
    report_metrics(in_spikes, out_spikes, W)
    plot_sparsity(t, in_spikes, out_spikes, W)
