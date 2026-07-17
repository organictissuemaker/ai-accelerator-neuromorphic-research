"""
Leaky Integrate-and-Fire (LIF) Neuron Simulator
------------------------------------------------
A basic SNN building block. Simulates how a biological neuron
integrates input current and fires a spike when voltage threshold is crossed.

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


# def simulate_lif(I_input):
#     """
#     Simulate a single LIF neuron given a constant input current.

#     Parameters:
#         I_input (float): Input current in amperes

#     Returns:
#         t      : time array
#         V      : membrane voltage array
#         spikes : list of spike times
#     """
#     t = np.arange(0, T_SIM, DT)
#     V = np.full(len(t), V_REST)
#     spikes = []

#     for i in range(1, len(t)):
#         # LIF differential equation (Euler integration)
#         dV = (-(V[i-1] - V_REST) + R_M * I_input) / TAU_M * DT
#         V[i] = V[i-1] + dV

#         # Fire and reset
#         if V[i] >= V_THRESH:
#             spikes.append(t[i])
#             V[i] = V_RESET

#     return t, V, spikes


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


def plot_lif(t, V, spikes, I_input):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t * 1000, np.array(V) * 1000, color="#2F5496", linewidth=1.5, label="Membrane Voltage")
    ax.axhline(V_THRESH * 1000, color="red", linestyle="--", linewidth=1, label=f"Threshold ({V_THRESH*1000:.0f} mV)")
    ax.axhline(V_REST * 1000, color="grey", linestyle=":", linewidth=1, label=f"Rest ({V_REST*1000:.0f} mV)")

    for s in spikes:
        ax.axvline(s * 1000, color="orange", alpha=0.6, linewidth=0.8)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Membrane Voltage (mV)")
    ax.set_title(f"LIF Neuron — Input Current: {I_input*1e9:.0f} nA | Spike Count: {len(spikes)}")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "lif_neuron_trace.png"), dpi=150)
    plt.show()
    print(f"Spikes fired: {len(spikes)}")
    print(f"Firing rate: {len(spikes)/T_SIM:.1f} Hz")

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
#  Purpose:  observe EVENT-DRIVEN SPARSITY — at any instant only a tiny
#  fraction of neurons are active. This spatiotemporal sparsity is exactly
#  why event-driven hardware (Loihi) is efficient: no spike → no compute.
# ─────────────────────────────────────────────────────────────────────────────

T_REF = 3e-3  # absolute refractory period (s) — neuron can't re-fire during this window

N_IN  = 10   # input-layer neurons
N_OUT = 10   # output-layer neurons  (the "10-neuron layer" success target)


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
    # Heterogeneous drive: input neurons span 1.25x–3x threshold → a spread of rates.
    I_in = np.linspace(1.05 * I_th, 1.6 * I_th, N_IN)

    # Sparse, random synaptic weights (post × pre). ~50% connectivity.
    mask = rng.random((N_OUT, N_IN)) < 0.6
    W = mask * rng.uniform(0.20 * W_SYN, 0.60 * W_SYN, size=(N_OUT, N_IN))

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


def plot_sparsity(t, in_spikes, out_spikes):
    """
    Two-panel figure:
      (top)    output-layer spike raster — one row per neuron, dot at each spike.
      (bottom) population activity — % of output neurons spiking per 5 ms bin,
               showing activity stays sparse (mostly near zero).

    House style: Georgia bold titles/axis labels, 8 pt bold ticks.
    """
    T = len(t)
    t_ms = t * 1000

    # Headline sparsity number: total spikes vs. total (neuron × timestep) slots.
    total_slots  = out_spikes.size
    total_spikes = int(out_spikes.sum())
    activity     = total_spikes / total_slots          # fraction of slots that fired
    sparsity     = 1.0 - activity

    # Population activity in 5 ms bins (fraction of neurons firing per bin).
    bin_ms   = 5.0
    bin_steps = int(bin_ms / (DT * 1000))
    n_bins   = T // bin_steps
    trimmed  = out_spikes[: n_bins * bin_steps]
    binned   = trimmed.reshape(n_bins, bin_steps, out_spikes.shape[1])
    # per bin: fraction of neurons that fired at least once in the bin
    frac_active = binned.any(axis=1).mean(axis=1) * 100.0
    bin_centers = (np.arange(n_bins) + 0.5) * bin_ms

    fig, (ax_r, ax_a) = plt.subplots(
        2, 1, figsize=(11, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    font_title = {"family": "Georgia", "weight": "bold", "size": 13}
    font_label = {"family": "Georgia", "weight": "bold", "size": 11}

    # ── Raster ──
    for n in range(out_spikes.shape[1]):
        spike_times = t_ms[out_spikes[:, n]]
        ax_r.scatter(spike_times, np.full_like(spike_times, n),
                     s=50, color="#2F5496", marker="|", linewidths=1.5)
    ax_r.set_ylabel("Output neuron", fontdict=font_label)
    ax_r.set_yticks(range(out_spikes.shape[1]))
    ax_r.set_title(
        f"Feedforward SNN Layer — Output Spike Raster  "
        f"(N={out_spikes.shape[1]}, {total_spikes} spikes, "
        f"{sparsity*100:.1f}% sparse)",
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

    for ax in (ax_r, ax_a):
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily("Georgia"); lbl.set_fontweight("bold"); lbl.set_fontsize(8)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "snn_feedforward_sparsity.png")
    plt.savefig(out_path, dpi=150)
    plt.show()

    print(f"\n=== Feedforward layer sparsity ===")
    print(f"Output spikes: {total_spikes} across {out_spikes.shape[1]} neurons × {T} steps")
    print(f"Event sparsity: {sparsity*100:.2f}% of (neuron × timestep) slots are silent")
    print(f"Mean population activity: {frac_active.mean():.1f}% of neurons per 5 ms bin")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    # ── Single neuron (original) ──
    I = 3e-9
    # t, V, spikes = simulate_lif(I)
    # plot_lif(t, V, spikes, I)

    # ── Two connected neurons ──
    print("\n=== Two Connected Neurons ===")
    t, V1, spikes1, V2, spikes2 = simulate_two_neurons(I)
    plot_two_neurons(t, V1, spikes1, V2, spikes2, I)

    # ── N-neuron feedforward layer (Week 4) ──
    print("\n=== N-Neuron Feedforward Layer ===")
    t, in_spikes, out_spikes, V_out_rec, W = simulate_feedforward_layer(seed=3)
    # sanity: report per-neuron output rates
    rates = out_spikes.sum(axis=0) / T_SIM
    print("Output firing rates (Hz):", np.round(rates, 1))
    plot_sparsity(t, in_spikes, out_spikes)
