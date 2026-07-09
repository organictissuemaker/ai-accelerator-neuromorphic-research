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
W_SYN = 5e-9


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

    Neuron 1: receives constant input current (same as before).
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

if __name__ == "__main__":
    # Original: single neuron
    I = 3e-9  
    t, V, spikes = simulate_lif(I)
    plot_lif(t, V, spikes, I)

    # Extended: two connected neurons
    print("\n=== Two Connected Neurons ===")
    t, V1, spikes1, V2, spikes2 = simulate_two_neurons(I)
    plot_two_neurons(t, V1, spikes1, V2, spikes2, I)

