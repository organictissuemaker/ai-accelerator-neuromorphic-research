"""
Leaky Integrate-and-Fire (LIF) Neuron Simulator
------------------------------------------------
A basic SNN building block. Simulates how a biological neuron
integrates input current and fires a spike when voltage threshold is crossed.

Extend this to build a full SNN layer.
"""

import numpy as np
import matplotlib.pyplot as plt


# ── LIF Neuron Parameters ────────────────────────────────────────────────────
TAU_M   = 20e-3   # membrane time constant (seconds)
V_REST  = -70e-3  # resting membrane potential (volts)
V_THRESH = -50e-3 # firing threshold (volts)
V_RESET  = -70e-3 # reset potential after spike (volts)
R_M     = 10e6    # membrane resistance (ohms)
DT      = 1e-4    # simulation time step (seconds)
T_SIM   = 0.5     # total simulation time (seconds)


def simulate_lif(I_input):
    """
    Simulate a single LIF neuron given a constant input current.

    Parameters:
        I_input (float): Input current in amperes

    Returns:
        t      : time array
        V      : membrane voltage array
        spikes : list of spike times
    """
    t = np.arange(0, T_SIM, DT)
    V = np.full(len(t), V_REST)
    spikes = []

    for i in range(1, len(t)):
        # LIF differential equation (Euler integration)
        dV = (-(V[i-1] - V_REST) + R_M * I_input) / TAU_M * DT
        V[i] = V[i-1] + dV

        # Fire and reset
        if V[i] >= V_THRESH:
            spikes.append(t[i])
            V[i] = V_RESET

    return t, V, spikes


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
    plt.savefig("../../results/figures/lif_neuron_trace.png", dpi=150)
    plt.show()
    print(f"Spikes fired: {len(spikes)}")
    print(f"Firing rate: {len(spikes)/T_SIM:.1f} Hz")


if __name__ == "__main__":
    I = 2e-9  # 2 nanoamps — try changing this and see what happens
    t, V, spikes = simulate_lif(I)
    plot_lif(t, V, spikes, I)
