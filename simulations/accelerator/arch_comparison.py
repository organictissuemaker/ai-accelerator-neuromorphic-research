"""
Architecture Comparison: CPU vs GPU vs Neuromorphic
----------------------------------------------------
Compares estimated power and latency for a simple inference workload
across three compute paradigms.

NOTE: Values here are illustrative estimates based on literature.
Replace with your own measured/sourced numbers as you progress.
Sources: Sze et al. 2017, Intel Loihi papers, public benchmarks.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Architecture Parameters (from literature) ─────────────────────────────────
# Format: (architecture, power_mW, latency_ms, notes)
architectures = {
    "CPU\n(Intel i7)": {
        "power_mw": 65000,
        "latency_ms": 120,
        "color": "#C55A11",
        "source": "General CPU benchmark for DNN inference"
    },
    "GPU\n(NVIDIA V100)": {
        "power_mw": 300000,
        "latency_ms": 5,
        "color": "#2F5496",
        "source": "NVIDIA V100 DNN inference"
    },
    "AI Accelerator\n(Google TPU v3)": {
        "power_mw": 200000,
        "latency_ms": 2,
        "color": "#375623",
        "source": "Google TPU v3 published specs"
    },
    "Neuromorphic\n(Intel Loihi)": {
        "power_mw": 30,
        "latency_ms": 10,
        "color": "#7030A0",
        "source": "Loihi: Davies et al. 2018, keyword spotting benchmark"
    },
    "Eyeriss\n(MIT)": {
        "power_mw": 278,
        "latency_ms": 28,
        "color": "#1F3864",
        "source": "Eyeriss: Chen et al. 2016, AlexNet inference"
    },
}


def plot_power_latency_tradeoff():
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, data in architectures.items():
        ax.scatter(data["latency_ms"], data["power_mw"],
                   color=data["color"], s=200, zorder=5)
        ax.annotate(name,
                    xy=(data["latency_ms"], data["power_mw"]),
                    xytext=(8, 5), textcoords="offset points",
                    fontsize=9, color=data["color"], fontweight="bold")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Inference Latency (ms) — lower is better", fontsize=11)
    ax.set_ylabel("Power Consumption (mW) — lower is better", fontsize=11)
    ax.set_title("Power vs. Latency Trade-off: AI Compute Architectures\n(Log scale — closer to bottom-left = more efficient)", fontsize=12)

    # Efficiency arrow
    ax.annotate("More efficient", xy=(0.15, 0.15), xytext=(0.35, 0.35),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="grey"),
                color="grey", fontsize=9)

    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("../../results/figures/power_latency_tradeoff.png", dpi=150)
    plt.show()


def plot_power_bar():
    names = list(architectures.keys())
    powers = [architectures[n]["power_mw"] for n in names]
    colors = [architectures[n]["color"] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, powers, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Power (mW, log scale)")
    ax.set_title("Power Consumption by Architecture")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, powers):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f"{val:,} mW", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig("../../results/figures/power_comparison_bar.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    print("Architecture Comparison Summary")
    print("-" * 50)
    for name, data in architectures.items():
        clean_name = name.replace("\n", " ")
        print(f"{clean_name:<30} Power: {data['power_mw']:>8,} mW | Latency: {data['latency_ms']:>5} ms")

    print("\nGenerating plots...")
    plot_power_latency_tradeoff()
    plot_power_bar()
