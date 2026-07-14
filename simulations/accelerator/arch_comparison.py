"""
Architecture Comparison: CPU vs GPU vs Neuromorphic
----------------------------------------------------
Compares estimated power and latency for a simple inference workload
across three compute paradigms.

NOTE: Values here are illustrative estimates based on literature.
Sources: Sze et al. 2017, Intel Loihi papers, public benchmarks.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Resolve output directory relative to this script's location ──────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../results/figures"))
os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Architecture Parameters (from literature) ─────────────────────────────────
# Format: (architecture, power_mW, latency_ms, notes)
architectures = {
    "CPU\n(Intel i7)": {
        "power_mw": 65000,
        "latency_ms": 120,
        "color": "#C55A11",
        "source": "General CPU benchmark for DNN inference"
    },
    "GPU\n(V100)": {
        "power_mw": 300000,
        "latency_ms": 5,
        "color": "#2F5496",
        "source": "NVIDIA V100 DNN inference"
    },
    "TPU v3\n(Google)": {
        "power_mw": 200000,
        "latency_ms": 2,
        "color": "#375623",
        "source": "Google TPU v3 published specs"
    },
    "Loihi\n(Intel)": {
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

    # 2025 reference chips

    "GB200\n(NVIDIA 2025)": {
        "power_mw": 1000000,   # ~1000W per GPU; Blackwell architecture
        "latency_ms": 0.3,     # sub-ms for datacenter inference
        "color": "#0D47A1",
        "source": "NVIDIA GB200; MLPerf v5.1 datacenter inference submission"
    },
    "TPU v5\n(Google 2025)": {
        "power_mw": 150000,    # ~150W per TPU v5e chip
        "latency_ms": 0.8,     # highly optimized for matrix ops
        "color": "#1B5E20",
        "source": "Google TPU v5e; MLPerf v5.1 inference submission"
    },
    "Loihi 2\n(Intel 2025)": {
        "power_mw": 1000,      # ~1W; 10x more powerful than Loihi 1
        "latency_ms": 8,       # event-driven; best for sparse workloads
        "color": "#4A148C",
        "source": "Intel Loihi 2; Davies et al. 2021; sparse workload benchmark"
    },
    "GB10 Spark\n(NVIDIA 2025)": {
        "power_mw": 140000,    # 140W chip TDP (CPU+GPU combined)
        "latency_ms": 3,       # small-DNN inference estimate; memory-bound
                               # (273 GB/s LPDDR5x vs ~TB/s HBM on GB200)
        "color": "#00695C",
        "source": "NVIDIA DGX Spark datasheet 2025; GB10 Grace Blackwell, "
                  "128GB unified memory, 1 PFLOP FP4"
    },
}

def plot_power_latency_tradeoff():
    from adjustText import adjust_text

    fig, ax = plt.subplots(figsize=(10, 6))

    texts = []
    for name, data in architectures.items():
        ax.scatter(data["latency_ms"], data["power_mw"],
                   color=data["color"], s=50, zorder=5)   # 50% smaller again
        # Bold chip name, regular parenthesis (mathtext bold on line 1)
        head, _, tail = name.partition("\n")
        label = rf"$\bf{{{head.replace(' ', r'\ ')}}}$" + ("\n" + tail if tail else "")
        texts.append(ax.text(data["latency_ms"], data["power_mw"], label,
                             fontsize=9, color=data["color"],
                             ha="center", va="bottom"))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.margins(x=0.15, y=0.25)

    # Tick labels: 2 pts smaller (default 10 → 8) and bold
    ax.tick_params(axis="both", labelsize=8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")

    # Auto-resolve overlaps: nudges labels apart, draws a thin grey
    # line from label to dot when it has to move one far away
    adjust_text(texts, ax=ax,
                expand_points=(1.8, 2.2), expand_text=(1.2, 1.4),
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.5, alpha=0.6))

    ax.set_xlabel("Inference Latency (ms) — lower is better",
                  fontsize=11, fontname="Georgia", fontweight="bold")
    ax.set_ylabel("Power Consumption (mW) — lower is better",
                  fontsize=11, fontname="Georgia", fontweight="bold")
    ax.set_title("Power vs. Latency Trade-off: AI Compute Architectures\n"
                 "(Log scale — closer to bottom-left = more efficient)",
                 fontsize=12, fontname="Georgia", fontweight="bold")

    ax.annotate("More efficient", xy=(0.15, 0.15), xytext=(0.35, 0.35),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="grey"),
                color="grey", fontsize=9)

    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "power_latency_tradeoff.png"), dpi=150)


def plot_power_bar():
    names = list(architectures.keys())
    powers = [architectures[n]["power_mw"] for n in names]
    colors = [architectures[n]["color"] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, powers, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Power (mW, log scale)",
                  fontsize=11, fontname="Georgia", fontweight="bold")
    ax.set_title("Power Consumption by Architecture",
                 fontsize=12, fontname="Georgia", fontweight="bold")

    # Match the scatter plot's tick formatting
    # Chip names: bold name, regular parenthesis
    styled = []
    for name in names:
        head, _, tail = name.partition("\n")
        styled.append(rf"$\bf{{{head.replace(' ', r'\ ')}}}$"
                      + ("\n" + tail if tail else ""))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(styled, fontsize=8)

    # y-axis (power scale) stays bold + small
    ax.tick_params(axis="y", labelsize=8)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, powers):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f"{val:,} mW", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "power_comparison_bar.png"), dpi=150)


if __name__ == "__main__":
    print("Architecture Comparison Summary")
    print("-" * 50)
    for name, data in architectures.items():
        clean_name = name.replace("\n", " ")
        print(f"{clean_name:<30} Power: {data['power_mw']:>8,} mW | Latency: {data['latency_ms']:>5} ms")

    print("\nGenerating plots...")
    plot_power_latency_tradeoff()
    plot_power_bar()
    plt.show()