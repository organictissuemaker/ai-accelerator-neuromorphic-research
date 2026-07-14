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

    #2025 reference chips
    
    "GPU 2025\n(NVIDIA GB200)": {
        "power_mw": 1000000,   # ~1000W per GPU; Blackwell architecture
        "latency_ms": 0.3,     # sub-ms for datacenter inference
        "color": "#0D47A1",
        "source": "NVIDIA GB200; MLPerf v5.1 datacenter inference submission"
    },
    "Accelerator 2025\n(Google TPU v5)": {
        "power_mw": 150000,    # ~150W per TPU v5e chip
        "latency_ms": 0.8,     # highly optimized for matrix ops
        "color": "#1B5E20",
        "source": "Google TPU v5e; MLPerf v5.1 inference submission"
    },
    "Neuromorphic 2025\n(Intel Loihi 2)": {
        "power_mw": 1000,      # ~1W; 10x more powerful than Loihi 1
        "latency_ms": 8,       # event-driven; best for sparse workloads
        "color": "#4A148C",
        "source": "Intel Loihi 2; Davies et al. 2021; sparse workload benchmark"
    },
    "Desktop AI 2025\n(NVIDIA GB10 / DGX Spark)": {
        "power_mw": 140000,    # 140W chip TDP (CPU+GPU combined)
        "latency_ms": 3,       # small-DNN inference estimate; memory-bound
                               # (273 GB/s LPDDR5x vs ~TB/s HBM on GB200)
        "color": "#00695C",
        "source": "NVIDIA DGX Spark datasheet 2025; GB10 Grace Blackwell, "
                  "128GB unified memory, 1 PFLOP FP4"
    },

}


def plot_power_latency_tradeoff():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Per-chip label placement: (x_off, y_off) in points, plus alignment.
    # Fixes the overlapping cluster (TPU v3 / V100 / GB10 / TPU v5) and
    # keeps edge labels (CPU, GB200) inside the axes.
    label_style = {
        "CPU\n(Intel i7)":                            ((-12,  0),  "right", "center"),
        "GPU\n(NVIDIA V100)":                         (( 10, 10),  "left",  "bottom"),
        "AI Accelerator\n(Google TPU v3)":            ((-10, 10),  "right", "bottom"),
        "Neuromorphic\n(Intel Loihi)":                (( 12,  0),  "left",  "center"),
        "Eyeriss\n(MIT)":                             (( 12,  0),  "left",  "center"),
        "GPU 2025\n(NVIDIA GB200)":                   (( 12, -4),  "left",  "top"),
        "Accelerator 2025\n(Google TPU v5)":          ((-10, -12), "right", "top"),
        "Neuromorphic 2025\n(Intel Loihi 2)":         (( 12,  0),  "left",  "center"),
        "Desktop AI 2025\n(NVIDIA GB10 / DGX Spark)": ((  0, -14), "center", "top"),
    }

    for name, data in architectures.items():
        ax.scatter(data["latency_ms"], data["power_mw"],
                   color=data["color"], s=100, zorder=5)   # was s=200 → 50% smaller
        offset, ha, va = label_style.get(name, ((8, 5), "left", "bottom"))
        ax.annotate(name,
                    xy=(data["latency_ms"], data["power_mw"]),
                    xytext=offset, textcoords="offset points",
                    ha=ha, va=va,
                    fontsize=9, color=data["color"], fontweight="bold")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.margins(x=0.15, y=0.25)   # breathing room so labels stay inside the axes

    ax.set_xlabel("Inference Latency (ms) — lower is better",
                  fontsize=11, fontname="EB Garamond")
    ax.set_ylabel("Power Consumption (mW) — lower is better",
                  fontsize=11, fontname="EB Garamond")
    ax.set_title("Power vs. Latency Trade-off: AI Compute Architectures\n"
                 "(Log scale — closer to bottom-left = more efficient)",
                 fontsize=12, fontname="EB Garamond")

    # Efficiency arrow
    ax.annotate("More efficient", xy=(0.15, 0.15), xytext=(0.35, 0.35),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="grey"),
                color="grey", fontsize=9)

    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "power_latency_tradeoff.png"), dpi=150)
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
    plt.savefig(os.path.join(FIGURES_DIR, "power_comparison_bar.png"), dpi=150)
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
