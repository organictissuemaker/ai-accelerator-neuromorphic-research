"""
Roofline Model: NVIDIA GB10 (DGX Spark)
----------------------------------------
Shows why LLM inference on GB10 is memory-bound: the ridge point sits at
~3,700 FLOPs/byte, but LLM token generation has an arithmetic intensity
of only ~4 FLOPs/byte (2 ops per weight, 0.5 bytes per FP4 weight).

Roofline model: attainable perf = min(peak_compute, bandwidth * intensity)
Reference: Williams et al. 2009, "Roofline: An Insightful Visual
Performance Model for Multicore Architectures" (CACM).

Chip specs: NVIDIA DGX Spark datasheet (2025).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../results/figures"))
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── GB10 specs (NVIDIA DGX Spark datasheet, 2025) ────────────────────────────
PEAK_FLOPS = 1e15      # 1 PFLOP @ FP4 (with sparsity; vendor claim)
BANDWIDTH  = 273e9     # 273 GB/s LPDDR5x unified memory

RIDGE = PEAK_FLOPS / BANDWIDTH   # ~3,663 FLOPs/byte


def styled_label(name):
    """House style: bold first line, regular parenthesis line."""
    head, _, tail = name.partition("\n")
    head_tex = head.replace(" ", "\\ ")   # mathtext needs escaped spaces
    return rf"$\bf{{{head_tex}}}$" + ("\n" + tail if tail else "")


# ── Workloads: (name, arithmetic intensity in FLOPs/byte, annotation offset) ──
# LLM decode: 2 FLOPs per weight per token; FP4 weight = 0.5 byte -> 4 FLOPs/B
# CNN conv layers reuse each weight many times -> orders of magnitude higher
workloads = {
    "LLM token generation\n(FP4, batch=1)": (4,      (10, -15)),
    "LLM decode\n(batch=8)":                (32,     (10, -15)),
    "CNN conv layer\n(high weight reuse)":  (2000,   (-160, 20)),
}

intensity = np.logspace(-1, 5, 400)   # FLOPs/byte
attainable = np.minimum(PEAK_FLOPS, BANDWIDTH * intensity)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(intensity, attainable, color="#1F3864", lw=2.5, zorder=3)

# Ridge point
ax.axvline(RIDGE, color="grey", ls=":", alpha=0.7)
ax.annotate(f"Ridge point\n≈ {RIDGE:,.0f} FLOPs/byte",
            xy=(RIDGE, PEAK_FLOPS), xytext=(RIDGE * 1.5, PEAK_FLOPS * 0.15),
            fontsize=9, color="grey",
            arrowprops=dict(arrowstyle="->", color="grey"))

# Shade regions
ax.axvspan(intensity[0], RIDGE, color="#C55A11", alpha=0.08)
ax.axvspan(RIDGE, intensity[-1], color="#375623", alpha=0.08)
ax.text(1, PEAK_FLOPS * 2, "MEMORY-BOUND\n(hose limits you)",
        fontsize=10, color="#C55A11", fontweight="bold")
ax.text(RIDGE * 3, PEAK_FLOPS * 2, "COMPUTE-BOUND",
        fontsize=10, color="#375623", fontweight="bold")

# Workload points on the roofline (bold name, regular parenthesis)
for name, (ai, offset) in workloads.items():
    perf = min(PEAK_FLOPS, BANDWIDTH * ai)
    ax.scatter(ai, perf, s=50, color="#7030A0", zorder=5)
    ax.annotate(styled_label(name), xy=(ai, perf), xytext=offset,
                textcoords="offset points", fontsize=8.5, color="#7030A0")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(1e10, PEAK_FLOPS * 8)
ax.set_xlabel("Arithmetic Intensity (FLOPs / byte)",
              fontsize=11, fontname="Georgia", fontweight="bold")
ax.set_ylabel("Attainable Performance (FLOPs/s)",
              fontsize=11, fontname="Georgia", fontweight="bold")
ax.set_title("Roofline Model — NVIDIA GB10 (DGX Spark)\n"
             "1 PFLOP FP4 ceiling vs 273 GB/s memory slope",
             fontsize=12, fontname="Georgia", fontweight="bold")

# Tick labels: 8pt bold (house style)
ax.tick_params(axis="both", labelsize=8)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontweight("bold")

ax.grid(True, which="both", ls="--", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "roofline_gb10.png"), dpi=150)

if __name__ == "__main__":
    print(f"Ridge point: {RIDGE:,.0f} FLOPs/byte")
    for name, (ai, _) in workloads.items():
        perf = min(PEAK_FLOPS, BANDWIDTH * ai)
        bound = "memory" if ai < RIDGE else "compute"
        print(f"{name.replace(chr(10), ' '):<40} AI={ai:>6} FLOPs/B "
              f"-> {perf/1e12:8.1f} TFLOP/s ({bound}-bound)")
    plt.show()
