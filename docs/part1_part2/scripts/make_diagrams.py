"""
Generate architecture + inference diagrams for Part 1 & Part 2 (house style).
Run:  python scripts/make_diagrams.py       (from the docs/part1_part2 folder)
Output: ../figures/*.png relative to this script.
Labels use Georgia bold; these are block diagrams, so the scatter/tick
conventions from the simulation plots don't apply here.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)

INK="#16233A"; NAVY="#1E3A5F"; TEAL="#2C7A7B"; AMBER="#E8A33D"; LIGHT="#EEF2F7"; MUTE="#6B7B8C"
TITLE_FONT = {"family": "Georgia", "weight": "bold"}

def box(ax, x, y, w, h, text, fc=LIGHT, ec=NAVY, tc=INK, fs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.4, edgecolor=ec, facecolor=fc))
    if text:
        ax.text(x+w/2, y+h/2, text, ha="center", va="center",
            fontfamily="Georgia", fontweight="bold", fontsize=fs, color=tc)

def label(ax, x, y, text, fs=11, tc=INK):
    ax.text(x, y, text, ha="center", va="center",
        fontfamily="Georgia", fontweight="bold", fontsize=fs, color=tc)

def arrow(ax, x1, y1, x2, y2, color=MUTE):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=16, linewidth=1.6, color=color))

def new_ax(title, w=8, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title(title, fontsize=15, color=INK, **TITLE_FONT, pad=12)
    return fig, ax

def grid(ax, x0, y0, cols, rows, cw, ch, gx, gy):
    for gi in range(cols):
        for gj in range(rows):
            ax.add_patch(FancyBboxPatch((x0+gi*gx, y0+gj*gy), cw, ch,
                boxstyle="square,pad=0.0", linewidth=0.6, edgecolor=NAVY, facecolor="#C6D6EA"))

def save(fig, name):
    p = os.path.join(FIGDIR, name)
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.normpath(p))

# ---------- CPU ----------
fig, ax = new_ax("CPU - few latency-optimized cores + deep cache")
for i in range(2):
    for j in range(2):
        box(ax, 1.2+i*3.2, 6.4-j*1.9, 2.6, 1.6, "Core {}\n(OoO, superscalar)".format(i*2+j+1), fc="#DCE6F2", ec=NAVY, fs=10)
box(ax, 1.2, 3.0, 7.6, 0.9, "Shared L3 cache", fc=LIGHT, ec=TEAL)
box(ax, 1.2, 1.6, 7.6, 0.9, "Main memory (DRAM) via memory controller", fc="#F4E9D6", ec=AMBER, fs=10)
arrow(ax, 5.0, 3.0, 5.0, 2.5)
save(fig, "part1_cpu.png")

# ---------- GPU ----------
fig, ax = new_ax("GPU - many SIMT cores + high-bandwidth memory")
for i in range(4):
    for j in range(3):
        box(ax, 1.0+i*1.6, 6.2-j*1.0, 1.4, 0.8, "SM", fc="#DCE6F2", ec=NAVY, fs=9)
box(ax, 1.0, 3.0, 6.4, 0.8, "Shared memory / L2", fc=LIGHT, ec=TEAL, fs=10)
box(ax, 1.0, 1.6, 6.4, 0.9, "HBM (high-bandwidth memory)", fc="#F4E9D6", ec=AMBER, fs=10)
arrow(ax, 4.2, 3.0, 4.2, 2.5)
ax.text(8.2, 4.9, "SM = streaming\nmultiprocessor\n(many SIMT lanes)", fontsize=9, color=MUTE, va="center", ha="center", fontfamily="Georgia")
save(fig, "part1_gpu.png")

# ---------- TPU ----------
fig, ax = new_ax("TPU - systolic MXU + unified buffer")
box(ax, 1.0, 5.0, 4.2, 3.2, "", fc="#DCE6F2", ec=NAVY)
label(ax, 3.1, 7.8, "Systolic MAC array (MXU)", fs=11)
grid(ax, 1.45, 5.35, 4, 3, 0.62, 0.42, 0.9, 0.6)
box(ax, 6.0, 6.4, 3.0, 1.4, "Unified buffer\n(on-chip SRAM)", fc=LIGHT, ec=TEAL, fs=10)
box(ax, 6.0, 4.6, 3.0, 1.2, "Accumulators", fc=LIGHT, ec=TEAL, fs=10)
box(ax, 1.0, 2.8, 8.0, 1.0, "Off-chip DRAM / HBM", fc="#F4E9D6", ec=AMBER, fs=10)
arrow(ax, 6.0, 7.1, 5.2, 7.0); arrow(ax, 5.2, 5.6, 6.0, 5.2); arrow(ax, 3.1, 5.0, 3.1, 3.8)
save(fig, "part1_tpu.png")

# ---------- NPU ----------
fig, ax = new_ax("NPU - dataflow MAC array + on-chip SRAM")
box(ax, 1.0, 5.0, 4.2, 3.2, "", fc="#DCE6F2", ec=NAVY)
label(ax, 3.1, 7.8, "PE / MAC dataflow array", fs=11)
grid(ax, 1.5, 5.35, 3, 3, 0.9, 0.44, 1.15, 0.62)
box(ax, 6.0, 6.6, 3.0, 1.2, "On-chip SRAM", fc=LIGHT, ec=TEAL, fs=10)
box(ax, 6.0, 5.0, 3.0, 1.2, "Activation / pooling", fc=LIGHT, ec=TEAL, fs=10)
box(ax, 1.0, 2.8, 8.0, 1.0, "Off-chip DRAM", fc="#F4E9D6", ec=AMBER, fs=10)
arrow(ax, 6.0, 7.2, 5.2, 7.0); arrow(ax, 5.2, 5.6, 6.0, 5.6); arrow(ax, 3.1, 5.0, 3.1, 3.8)
save(fig, "part1_npu.png")

# ---------- Part 2: pipeline ----------
fig, ax = new_ax("AI inference pipeline - data flow", w=11, h=3.4)
ax.set_xlim(0, 22); ax.set_ylim(0, 6)
stages = ["Input", "Preprocess", "Weight /\nmodel load", "Accelerator\ncompute (MACs)", "Post-\nprocess", "Output"]
cols = [LIGHT, LIGHT, "#F4E9D6", "#DCE6F2", LIGHT, LIGHT]
x = 0.5
for i, (s, c) in enumerate(zip(stages, cols)):
    box(ax, x, 2.2, 3.0, 1.8, s, fc=c, ec=NAVY if i==3 else TEAL, fs=10)
    if i < len(stages)-1: arrow(ax, x+3.0, 3.1, x+3.5, 3.1)
    x += 3.5
save(fig, "part2_pipeline.png")

# ---------- Part 2: engine ----------
fig, ax = new_ax("Generic AI inference engine - architecture")
box(ax, 0.6, 8.2, 3.4, 1.2, "Host interface + DMA", fc=LIGHT, ec=TEAL, fs=10)
box(ax, 0.6, 5.6, 3.4, 1.2, "Off-chip DRAM", fc="#F4E9D6", ec=AMBER, fs=10)
box(ax, 4.6, 5.4, 2.6, 3.4, "On-chip SRAM /\nbuffers", fc=LIGHT, ec=TEAL, fs=10)
box(ax, 7.6, 5.4, 2.0, 3.4, "PE / MAC\narray", fc="#DCE6F2", ec=NAVY, fs=11)
box(ax, 7.6, 3.4, 2.0, 1.4, "Accumulators", fc=LIGHT, ec=TEAL, fs=9)
box(ax, 4.6, 3.4, 2.6, 1.4, "Activation /\npooling", fc=LIGHT, ec=TEAL, fs=9)
box(ax, 0.6, 3.4, 3.4, 1.4, "Control unit", fc="#E7E0EF", ec=MUTE, fs=10)
arrow(ax, 2.3, 8.2, 2.3, 6.8); arrow(ax, 4.0, 6.2, 4.6, 6.6); arrow(ax, 7.2, 7.1, 7.6, 7.1)
arrow(ax, 8.6, 5.4, 8.6, 4.8); arrow(ax, 7.6, 4.1, 7.2, 4.1)
save(fig, "part2_inference_engine.png")
print("done")
