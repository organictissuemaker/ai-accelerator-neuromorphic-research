#!/usr/bin/env python3
"""
Route A - Steps 4 & 5: Comparison table + headline figures.

Reads the combined metrics CSV produced by cnn_baseline.py (workload=MACs) and
snn_baseline.py (workload=spikes), then:

  Step 4  Builds a CPU/GPU/SNN comparison table (accuracy, latency, throughput,
          workload) and writes it as Markdown + CSV to results/.
  Step 5  Generates two headline figures into results/figures/:
            1. workload-per-inference bar chart  (MACs vs spikes, log scale)
            2. accuracy-vs-workload scatter       (the same-task tradeoff plot)
          Plus a bonus per-device latency bar chart.

Usage
-----
    python make_figures.py                          # uses default paths
    python make_figures.py --csv results/metrics_combined.csv
    python make_figures.py --outdir results/figures

Run from the repo root. Dependencies: pandas, matplotlib.
Repo placement: results/make_figures.py  (or a scripts/ folder)
"""

import argparse
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load(csv_path):
    df = pd.read_csv(csv_path)
    # A friendly approach label: "CNN" for MAC rows, "SNN" for spike rows.
    df["approach"] = df["workload_metric"].map({"MACs": "CNN", "spikes": "SNN"})
    df["approach"] = df["approach"].fillna(df["model"])
    return df


# --------------------------------------------------------------------------- #
# Step 4: comparison table
# --------------------------------------------------------------------------- #
def build_table(df, outdir):
    cols = ["approach", "model", "device_name", "device_type", "accuracy",
            "latency_ms_per_inf", "throughput_inf_per_s",
            "workload_metric", "workload_per_inf"]
    table = df[cols].copy()
    table = table.sort_values(["approach", "device_type", "device_name"])

    # Human-readable formatting for display/markdown.
    disp = table.copy()
    disp["accuracy"] = (disp["accuracy"] * 100).map(lambda v: f"{v:.2f}%")
    disp["latency_ms_per_inf"] = disp["latency_ms_per_inf"].map(lambda v: f"{v:.4f}")
    disp["throughput_inf_per_s"] = disp["throughput_inf_per_s"].map(
        lambda v: f"{v:,.0f}")
    disp["workload_per_inf"] = disp["workload_per_inf"].map(lambda v: f"{v:,.1f}")
    disp = disp.rename(columns={
        "approach": "Approach", "model": "Model", "device_name": "Device",
        "device_type": "Backend", "accuracy": "Accuracy",
        "latency_ms_per_inf": "Latency (ms/inf)",
        "throughput_inf_per_s": "Throughput (inf/s)",
        "workload_metric": "Workload metric", "workload_per_inf": "Workload/inf",
    })

    csv_out = os.path.join(outdir, "comparison_table.csv")
    md_out = os.path.join(outdir, "comparison_table.md")
    table.to_csv(csv_out, index=False)
    with open(md_out, "w") as f:
        f.write("# Route A - CPU / GPU / SNN comparison (MNIST, LeNet)\n\n")
        f.write(disp.to_markdown(index=False))
        f.write("\n")
    print(f"[table] wrote {csv_out}")
    print(f"[table] wrote {md_out}")
    print("\n" + disp.to_string(index=False) + "\n")
    return table


# --------------------------------------------------------------------------- #
# Step 5: figures
# --------------------------------------------------------------------------- #
CNN_COLOR = "#4C72B0"
SNN_COLOR = "#DD8452"


def _approach_summary(df):
    """One representative row per approach (workload/accuracy are ~constant)."""
    g = df.groupby("approach").agg(
        accuracy=("accuracy", "mean"),
        workload=("workload_per_inf", "mean"),
        metric=("workload_metric", "first"),
    ).reset_index()
    return g


def fig_workload_bar(df, outdir):
    g = _approach_summary(df).sort_values("workload")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = [CNN_COLOR if a == "CNN" else SNN_COLOR for a in g["approach"]]
    bars = ax.bar(g["approach"], g["workload"], color=colors, width=0.55)
    ax.set_yscale("log")
    ax.set_ylabel("Workload per inference (log scale)")
    ax.set_title("Workload per inference: CNN (MACs) vs SNN (spikes)")
    for bar, val, metric in zip(bars, g["workload"], g["metric"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.15,
                f"{val:,.0f}\n{metric}", ha="center", va="bottom", fontsize=10)

    # Annotate the reduction factor.
    cnn = g.loc[g["approach"] == "CNN", "workload"].values[0]
    snn = g.loc[g["approach"] == "SNN", "workload"].values[0]
    factor = cnn / snn
    ax.text(0.5, 0.92, f"~{factor:.0f}x fewer events (proxy)",
            transform=ax.transAxes, ha="center", fontsize=10,
            bbox=dict(boxstyle="round", fc="#f0f0f0", ec="gray"))
    ax.set_ylim(top=cnn * 4)

    fig.text(0.5, 0.005,
             "Note: MACs and spikes are different units; this is a workload "
             "proxy. Route B converts both to joules.",
             ha="center", fontsize=7.5, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = os.path.join(outdir, "workload_per_inference.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[figure] wrote {out}")


def fig_accuracy_scatter(df, outdir):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for approach, color in [("CNN", CNN_COLOR), ("SNN", SNN_COLOR)]:
        sub = df[df["approach"] == approach]
        ax.scatter(sub["workload_per_inf"], sub["accuracy"] * 100,
                   s=90, color=color, edgecolor="black", linewidth=0.6,
                   label=approach, zorder=3)
    # Label each cluster once.
    g = _approach_summary(df)
    for _, r in g.iterrows():
        ax.annotate(f"{r['approach']}\n({r['metric']})",
                    (r["workload"], r["accuracy"] * 100),
                    textcoords="offset points", xytext=(8, 8), fontsize=9)

    ax.set_xscale("log")
    ax.set_xlabel("Workload per inference (log scale) - MACs (CNN) / spikes (SNN)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Same task: accuracy vs. per-inference workload")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(title="Approach", loc="lower right")
    # Zoom y to make the small accuracy gap legible.
    lo = df["accuracy"].min() * 100 - 0.5
    hi = df["accuracy"].max() * 100 + 0.5
    ax.set_ylim(lo, hi)
    fig.tight_layout()
    out = os.path.join(outdir, "accuracy_vs_workload.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[figure] wrote {out}")


def fig_latency_bar(df, outdir):
    """Bonus: per-device latency, grouped by approach."""
    piv = df.pivot_table(index="device_name", columns="approach",
                         values="latency_ms_per_inf", aggfunc="mean")
    order = [c for c in ["CNN", "SNN"] if c in piv.columns]
    piv = piv[order]
    ax = piv.plot(kind="bar", figsize=(7.5, 4.5),
                  color=[CNN_COLOR, SNN_COLOR][:len(order)])
    ax.set_ylabel("Latency (ms / inference)")
    ax.set_xlabel("")
    ax.set_title("Inference latency per device (lower is better)")
    ax.legend(title="Approach")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out = os.path.join(outdir, "latency_per_device.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[figure] wrote {out}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Step 4/5 table + figures")
    p.add_argument("--csv", default="results/metrics_combined.csv")
    p.add_argument("--outdir", default="results/figures")
    p.add_argument("--table-dir", default="results")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.table_dir, exist_ok=True)

    df = load(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv} "
          f"({df['approach'].nunique()} approaches, "
          f"{df['device_name'].nunique()} devices).\n")

    build_table(df, args.table_dir)
    fig_workload_bar(df, args.outdir)
    fig_accuracy_scatter(df, args.outdir)
    fig_latency_bar(df, args.outdir)
    print("\nDone. Headline figures + table are ready for the Step 6 write-up.")


if __name__ == "__main__":
    main()
