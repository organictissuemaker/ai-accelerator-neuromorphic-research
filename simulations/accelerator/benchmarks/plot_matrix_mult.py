"""Plot matrix_mult benchmark results: latency-vs-size and speedup-vs-size.

Distinguishes precision, so CPU / GPU-fp32 / GPU-fp16 appear as separate series.
Speedup is measured against the CPU (fp32) baseline.

Usage (from benchmarks/ folder):
    python plot_matrix_mult.py --csv matrix_mult/results.csv --out ../../../results/figures
"""
import argparse
import csv
import os
from collections import defaultdict


def series_label(device, precision):
    if device == "cpu":
        return "CPU"
    return f"{device.upper()} ({precision})"


def load(csv_path):
    """Return {series_label: {size_int: latency_ms}} for matrix_mult rows."""
    data = defaultdict(dict)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("task") != "matrix_mult":
                continue
            prec = row.get("precision") or "fp32"
            label = series_label(row["device"], prec)
            n = int(row["size"].split("x")[0])   # "1000x1000" -> 1000
            data[label][n] = float(row["latency_ms"])
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results.csv")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load(args.csv)
    if not data:
        raise SystemExit("No matrix_mult rows found in " + args.csv)
    os.makedirs(args.out, exist_ok=True)

    # --- Figure 1: latency vs size (log-log) ---
    plt.figure(figsize=(7, 5))
    for label in sorted(data):
        sizes = sorted(data[label])
        lat = [data[label][s] for s in sizes]
        plt.plot(sizes, lat, marker="o", label=label)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Matrix size (N, for N x N)")
    plt.ylabel("Latency per matmul (ms)")
    plt.title("Matrix Multiply: Latency vs Size")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    f1 = os.path.join(args.out, "matmul_latency_vs_size.png")
    plt.tight_layout(); plt.savefig(f1, dpi=150); plt.close()

    # --- Figure 2: speedup vs size (all non-CPU series vs CPU baseline) ---
    saved = [f1]
    if "CPU" in data:
        base = data["CPU"]
        plt.figure(figsize=(7, 5))
        plotted = False
        for label in sorted(data):
            if label == "CPU":
                continue
            common = sorted(set(base) & set(data[label]))
            if not common:
                continue
            speedup = [base[s] / data[label][s] for s in common]
            plt.plot(common, speedup, marker="s", label=f"{label} vs CPU")
            for s, sp in zip(common, speedup):
                plt.annotate(f"{sp:.1f}x", (s, sp),
                             textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
            plotted = True
        if plotted:
            plt.xscale("log")
            plt.xlabel("Matrix size (N, for N x N)")
            plt.ylabel("Speedup vs CPU (CPU latency / device latency)")
            plt.title("Matrix Multiply: Speedup vs Size")
            plt.legend(); plt.grid(True, which="both", alpha=0.3)
            f2 = os.path.join(args.out, "matmul_speedup_vs_size.png")
            plt.tight_layout(); plt.savefig(f2, dpi=150); plt.close()
            saved.append(f2)

    print("saved:", ", ".join(saved))


if __name__ == "__main__":
    main()
