"""Plot image_classification benchmark results as a grouped bar chart.

X-axis = batch size; one bar group per batch, one bar per device/precision series.
Shows how GPU throughput climbs with batch while CPU does not.

Usage (from benchmarks/ folder):
    python plot_image_classification.py --csv image_classification/results.csv --out ../../../results/figures
"""
import argparse
import csv
import os
from collections import defaultdict


def series_label(device, precision):
    return "CPU" if device == "cpu" else f"{device.upper()} ({precision})"


def load(csv_path, model):
    """Return ({series: {batch: throughput}}, sorted_batches)."""
    data = defaultdict(dict)
    batches = set()
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("task") != "image_classification":
                continue
            if row.get("model") != model:
                continue
            b = int(row.get("batch", 1))
            label = series_label(row["device"], row.get("precision") or "fp32")
            data[label][b] = float(row["throughput_ips"])
            batches.add(b)
    return data, sorted(batches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results.csv")
    ap.add_argument("--out", default=".")
    ap.add_argument("--model", default="resnet18")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data, batches = load(args.csv, args.model)
    if not data:
        raise SystemExit(f"No image_classification rows for model={args.model}")
    os.makedirs(args.out, exist_ok=True)

    order = ["CPU", "CUDA (fp32)", "CUDA (fp16)"]
    series = [s for s in order if s in data] + [s for s in data if s not in order]
    colors = {"CPU": "#4C72B0", "CUDA (fp32)": "#55A868", "CUDA (fp16)": "#DD8452"}

    x = np.arange(len(batches))
    width = 0.8 / max(len(series), 1)

    plt.figure(figsize=(8, 5))
    for i, s in enumerate(series):
        vals = [data[s].get(b, 0) for b in batches]
        offset = (i - (len(series) - 1) / 2) * width
        bars = plt.bar(x + offset, vals, width, label=s, color=colors.get(s, "#888"))
        for bar, v in zip(bars, vals):
            if v > 0:
                plt.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.0f}",
                         ha="center", va="bottom", fontsize=7)

    plt.xticks(x, [f"batch {b}" for b in batches])
    plt.ylabel("Throughput (images / sec)")
    plt.title(f"Image Classification Throughput ({args.model})")
    plt.legend(); plt.grid(True, axis="y", alpha=0.3); plt.margins(y=0.12)
    f1 = os.path.join(args.out, f"imgcls_throughput_{args.model}.png")
    plt.tight_layout(); plt.savefig(f1, dpi=150); plt.close()
    print("saved:", f1)


if __name__ == "__main__":
    main()
