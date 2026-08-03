"""Plot image_classification benchmark results.

Bar chart of throughput (images/sec) by device+precision, plus a speedup annotation
vs the CPU baseline. Filters to one model (default resnet18) and one batch size.

Usage (from benchmarks/ folder):
    python plot_image_classification.py --csv image_classification/results.csv --out ../../../results/figures
"""
import argparse
import csv
import os


def series_label(device, precision):
    return "CPU" if device == "cpu" else f"{device.upper()} ({precision})"


def load(csv_path, model, batch):
    """Return {series_label: throughput_ips} for matching rows."""
    out = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("task") != "image_classification":
                continue
            if row.get("model") != model:
                continue
            if batch is not None and int(row.get("batch", 1)) != batch:
                continue
            label = series_label(row["device"], row.get("precision") or "fp32")
            out[label] = float(row["throughput_ips"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results.csv")
    ap.add_argument("--out", default=".")
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load(args.csv, args.model, args.batch)
    if not data:
        raise SystemExit(f"No image_classification rows for model={args.model} batch={args.batch}")
    os.makedirs(args.out, exist_ok=True)

    order = ["CPU", "CUDA (fp32)", "CUDA (fp16)"]
    labels = [l for l in order if l in data] + [l for l in data if l not in order]
    vals = [data[l] for l in labels]
    base = data.get("CPU")

    plt.figure(figsize=(7, 5))
    colors = {"CPU": "#4C72B0", "CUDA (fp32)": "#55A868", "CUDA (fp16)": "#DD8452"}
    bars = plt.bar(labels, vals, color=[colors.get(l, "#888") for l in labels])
    for bar, l, v in zip(bars, labels, vals):
        txt = f"{v:.0f} img/s"
        if base and l != "CPU":
            txt += f"\n({v / base:.1f}x)"
        plt.text(bar.get_x() + bar.get_width() / 2, v, txt,
                 ha="center", va="bottom", fontsize=9)
    plt.ylabel("Throughput (images / sec)")
    plt.title(f"Image Classification ({args.model}, batch={args.batch})")
    plt.margins(y=0.15)
    plt.grid(True, axis="y", alpha=0.3)
    f1 = os.path.join(args.out, f"imgcls_throughput_{args.model}.png")
    plt.tight_layout(); plt.savefig(f1, dpi=150); plt.close()
    print("saved:", f1)


if __name__ == "__main__":
    main()
