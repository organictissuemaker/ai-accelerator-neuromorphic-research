"""Plot object_detection benchmark results: FPS bar chart by device/precision.

Usage (from benchmarks/ folder):
    python plot_object_detection.py --csv object_detection/results.csv --out ../../../results/figures
"""
import argparse
import csv
import os


def series_label(device, precision):
    return "CPU" if device == "cpu" else f"{device.upper()} ({precision})"


def load(csv_path, model):
    out = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("task") != "object_detection":
                continue
            if row.get("model") != model:
                continue
            label = series_label(row["device"], row.get("precision") or "fp32")
            out[label] = float(row["throughput_ips"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results.csv")
    ap.add_argument("--out", default=".")
    ap.add_argument("--model", default="yolov8n")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load(args.csv, args.model)
    if not data:
        raise SystemExit(f"No object_detection rows for model={args.model}")
    os.makedirs(args.out, exist_ok=True)

    order = ["CPU", "CUDA (fp32)", "CUDA (fp16)"]
    labels = [l for l in order if l in data] + [l for l in data if l not in order]
    vals = [data[l] for l in labels]
    base = data.get("CPU")
    colors = {"CPU": "#4C72B0", "CUDA (fp32)": "#55A868", "CUDA (fp16)": "#DD8452"}

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, vals, color=[colors.get(l, "#888") for l in labels])
    for bar, l, v in zip(bars, labels, vals):
        txt = f"{v:.1f} FPS"
        if base and l != "CPU":
            txt += f"\n({v / base:.1f}x)"
        plt.text(bar.get_x() + bar.get_width() / 2, v, txt,
                 ha="center", va="bottom", fontsize=9)
    plt.ylabel("Throughput (frames / sec)")
    plt.title(f"Object Detection FPS ({args.model})")
    plt.margins(y=0.15); plt.grid(True, axis="y", alpha=0.3)
    f1 = os.path.join(args.out, f"objdet_fps_{args.model}.png")
    plt.tight_layout(); plt.savefig(f1, dpi=150); plt.close()
    print("saved:", f1)


if __name__ == "__main__":
    main()
