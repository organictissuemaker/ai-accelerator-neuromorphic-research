"""Plot OCR benchmark results: processing latency (ms) by engine/device.

Usage (from benchmarks/ folder):
    python plot_ocr.py --csv ocr/results.csv --out ../../../results/figures
"""
import argparse
import csv
import os


def series_label(engine, device):
    name = {"tesseract": "Tesseract", "easyocr": "EasyOCR"}.get(engine, engine)
    return f"{name}\n(CPU)" if device == "cpu" else f"{name}\n(GPU)"


def load(csv_path):
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("task") != "ocr":
                continue
            rows.append((series_label(row["engine"], row["device"]),
                         float(row["latency_ms"])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results.csv")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load(args.csv)
    if not rows:
        raise SystemExit("No ocr rows found in " + args.csv)
    os.makedirs(args.out, exist_ok=True)

    # keep a sensible order
    order = ["Tesseract\n(CPU)", "EasyOCR\n(CPU)", "EasyOCR\n(GPU)"]
    rows.sort(key=lambda r: order.index(r[0]) if r[0] in order else 99)
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = {"Tesseract\n(CPU)": "#8172B3", "EasyOCR\n(CPU)": "#4C72B0",
              "EasyOCR\n(GPU)": "#55A868"}

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, vals, color=[colors.get(l, "#888") for l in labels])
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.0f} ms",
                 ha="center", va="bottom", fontsize=10)
    plt.ylabel("Processing time per image (ms, lower = faster)")
    plt.title("OCR Processing Time by Engine / Hardware")
    plt.margins(y=0.15); plt.grid(True, axis="y", alpha=0.3)
    f1 = os.path.join(args.out, "ocr_latency.png")
    plt.tight_layout(); plt.savefig(f1, dpi=150); plt.close()
    print("saved:", f1)


if __name__ == "__main__":
    main()
