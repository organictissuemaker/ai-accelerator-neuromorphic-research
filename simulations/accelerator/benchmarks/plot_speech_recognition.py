"""Plot speech_recognition benchmark results.

Bar chart of real-time factor (RTF, lower = faster than real time) by device/precision.
RTF < 1 means transcription is faster than the audio plays.

Usage (from benchmarks/ folder):
    python plot_speech_recognition.py --csv speech_recognition/results.csv --out ../../../results/figures
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
            if row.get("task") != "speech_recognition":
                continue
            if row.get("model") != model:
                continue
            label = series_label(row["device"], row.get("precision") or "fp32")
            out[label] = float(row["rtf"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results.csv")
    ap.add_argument("--out", default=".")
    ap.add_argument("--model", default="tiny")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load(args.csv, args.model)
    if not data:
        raise SystemExit(f"No speech_recognition rows for model={args.model}")
    os.makedirs(args.out, exist_ok=True)

    order = ["CPU", "CUDA (fp32)", "CUDA (fp16)"]
    labels = [l for l in order if l in data] + [l for l in data if l not in order]
    vals = [data[l] for l in labels]
    base = data.get("CPU")
    colors = {"CPU": "#4C72B0", "CUDA (fp32)": "#55A868", "CUDA (fp16)": "#DD8452"}

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, vals, color=[colors.get(l, "#888") for l in labels])
    for bar, l, v in zip(bars, labels, vals):
        txt = f"RTF {v:.3f}"
        if base and l != "CPU":
            txt += f"\n({base / v:.1f}x faster)"
        plt.text(bar.get_x() + bar.get_width() / 2, v, txt,
                 ha="center", va="bottom", fontsize=9)
    plt.axhline(1.0, color="grey", ls="--", lw=1)
    plt.text(len(labels) - 0.5, 1.02, "real-time (RTF=1)", ha="right", fontsize=8, color="grey")
    plt.ylabel("Real-time factor (processing time / audio time, lower = faster)")
    plt.title(f"Speech Recognition RTF (Whisper {args.model})")
    plt.margins(y=0.15); plt.grid(True, axis="y", alpha=0.3)
    f1 = os.path.join(args.out, f"speech_rtf_whisper_{args.model}.png")
    plt.tight_layout(); plt.savefig(f1, dpi=150); plt.close()
    print("saved:", f1)


if __name__ == "__main__":
    main()
