"""Plot MediaPipe CPU benchmark: FPS for face detection and hand tracking.

Reads rows for both tasks from one or more results.csv files and shows FPS on CPU,
with a 30 FPS real-time reference line.

Usage (from benchmarks/ folder):
    python plot_mediapipe.py --csv face_detection/results.csv hand_gesture/results.csv --out ../../../results/figures
"""
import argparse
import csv
import os

LABELS = {"face_detection": "Face Detection", "hand_gesture": "Hand Tracking"}


def load(csv_paths):
    out = {}
    for p in csv_paths:
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for row in csv.DictReader(f):
                t = row.get("task")
                if t in LABELS:
                    out[t] = float(row["throughput_ips"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", default=["results.csv"])
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load(args.csv)
    if not data:
        raise SystemExit("No face_detection / hand_gesture rows found")
    os.makedirs(args.out, exist_ok=True)

    tasks = [t for t in LABELS if t in data]
    labels = [LABELS[t] for t in tasks]
    vals = [data[t] for t in tasks]

    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, vals, color=["#4C72B0", "#DD8452"])
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.0f} FPS",
                 ha="center", va="bottom", fontsize=10)
    plt.axhline(30, color="grey", ls="--", lw=1)
    plt.text(len(labels) - 0.5, 31, "30 FPS real-time", ha="right", fontsize=8, color="grey")
    plt.ylabel("Throughput on CPU (frames / sec)")
    plt.title("MediaPipe Real-Time Performance (CPU)")
    plt.margins(y=0.15); plt.grid(True, axis="y", alpha=0.3)
    f1 = os.path.join(args.out, "mediapipe_cpu_fps.png")
    plt.tight_layout(); plt.savefig(f1, dpi=150); plt.close()
    print("saved:", f1)


if __name__ == "__main__":
    main()
