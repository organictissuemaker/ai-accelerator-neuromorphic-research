#!/usr/bin/env python3
"""Task 4 - Hardware performance evaluation of the failure classifier.

Loads the trained model from Task 3 and benchmarks inference of the SAME model on
each available device (CPU, CUDA GPU). Records the metrics the assignment asks for:
model loading time, average inference time per image, total processing time,
throughput, memory consumption, CPU/GPU utilization, and (GPU) power / energy per
inference. Appends one row per device to a CSV for the Task 5 tables and figures.

Repo placement: failure_analysis/ai_classification/fa_benchmark.py
Run (Colab, once per session):  python fa_benchmark.py --data fa_prepared --weights fa_model.pt
Dependencies: torch, torchvision, psutil (pynvml optional for power).
"""
import argparse
import csv
import os
import time
import statistics

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def gpu_power_w():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0, 1)
    except Exception:
        return None


def build(name, n_classes):
    if name == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    else:
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    return m


def load_images(data_dir):
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize(MEAN, STD)])
    imgs = []
    for split in ("test", "train"):
        p = os.path.join(data_dir, split)
        if os.path.isdir(p):
            for x, _ in datasets.ImageFolder(p, tf):
                imgs.append(x.unsqueeze(0))
    return imgs


def bench_device(ckpt, images, device, passes=30):
    proc = None
    try:
        import psutil
        proc = psutil.Process()
    except Exception:
        pass

    on_cuda = device.type == "cuda"

    # ---- model loading time (construct + load weights + move to device) ----
    t0 = time.perf_counter()
    model = build(ckpt["model"], len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    if on_cuda:
        torch.cuda.synchronize()
    load_ms = (time.perf_counter() - t0) * 1000

    imgs = [x.to(device) for x in images]

    # warmup
    with torch.no_grad():
        for x in imgs[:min(5, len(imgs))]:
            model(x)
    if on_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    if proc:
        proc.cpu_percent(None)
        peak_ram = proc.memory_info().rss

    # timed: `passes` sweeps over all images, one image at a time
    times = []
    with torch.no_grad():
        for _ in range(passes):
            for x in imgs:
                t = time.perf_counter()
                model(x)
                if on_cuda:
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t)
                if proc:
                    peak_ram = max(peak_ram, proc.memory_info().rss)

    n = len(times)
    total_s = sum(times)
    avg_ms = statistics.mean(times) * 1000
    return {
        "device": device.type,
        "n_images": len(imgs),
        "inferences_timed": n,
        "model_load_ms": round(load_ms, 2),
        "avg_infer_ms": round(avg_ms, 3),
        "total_time_s": round(total_s, 3),
        "throughput_ips": round(n / total_s, 1),
        "cpu_util_pct": round(proc.cpu_percent(None), 1) if proc else None,
        "ram_mb": round(peak_ram / 1e6, 1) if proc else None,
        "gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if on_cuda else None,
        "gpu_power_w": gpu_power_w() if on_cuda else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="fa_prepared")
    ap.add_argument("--weights", default="fa_model.pt")
    ap.add_argument("--passes", type=int, default=30)
    ap.add_argument("--out", default="fa_hardware_results.csv")
    args = ap.parse_args()

    ckpt = torch.load(args.weights, map_location="cpu")
    images = load_images(args.data)
    print(f"model={ckpt['model']} | classes={len(ckpt['classes'])} | images={len(images)}\n")

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    rows = []
    for dev in devices:
        r = bench_device(ckpt, images, dev, args.passes)
        rows.append(r)
        print(f"[{r['device'].upper():4}] load={r['model_load_ms']}ms  "
              f"avg={r['avg_infer_ms']}ms/img  thr={r['throughput_ips']} img/s  "
              f"cpu={r['cpu_util_pct']}%  ram={r['ram_mb']}MB  "
              f"gpu_mem={r['gpu_mem_mb']}MB  power={r['gpu_power_w']}W")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nsaved: {args.out}")
    if len(rows) == 2:
        sp = rows[0]["avg_infer_ms"] / rows[1]["avg_infer_ms"]
        print(f"GPU speedup vs CPU (avg inference): {sp:.1f}x")


if __name__ == "__main__":
    main()
