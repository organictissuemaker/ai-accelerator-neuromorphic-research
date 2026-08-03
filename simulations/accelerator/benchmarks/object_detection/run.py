"""Object detection benchmark (YOLOv8 via ultralytics).

Measures detection latency and FPS across CPU / GPU-fp32 / GPU-fp16.
Install once:  pip install ultralytics
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
from benchmark_utils import benchmark, gpu_power_w, log_result

TASK = "object_detection"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--model", default="yolov8n", help="yolov8n / yolov8s / ...")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    if args.dtype == "fp16" and args.device == "cpu":
        print("note: fp16 needs a GPU — run fp16 with --device cuda.")
        return

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("note: no CUDA available; run GPU benchmarks on Colab.")
        return

    from ultralytics import YOLO
    model = YOLO(f"{args.model}.pt")           # auto-downloads weights
    dev = 0 if args.device == "cuda" else "cpu"

    # only pass the half-precision flag when we actually want fp16
    # (passing half=False triggers a deprecation warning on newer ultralytics)
    extra = {"half": True} if args.dtype == "fp16" else {}

    # fixed synthetic frame so timing doesn't depend on disk / content
    frame = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)

    def run_once():
        return model.predict(frame, device=dev, imgsz=args.imgsz,
                             verbose=False, **extra)

    m = benchmark(run_once, n_warmup=5, n_iter=args.iters)
    m["throughput_ips"] = round(1000.0 / m["latency_ms"], 2)   # FPS
    m.update({
        "task": TASK,
        "model": args.model,
        "device": args.device,
        "gpu_power_w": gpu_power_w() if args.device == "cuda" else None,
        "precision": args.dtype,
    })
    log_result("results.csv", m)


if __name__ == "__main__":
    main()
