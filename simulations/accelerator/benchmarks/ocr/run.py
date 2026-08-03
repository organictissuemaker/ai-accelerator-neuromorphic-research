"""OCR benchmark: Tesseract (classical, CPU) vs EasyOCR (deep learning, CPU/GPU).

Compares text-recognition processing time across engines and hardware.
Install:  pip install pytesseract easyocr  (+ the tesseract binary:
          Colab: apt-get install -y tesseract-ocr ; Mac: brew install tesseract)
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
from benchmark_utils import benchmark, gpu_power_w, log_result

TASK = "ocr"
SAMPLE = ["The quick brown fox", "jumps over the lazy dog",
          "0123456789", "AI hardware benchmark"]


def make_image(w=640, h=480):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 36)
    except Exception:
        font = ImageFont.load_default()
    y = 30
    for line in SAMPLE:
        d.text((30, y), line, fill="black", font=font)
        y += 90
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["tesseract", "easyocr"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    device = args.device
    if args.engine == "tesseract" and device == "cuda":
        print("note: Tesseract is CPU-only; recording as CPU.")
        device = "cpu"

    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            print("note: no CUDA available; run GPU benchmarks on Colab.")
            return

    img = make_image()

    if args.engine == "tesseract":
        import pytesseract
        def run_once():
            return pytesseract.image_to_string(img)
    else:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=(device == "cuda"))
        arr = np.array(img)
        def run_once():
            return reader.readtext(arr)

    m = benchmark(run_once, n_warmup=2, n_iter=args.iters, sync_cuda=(device == "cuda"))
    m["throughput_ips"] = round(1000.0 / m["latency_ms"], 3)   # images/sec
    m.update({
        "task": TASK,
        "engine": args.engine,
        "device": device,
        "gpu_power_w": gpu_power_w() if device == "cuda" else None,
        "precision": "fp32",
    })
    log_result("results.csv", m)


if __name__ == "__main__":
    main()
