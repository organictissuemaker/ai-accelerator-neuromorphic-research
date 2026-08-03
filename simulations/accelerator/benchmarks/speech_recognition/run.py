"""Speech recognition benchmark (OpenAI Whisper Tiny).

Measures transcription latency and real-time factor across CPU / GPU-fp32 / GPU-fp16.
Install once:  pip install openai-whisper
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
from benchmark_utils import benchmark, gpu_power_w, log_result

TASK = "speech_recognition"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--model", default="tiny", help="tiny / base / small ...")
    ap.add_argument("--seconds", type=int, default=15, help="length of test audio")
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    if args.dtype == "fp16" and args.device == "cpu":
        print("note: fp16 needs a GPU — run fp16 with --device cuda.")
        return

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("note: no CUDA available; run GPU benchmarks on Colab.")
        return

    import whisper
    model = whisper.load_model(args.model, device=args.device)
    half = args.dtype == "fp16"

    # fixed synthetic 16 kHz audio so timing is content-independent
    audio = (np.random.randn(16000 * args.seconds).astype(np.float32)) * 0.01

    def run_once():
        return model.transcribe(audio, fp16=half, verbose=False)

    m = benchmark(run_once, n_warmup=1, n_iter=args.iters)
    m["throughput_ips"] = round(1000.0 / m["latency_ms"], 3)     # clips/sec
    rtf = (m["latency_ms"] / 1000.0) / args.seconds              # real-time factor
    m.update({
        "task": TASK,
        "model": args.model,
        "audio_s": args.seconds,
        "rtf": round(rtf, 3),
        "device": args.device,
        "gpu_power_w": gpu_power_w() if args.device == "cuda" else None,
        "precision": args.dtype,
    })
    log_result("results.csv", m)


if __name__ == "__main__":
    main()
