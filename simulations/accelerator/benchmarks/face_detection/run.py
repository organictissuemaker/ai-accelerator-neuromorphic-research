"""MediaPipe benchmark: face detection and hand tracking (CPU real-time).

MediaPipe's Python package runs on CPU only (no desktop CUDA path), so this measures
how fast these efficiency-optimized models run without a GPU.
Install once:  pip install mediapipe
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
from benchmark_utils import benchmark, log_result

TASKS = ("face_detection", "hand_gesture")


def build(task):
    import mediapipe as mp
    if task == "face_detection":
        return mp.solutions.face_detection.FaceDetection(model_selection=0,
                                                         min_detection_confidence=0.5)
    return mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                    min_detection_confidence=0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    detector = build(args.task)
    frame = np.random.randint(0, 255, (args.height, args.width, 3), dtype=np.uint8)

    def run_once():
        return detector.process(frame)   # MediaPipe expects an RGB frame

    m = benchmark(run_once, n_warmup=10, n_iter=args.iters, sync_cuda=False)
    m["throughput_ips"] = round(1000.0 / m["latency_ms"], 2)   # FPS
    m.update({
        "task": args.task,
        "model": "mediapipe",
        "resolution": f"{args.width}x{args.height}",
        "device": "cpu",
        "gpu_power_w": None,
        "precision": "fp32",
    })
    log_result("results.csv", m)


if __name__ == "__main__":
    main()
