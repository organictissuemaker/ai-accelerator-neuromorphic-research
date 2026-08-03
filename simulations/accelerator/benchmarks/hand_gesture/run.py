"""MediaPipe benchmark: face detection and hand tracking (CPU real-time).

Uses the current MediaPipe Tasks API (the legacy mp.solutions API was removed).
Model bundles are downloaded automatically on first run.
MediaPipe Python runs on CPU only (no desktop CUDA path).
Install once:  pip install mediapipe
"""
import sys, os, urllib.request
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
from benchmark_utils import benchmark, log_result

TASKS = ("face_detection", "hand_gesture")

MODELS = {
    "face_detection": (
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        "blaze_face_short_range.tflite",
    ),
    "hand_gesture": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task",
        "hand_landmarker.task",
    ),
}


def ensure_model(task):
    url, fname = MODELS[task]
    if not os.path.exists(fname):
        print("downloading", fname)
        urllib.request.urlretrieve(url, fname)
    return fname


def build(task):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    base = mp_python.BaseOptions(model_asset_path=ensure_model(task))
    if task == "face_detection":
        return vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(base_options=base))
    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(base_options=base, num_hands=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    import mediapipe as mp
    detector = build(args.task)
    frame = np.random.randint(0, 255, (args.height, args.width, 3), dtype=np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    def run_once():
        return detector.detect(mp_image)

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
