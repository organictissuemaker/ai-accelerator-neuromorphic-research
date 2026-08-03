import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, torch
import torchvision.models as models
from benchmark_utils import get_device, benchmark, gpu_power_w, log_result

TASK = "image_classification"

DTYPES = {"fp32": torch.float32, "fp16": torch.float16}

MODELS = {
    "resnet18": models.resnet18,
    "mobilenet_v2": models.mobilenet_v2,
}


def load_model(name, device, dtype):
    m = MODELS[name](weights="DEFAULT")   # pretrained ImageNet weights
    m = m.to(device=device, dtype=dtype).eval()
    return m


def make_input(device, dtype, batch):
    return torch.randn(batch, 3, 224, 224, device=device, dtype=dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--model", default="resnet18", choices=list(MODELS))
    ap.add_argument("--batch", type=int, nargs="+", default=[1],
                    help="one or more batch sizes, e.g. --batch 1 8 32")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    device = get_device(args.device)
    dtype = DTYPES[args.dtype]

    if args.dtype == "fp16" and device.type == "cpu":
        print("note: fp16 on CPU is unsupported for these models — run fp16 on --device cuda.")
        return

    model = load_model(args.model, device, dtype)   # load once, reuse across batches

    for batch in args.batch:
        x = make_input(device, dtype, batch)

        @torch.inference_mode()
        def run_once():
            return model(x)

        m = benchmark(run_once, n_iter=args.iters)
        # throughput in images/sec accounts for batch size
        m["throughput_ips"] = round(batch / (m["latency_ms"] / 1000.0), 2)
        m.update({
            "task": TASK,
            "model": args.model,
            "batch": batch,
            "device": device.type,
            "gpu_power_w": gpu_power_w() if device.type == "cuda" else None,
            "precision": args.dtype,
        })
        log_result("results.csv", m)


if __name__ == "__main__":
    main()
