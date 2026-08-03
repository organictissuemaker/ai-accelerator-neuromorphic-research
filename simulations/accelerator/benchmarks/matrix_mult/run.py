import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, torch
from benchmark_utils import get_device, benchmark, gpu_power_w, log_result

TASK = "matrix_mult"

DTYPES = {"fp32": torch.float32, "fp16": torch.float16}


def make_input(device, n, dtype):
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--size", type=int, nargs="+", default=[1000],
                    help="one or more matrix sizes, e.g. --size 1000 2000 4000 8000")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"],
                    help="fp16 shows the tensor-core / low-precision advantage on GPU")
    args = ap.parse_args()

    device = get_device(args.device)
    dtype = DTYPES[args.dtype]

    if args.dtype == "fp16" and device.type == "cpu":
        print("note: fp16 matmul on CPU is slow/limited — run fp16 on --device cuda.")

    for n in args.size:
        a, b = make_input(device, n, dtype)

        @torch.inference_mode()
        def run_once():
            return torch.matmul(a, b)

        m = benchmark(run_once, n_iter=args.iters)
        m.update({
            "task": TASK,
            "size": f"{n}x{n}",
            "device": device.type,
            "gpu_power_w": gpu_power_w() if device.type == "cuda" else None,
            "precision": args.dtype,
        })
        log_result("results.csv", m)


if __name__ == "__main__":
    main()