import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, torch
from benchmark_utils import get_device, benchmark, gpu_power_w, log_result

TASK = "matrix_mult"


def make_input(device, n):
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--size", type=int, nargs="+", default=[1000],
                    help="one or more matrix sizes, e.g. --size 1000 2000 4000 8000")
    args = ap.parse_args()

    device = get_device(args.device)

    for n in args.size:
        a, b = make_input(device, n)

        @torch.inference_mode()
        def run_once():
            return torch.matmul(a, b)

        m = benchmark(run_once, n_iter=args.iters)
        m.update({
            "task": TASK,
            "size": f"{n}x{n}",
            "device": device.type,
            "gpu_power_w": gpu_power_w() if device.type == "cuda" else None,
        })
        log_result("results.csv", m)


if __name__ == "__main__":
    main()