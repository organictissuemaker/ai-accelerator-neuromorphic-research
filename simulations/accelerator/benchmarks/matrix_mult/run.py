import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, torch
from benchmark_utils import get_device, benchmark, gpu_power_w, log_result

TASK = "matrix_mult"
N = 1000

def load_model(device):
    return None  # no model — just the matmul op

def make_input(device):
    a = torch.randn(N, N, device=device)
    b = torch.randn(N, N, device=device)
    return a, b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["cpu","cuda","auto"])
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    device = get_device(args.device)
    a, b = make_input(device)

    @torch.inference_mode()
    def run_once():
        return torch.matmul(a, b)

    m = benchmark(run_once, n_iter=args.iters)
    m.update({"task": TASK, "size": f"{N}x{N}", "device": device.type,
              "gpu_power_w": gpu_power_w() if device.type=="cuda" else None})
    log_result("results.csv", m)

if __name__ == "__main__":
    main()