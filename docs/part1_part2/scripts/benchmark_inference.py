"""
Part 2 — measured inference on different hardware (this Mac).

Times a small MLP on CPU vs Apple MPS (and CUDA if present) and reports
per-batch latency and throughput. Uses random MNIST-sized input, so no dataset
download is needed. This is a starting point — swap in the real MNIST SNN from
simulations/ if you want the numbers to match Thread A.

Run:  python scripts/benchmark_inference.py
"""
import time
import torch
import torch.nn as nn

BATCH = 256
IN_DIM = 784
ITERS = 50
WARMUP = 10


class MLP(nn.Module):
    def __init__(self, in_dim=IN_DIM, hidden=1000, out=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out),
        )

    def forward(self, x):
        return self.net(x)


def available_devices():
    devs = ["cpu"]
    if torch.backends.mps.is_available():
        devs.append("mps")
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def bench(device):
    model = MLP().to(device).eval()
    x = torch.randn(BATCH, IN_DIM, device=device)
    for _ in range(WARMUP):          # warmup
        model(x)
    sync(device)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        model(x)
    sync(device)
    dt = (time.perf_counter() - t0) / ITERS      # seconds per batch
    latency_ms = dt * 1000
    throughput = BATCH / dt                        # samples / second
    return latency_ms, throughput


def main():
    print("torch", torch.__version__)
    print("model: MLP {}->1000->10  |  batch={}  |  iters={}".format(IN_DIM, BATCH, ITERS))
    print("{:<8} {:>14} {:>18}".format("device", "latency/batch", "throughput"))
    print("-" * 42)
    for d in available_devices():
        lat, thr = bench(d)
        print("{:<8} {:>11.2f} ms {:>13,.0f} img/s".format(d, lat, thr))
    print("\nNote: TPU / NPU are not runnable locally — keep those literature-based.")


if __name__ == "__main__":
    main()
