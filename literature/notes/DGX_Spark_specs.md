# Paper Notes Template

---

## Paper Info

- **Title:** NVIDIA DGX Spark — GB10 Grace Blackwell Superchip (product page, datasheet + Tom's Hardware review)
- **Authors:** NVIDIA (GB10 co-designed with MediaTek)
- **Year:** 2025 (launched Oct 2025)
- **Venue:** Industry product (not peer-reviewed — treat numbers as vendor claims)
- **Link:** https://www.nvidia.com/en-us/products/workstations/dgx-spark/ | review: https://www.tomshardware.com/pc-components/gpus/nvidia-dgx-spark-review

---

## What Problem Does It Solve?

Large models (70–200B parameters) don't fit in the VRAM of any desktop GPU, so local
development requires the cloud. DGX Spark puts enough memory capacity on a desk
(140 W, 1.2 kg box) that a single developer can load, fine-tune, and run models locally.
---

## Key Contribution

A desktop "AI supercomputer" built around capacity over bandwidth: instead of fast
but small HBM VRAM, the GB10 uses one large pool of 128 GB coherent unified LPDDR5x
shared by CPU and GPU — eliminating the CPU-RAM → PCIe → VRAM copy entirely.
Not a new architecture idea (it's the von Neumann model, accepted and worked around),
but a new design point: fits-on-a-desk capacity at 140 W.

---

## Architecture / Method (if applicable)

- CPU: 20-core Arm (10× Cortex-X925 performance + 10× Cortex-A725 efficiency, big.LITTLE)
- GPU: Blackwell, 6,144 CUDA cores, 192 5th-gen Tensor Cores (the MAC units — same job as Eyeriss PEs)
- Memory: 128 GB LPDDR5x, 256-bit interface, 273 GB/s, coherent unified address space (one bucket, CPU+GPU share it — zero-copy)
- Interconnect: NVLink C2C joins the CPU and GPU chiplets (TSMC 3nm-class)
- Scale-out: ConnectX-7 @ 200 Gbps links two Sparks → models up to 405B params
- Key design decision: trade memory bandwidth (273 GB/s vs ~TB/s HBM) for memory capacity (128 GB vs 24–32 GB VRAM), at desktop power/price.

---

## Results

*What numbers matter? Fill in what they measured.*

| Metric | Their Result | Compared To |
|---|---|---|
| Power (chip TDP) | 140W | Loihi 2 ~1 W · V100 ~300 W · GB200 ~kW-scale |
| Compute | 1 PFLOP @ FP4 (with sparsity — marketing-best number) | | 
| Memory bandwidth | 273 GB/s | V100 HBM2 ~900 GB/s (2017!) |
| Capacity | 128 GB unified → ~200B params @ FP4 |  FP4RTX 4090: 24 GB |
| Est. LLM inference | ~2.7 tokens/s for a 100 GB model (273/100 — my back-of-envelope) | Fine for 1 dev, not for serving |

---

## How This Connects to My Project

- Memory-bound, not compute-bound. One token = one full read of all weights.
100 GB weights ÷ 273 GB/s ≈ 2.7 tok/s while the 1-PFLOP compute sits mostly idle. This is the Sze 2017 / von Neumann bottleneck at chip scale — GB10 accepts it and optimizes capacity; PIM/IMC (my Week 4 RRAM reading) tries to eliminate it by computing where the weights live.
- FP4 = bandwidth compression. When memory-bound, quantizing 16→4 bits is a ~4× speedup because 4× less data moves per token. Compression is speed.
- Add GB10 to arch_comparison.py: ~140 W, latency point from the tokens/s estimate above; sits between Loihi 2 (~1 W, sparse/event-driven) and GB200
(kW, datacenter) on the power axis — nice middle point on the scatter.
- Design-space framing for slides: Loihi 2 = minimize power via sparsity · GB10 = maximize local capacity per watt · GB200 = maximize throughput. Same bottleneck, three different answers.

---

## 2-Sentence Explanation

DGX Spark packs a 20-core Arm CPU and a Blackwell GPU sharing one 128 GB unified memory pool into a 140 W desktop box, so models up to ~200B parameters fit locally with no CPU→GPU copying. Its modest 273 GB/s bandwidth makes it memory-bound (~2–3 tokens/s on a 100 GB model) — proof that in modern AI hardware, capacity decides what you can run and bandwidth decides how fast you run it.
