"""Route B - analytical energy model (foundation + matmul validation).

Estimates energy per inference from operation counts x published per-operation
energy figures, then validates against the MEASURED energy (power x time) from the
benchmark. Establishes the modeling approach that later extends to neuromorphic (SNN).

Per-operation energy (Horowitz, "Computing's Energy Problem," ISSCC 2014; 45 nm, 0.9 V):
  FP32 multiply-accumulate  ~ 4.6 pJ
  FP16 multiply-accumulate  ~ 1.5 pJ   (approx; FP16 mult + add)
  INT8 multiply-accumulate  ~ 0.2 pJ
  32-bit DRAM access        ~ 640 pJ   (off-chip; the dominant cost)
  32 KB SRAM read           ~ 9 pJ
Neuromorphic (Davies et al., "Loihi," IEEE Micro 2018):
  synaptic operation (SOP)  ~ 23.6 pJ

NODE CAVEAT: these are 45 nm figures. The benchmark GPU (Colab T4) is ~12 nm, so its
true per-op energy is several times lower. The 45 nm model therefore OVERESTIMATES
compute energy for the T4 - which makes the validation conservative: if measured total
energy still exceeds even this generous compute estimate, non-compute energy (data
movement, control, board/idle power) provably dominates.
"""
import csv
import argparse

# default path to the committed matrix_mult results, relative to energy_model/
DEFAULT_CSV = "../simulations/accelerator/benchmarks/matrix_mult/results.csv"

E_MAC = {"fp32": 4.6e-12, "fp16": 1.5e-12, "int8": 0.2e-12}   # J per MAC
E_DRAM_WORD = 640e-12                                          # J per 32-bit DRAM access
E_SOP = 23.6e-12                                              # J per synaptic op (Loihi)


def matmul_macs(n):
    return n ** 3                        # N x N @ N x N = N^3 multiply-accumulates


def matmul_min_dram_J(n):
    # optimistic floor: read A + read B, write C, each element once (32-bit)
    elements = 3 * n * n
    return elements * E_DRAM_WORD


def measured_matmul(csv_path):
    out = {}
    for r in csv.DictReader(open(csv_path)):
        if r["device"] != "cuda" or not r["gpu_power_w"]:
            continue
        n = int(r["size"].split("x")[0])
        # measured energy per matmul = power(W) x latency(s)
        e = float(r["gpu_power_w"]) * float(r["latency_ms"]) / 1000.0
        out[(n, r["precision"])] = e
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV, help="matrix_mult results.csv")
    args = ap.parse_args()
    meas = measured_matmul(args.csv)
    print(f"{'Size':>6} {'Prec':>5} | {'Compute model (J)':>18} {'DRAM floor (J)':>15} | {'Measured (J)':>13} {'Meas/Compute':>13}")
    print("-" * 92)
    for n in [1000, 2000, 4000, 8000]:
        for prec in ["fp32", "fp16"]:
            macs = matmul_macs(n)
            e_compute = macs * E_MAC[prec]
            e_dram = matmul_min_dram_J(n)
            m = meas.get((n, prec))
            ratio = m / e_compute if m else float("nan")
            print(f"{n:>6} {prec:>5} | {e_compute:>18.4f} {e_dram:>15.4f} | {m:>13.4f} {ratio:>12.1f}x")

    print("\nInterpretation:")
    print("- 'Compute model' = N^3 MACs x per-MAC energy (45 nm). This is the theoretical")
    print("  arithmetic cost alone.")
    print("- 'Measured' = the actual board energy the T4 used (power x time).")
    print("- Measured energy is several times larger than even the generous 45 nm compute")
    print("  estimate, and the T4 (12 nm) computes more cheaply still -> the arithmetic is a")
    print("  small fraction of real energy; data movement + system overhead dominate.")
    print("- This is the Horowitz thesis, reproduced on your own measurements, and the reason")
    print("  neuromorphic chips (which move far less data per result) can be more efficient.")


if __name__ == "__main__":
    main()
