# Step 2 — Spiking neural network baseline (`simulations/snn/`)

Spiking LeNet (Leaky Integrate-and-Fire neurons) on MNIST, trained with
surrogate-gradient backprop-through-time in snnTorch. Produces the neuromorphic
arm of the Route A comparison and the spike-count energy proxy (Step 3).

## Files

- `snn_baseline.py` — train + benchmark + spike counting.

## Install

```bash
pip install torch torchvision snntorch numpy
```

## Run locally (CPU + Apple GPU)

```bash
cd simulations/snn
python snn_baseline.py --epochs 5 --timesteps 25
```

## Run the CUDA arm on Colab

Same procedure as Step 1 (Runtime → GPU, upload the file), then:

```python
!pip -q install torch torchvision snntorch numpy
!python snn_baseline.py --epochs 5 --timesteps 25 --device auto
```

## Key flags

| Flag | Default | Notes |
|------|---------|-------|
| `--epochs` | 5 | |
| `--batch-size` | 128 | **Must match the CNN run.** |
| `--timesteps` | 25 | T: simulation steps per inference. More T = more accuracy + more spikes. |
| `--beta` | 0.95 | LIF membrane decay. |
| `--weights` | none | Load if present, else train and save. |
| `--device` | auto | `auto` \| `cpu` \| `cuda` \| `mps`. |
| `--csv` | `../../results/metrics.csv` | Appends to the SAME file as the CNN. |

## What it logs

Same CSV schema as `cnn_baseline.py`, so both arms live in one file:

- `model` = `SpikingLeNet(T=25)` (records the timestep count).
- `workload_metric` = **`spikes`** (vs. `MACs` for the CNN).
- `workload_per_inf` = **spikes-per-inference** = total spikes / test samples.
  This is the Step 3 proxy metric. It is device-independent, so the CPU/GPU rows
  share the same value; only latency/throughput differ per device.

## How the SNN differs from the CNN (for the Step 6 write-up)

- **LIF neurons** integrate input into a membrane potential and emit a binary
  spike only when it crosses threshold — sparse, event-driven activations vs. the
  CNN's dense floating-point activations.
- **Time dimension:** each inference runs over T timesteps. Input uses *direct
  encoding* (the static image is fed as constant current every step; conv1+LIF1
  act as the learned encoder), which beats Poisson rate coding at low T.
- **Classification:** output spikes are summed over time; argmax gives the class.
- **Energy proxy:** total spikes across all LIF layers / samples. Fewer spikes
  ≈ lower neuromorphic energy — the efficiency claim the project tests.
- **Training:** surrogate gradients (`fast_sigmoid`) make the non-differentiable
  spike function trainable via ordinary backprop through the unrolled T steps.

## Expected ballpark

At T=25, spiking LeNet on MNIST typically reaches ~98% accuracy. Exact spikes/inf
depends on beta and training; record whatever the run reports. To explore the
accuracy-vs-energy tradeoff (Step 5 scatter), rerun with `--timesteps 10` and
`--timesteps 50` — each adds rows to the CSV.
