# Step 1 — Conventional CNN baseline (`simulations/accelerator/`)

LeNet-style CNN on MNIST, benchmarked for inference on CPU and GPU. This is the
conventional-accelerator arm of the Route A same-task comparison against the SNN.

## Files

- `cnn_baseline.py` — train + benchmark script.

## Install

```bash
pip install torch torchvision numpy
```

## Run locally (CPU, and GPU if you have CUDA)

```bash
cd simulations/accelerator
python cnn_baseline.py --epochs 5
```

`--device auto` (default) benchmarks CPU and, if a CUDA GPU is present, GPU too.

## Run the GPU arm on Google Colab

1. Runtime → Change runtime type → **GPU**.
2. In a cell:

   ```python
   !pip -q install torch torchvision numpy
   !python cnn_baseline.py --epochs 5 --device auto --csv results/metrics.csv
   ```

3. Train once and reuse the weights so CPU and GPU numbers come from the *same*
   model:

   ```bash
   python cnn_baseline.py --weights lenet_mnist.pt        # trains + saves
   python cnn_baseline.py --weights lenet_mnist.pt --device cpu
   ```

## Key flags

| Flag | Default | Notes |
|------|---------|-------|
| `--epochs` | 5 | ~99% test accuracy is typical at 5. |
| `--batch-size` | 128 | **Keep identical across CNN and SNN runs.** |
| `--weights` | none | Load if the file exists, else train and save to it. |
| `--device` | auto | `auto` \| `cpu` \| `cuda`. |
| `--csv` | `../../results/metrics.csv` | Appended, not overwritten. |
| `--seed` | 42 | Fixed for reproducibility. |

## Output CSV schema (`results/metrics.csv`)

One row per device per run. The SNN script (Step 2) appends to the **same file**
using `workload_metric=spikes`, so Step 4 can build the comparison table and
Step 5 the spikes-per-inference figure directly from this CSV.

| Column | Meaning |
|--------|---------|
| `timestamp` | UTC ISO timestamp |
| `device_type` / `device_name` | `cpu`/`cuda` and human-readable label |
| `model` / `dataset` | `LeNet` / `MNIST` |
| `batch_size` / `test_samples` | benchmark config (10,000 for MNIST test) |
| `accuracy` | top-1 fraction |
| `latency_ms_per_inf` | wall-clock ms per inference (batched, amortized) |
| `throughput_inf_per_s` | inferences/second |
| `workload_metric` | `MACs` for CNN, `spikes` for SNN |
| `workload_per_inf` | MACs/inference (analytical) or spikes/inference |
| `params` | parameter count |

## Methodology notes (for the Step 6 write-up)

- **MACs** are counted analytically from output tensor shapes via forward hooks
  (Conv: `H·W·Cout·Cin·kH·kW`; Linear: `in·out`). Pooling/ReLU excluded. No
  `thop`/`ptflops` dependency, so the count is transparent and citable.
- **Timing** uses `time.perf_counter()` with CUDA `synchronize()` around each
  batch and a 3-batch warmup (lazy CUDA init + cuDNN autotune). Latency is total
  inference time / test samples.
- **Fixed scope:** same test set (no shuffle) and batch size across every run,
  so CNN vs. SNN numbers are directly comparable.
