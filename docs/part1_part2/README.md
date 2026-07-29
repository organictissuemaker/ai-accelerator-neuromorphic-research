# Part 1 & Part 2 — AI Hardware Accelerators & Inference

Documentation for the July 2026 task sheet (Part 1: accelerator study; Part 2:
inference on different hardware). These two parts are a self-contained hardware
study — separate from the SNN / RRAM work in `simulations/`, but they formalize
the groundwork Thread A assumes.

## Layout

```
docs/part1_part2/
├── README.md                  # this file
├── part1_accelerators.md      # Part 1 writeup (CPU/GPU/TPU/NPU study)
├── part2_inference.md         # Part 2 writeup (training vs inference, pipeline, engine)
├── data/
│   └── accelerator_comparison.csv   # the comparison table as data
├── scripts/
│   ├── make_diagrams.py        # generates all architecture + pipeline figures
│   └── benchmark_inference.py  # optional: measured CPU-vs-MPS inference on this Mac
└── figures/                    # generated PNGs (run make_diagrams.py to populate)
```

## Regenerate the figures

```bash
cd docs/part1_part2
python scripts/make_diagrams.py     # writes figures/*.png
```

Diagrams use Georgia bold labels (house style). They render correctly on macOS;
on machines without Georgia, matplotlib substitutes a default serif.

## Optional measured benchmark (Part 2)

```bash
cd docs/part1_part2
python scripts/benchmark_inference.py
```

Times a small model on CPU vs Apple MPS and prints latency / throughput. This
turns "inference on different hardware" from theory into measured numbers on your
own machine. TPU / NPU can't be run locally, so those stay literature-based.

## Requirements

`matplotlib` for diagrams; `torch` for the benchmark (already installed for the
simulations). Nothing new to add to `requirements.txt` except `matplotlib` if it
isn't there yet.
