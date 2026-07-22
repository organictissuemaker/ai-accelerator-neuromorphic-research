# Paper Notes Template

> Copy this file and rename it to the paper's short title, e.g. `sze2017_efficient_processing.md`

---

## Paper Info
 
- **Title:** Advancing Neuromorphic Computing With Loihi: A Survey of Results and Outlook
- **Authors:** Mike Davies, Andreas Wild, Garrick Orchard, Yulia Sandamirskaya, Gabriel A. Fonseca Guerra, Prasad Joshi, Philipp Plank, Sumedh R. Risbud (Intel Labs)
- **Year:** 2021
- **Venue:** Proceedings of the IEEE, Vol. 109, No. 5 (DOI 10.1109/JPROC.2021.3067593)
- **Link:** https://ieeexplore.ieee.org/document/9395703

---

## What Problem Does It Solve?

For decades neuromorphic chips were proof-of-concept demos that never quantitatively beat conventional hardware. This survey collects the first body of rigorously benchmarked results (accuracy + energy + latency) showing where Intel's Loihi actually wins against CPUs/GPUs/ASICs, and where it doesn't, to map out the real niche for event-driven neuromorphic computing.

---

## Key Contribution

Survey establishing converting ordinary feedforward DNNs to SNNs gives modest energy savings and often worse latency; the order-of-magnitude EDP wins appear only when the workload genuinely uses spiking features — recurrence, precise spike timing, on-chip plasticity, stochasticity, and sparsity. Loihi is explicitly NOT a DNN accelerator. Backs this with ~10 benchmarked workloads on a unified energy-vs-delay plot, all accuracy-controlled.

---

## Architecture / Method (if applicable)

Loihi = a digital, discrete-time, asynchronous many-core SNN research processor.

- Neurons/cores: 131,072 leaky-integrate-and-fire neurons over 128 cores; 128 kB synaptic state + 20 kB routing per core; ~1024 neurons/core.
- Communication: everything is a 32-bit spike message routed over an async on-chip mesh; no clock edges, no compute when idle. 4 off-chip links extend the mesh across chips (but ~30x lower bandwidth than on-chip links → congestion for large nets).
- Configurable weight precision: 1- to signed 9-bit weights; synaptic delays up to 63 timesteps; on-chip programmable plasticity (STDP, Hebbian, reward-modulated).
- von Neumann inside: 3 embedded x86 cores per chip handle spike/dense format conversion and infrequent management tasks.
- Systems: Kapoho Bay (USB, 2 chips) → Nahuku (32 chips) → Pohoiki Springs (768 chips, 100M neurons, <300 W in 5 rack units).
- Key design decision: optimized for sparse, non-batched, batch-size-1 computation — the diametric opposite of the wide-datapath, dense-batch regime of GPUs/DL accelerators.

---

## Results

*What numbers matter? Fill in what they measured.*

| Metric | Their Result | Compared To |
|---|---|---|
| Power | 	Pohoiki Springs <300 W for 100M neurons | a single datacenter CPU server |
| Energy (converted DNNs) | to 100x lower energy	| ANNs on conventional HW |
| Latency (converted DNNs) | on par (small) but worse at multichip scale | rate coding needs many timesteps |
| EDP — Sequential MNIST (LSNN) |	6x10^4 lower (batch 1); 37x vs batched GPU |	GPU |
| EDP — DVS gesture (SLAYER) |	50x lower |	TrueNorth / conventional |
| EDP — robot nav (STDB) |	83x lower |	edge GPU |
| EDP — adaptive PID control (PES)	| ~100x lower; GPU used 43x more power	| CPU / GPU |
| Odor recognition |	92% acc, <3 ms, <1 mJ; +40% over 4 conventional algos on 1-shot |	deep autoencoder |
| SLAM (1-D) |	100x less dynamic power, similar accuracy	| GMapping particle filter on CPU |

---

## How This Connects to My Project

- A synaptic operation on Loihi costs MORE energy than a MAC on a custom ANN accelerator — sparsity support has overhead, and rate coding needs many synops per MAC. 
- Section VII-G, Economic Viability. Loihi is nearly all logic/SRAM; per-bit DRAM is ~100x cheaper than logic-process memory, so pure-logic neuromorphic is cornered into a high-end niche. Davies names the fix explicitly: "dense and cheaply integrated emerging memories, such as crossbars of resistive, magnetic, or phase change devices," plus "storing multiple bits per device." That is SNN + RRAM crossbar. 
- Configurable 1–9 bit weight precision is the digital counterpart of my planned "≤2 bits/cell" RRAM design recommendation — evidence the field already treats weight precision as a first-class knob.

---

## 2-Sentence Explanation

Intel's Loihi is a 131,072-neuron digital spiking chip that beats CPUs/GPUs by 10–100x in energy-delay — but only on brain-like workloads (sparse, recurrent, spike-timed), not on converted feedforward DNNs, and a synaptic op isn't even intrinsically cheaper than a MAC. Its remaining problem is cost: it's all logic, so the survey itself points to RRAM/PCM crossbars with multiple bits per device as the path to affordable large-scale neuromorphic. 

