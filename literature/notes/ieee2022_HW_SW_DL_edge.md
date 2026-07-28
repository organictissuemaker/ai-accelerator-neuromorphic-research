# Paper Notes Template

> Copy this file and rename it to the paper's short title, e.g. `sze2017_efficient_processing.md`

---

## Paper Info

- **Title:** Networks of Spiking Neurons: The Third Generation of Neural Network Models
- **Authors:** F. Ponzina, S. Machetti, M. Rios, B. W. Denkinger, A. Levisse, G. Ansaloni, M. Peón-Quirós, D. Atienza (Embedded Systems Lab, EPFL)
- **Year:** 2022 (Nov/Dec issue) 
- **Venue:** IEEE Micro, vol. 42, no. 6, pp. 48–54. DOI 10.1109/MM.2022.3195617
- **Link:** https://ieeexplore.ieee.org/document/9847079

---

## What Problem Does It Solve?

Edge AI must run compute-intensive CNNs (~10⁹ FLOPs/inference) on devices tightly constrained in compute, memory, and energy. Hardware and software optimizations are usually done in isolation, leaving efficiency on the table; the paper argues they must be co-designed — algorithmic transformations conceived jointly with the hardware that exploits them.

---

## Key Contribution

A concrete co-design framework: software resiliency → aggressive quantization → hardware exploitation.

- Ensembles (E²CNNs) raise robustness/accuracy
- Aggressive, heterogeneous quantization of weights + activations
- Domain-specific hardware — multiple ultra-low-power cores, independently power-managed memory banks, voltage scaling
- In-memory computing (BLADE), an SRAM-based digital IMC accelerator.

Each software choice is made because a hardware mechanism can exploit it, and vice versa. That loop is the "interaction."

---

## Architecture / Method (if applicable)

- E²CNNs (software I): prune an untrained CNN, then replicate it into several lightweight instances, each trained independently from different initial weights. If pruning factor > replication factor, total storage drops. Because instances have different weight distributions, a memory error that wrecks one instance is often harmless to the others → ensemble votes correctly. This is redundancy-based error tolerance.
- Heterogeneous quantization (software II): greedy heuristic picks which layer to shrink next, ranked by sensitivity × layer size (big layers = bigger gains). Prunes all-zero filters. Produces fine-grained per-layer bitwidths well-suited to IMC.
- Domain-specific hardware (III): multiple RISC-V in-order cores + DMA + CGRA accelerator (up to 32% energy gain vs single-core); multibanked memory where each bank is independently power-gated / put in retention; clock- and power-gating during idle/sync; multi-Vt cell libraries + voltage/frequency scaling.
- BLADE IMC (IV): SRAM-based (6T bitcells), fully digital — no analog elements, no ADC/DAC. Activates two word-lines of different local groups; computes on global bit-lines read by sense amplifiers. Supports add/subtract/shift/bitwise → chained into MAC → SIMD across subarrays.
- Key design decision: they explicitly considered eNVM crossbars (RRAM, PCM, MRAM) for IMC but rejected them here, citing nonconventional fabrication, complex ADC periphery, and high write currents — choosing SRAM IMC for manufacturability and drop-in compatibility.

---

## Results

*What numbers matter? Fill in what they measured.*

| Metric | Their Result | Compared To |
|---|---|---|
| Power/Energy | E²CNN: −55% energy (GoogLeNet 8× prune → 4 instances, halves mem/compute); voltage scaling: up to −60% (memory subsystem), up to −90% memory energy; CGRA: −32%; BLADE: −72% energy | full-precision / single-core / homogeneous 8-bit single-instance CNN |
| Latency | 	−72% (BLADE), from low-bitwidth SIMD MAC in-memory | homogeneous 8-bit single-instance CNN |
| Accuracy | ≤1% degradation (BLADE result); no degradation for E²CNN GoogLeNet on CIFAR-100 | full-precision baseline |

---

## How This Connects to My Project

- It names the interaction — co-design: energy ↔ latency ↔ accuracy as one trade-off. Direct support for scaffolding operations → energy/op → latency.
- Differing position: They pick SRAM IMC over RRAM crossbars, and the reasons they list (ADC periphery, high write current, variability) are the RRAM weaknesses my project confronts. I can position my work as: "given RRAM's ~10–100× density/energy upside, how much of its variability can an SNN actually tolerate before it stops being worth it?" — the question this paper sidesteps by avoiding RRAM.
- Ensemble error-tolerance ≈ my σ-sweep, mirror image. They add redundancy to tolerate memory/weight errors; I measure the raw tolerance of one SNN to weight variability (W' = W·(1+N(0,σ))). 
- Quantization-as-a-knob backs my ≤2 bits/cell recommendation. Their "cycles ∝ bitwidth" and heterogeneous low-bitwidth scheme is the general principle; my experiment supplies the SNN+RRAM evidence for where the bit floor sits.
- Contrast for my synthesis: this is a CNN + digital SRAM-IMC design; mine is SNN + analog RRAM-IMC. 

---

## 2-Sentence Explanation

Running modern CNNs on tiny edge devices only works if you design the model and the chip together: make the model robust with ensembles so you can quantize it aggressively, then build hardware (low-power cores + in-SRAM computing) that turns those low bitwidths into real energy and speed. Doing this jointly yielded ~72% energy and latency savings at ≤1% accuracy loss — and notably they chose SRAM in-memory computing over RRAM crossbars, sidestepping exactly the device-variability problem my project measures.
