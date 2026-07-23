# Paper Notes Template

> Copy this file and rename it to the paper's short title, e.g. `sze2017_efficient_processing.md`

---

## Paper Info

- **Title:** Review of Memristors for In-Memory Computing and Spiking Neural Networks
- **Authors:** Shooshtari et al.
- **Year:** 2026 (published online Dec 2025)
- **Venue:** Advanced Intelligent Systems, Vol. 8, Article e202500806 (41 pp.; open access, CC BY)
- **Link:** https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202500806

---

## What Problem Does It Solve?

IMC and neuromorphic (SNN) computing are both promising routes past the von Neumann bottleneck, but they're usually treated separately. This review unifies them around the one device that serves both roles — the memristor — surveying materials, crossbar integration, synapse/neuron emulation, non-idealities, and applications in one place.

---

## Key Contribution

A single, current review establishing that a memristor is simultaneously (a) an IMC compute unit for crossbar matrix-vector multiply and (b) a spiking synapse/neuron for STDP-based learning — and cataloguing the device non-idealities (variability, drift, endurance, sneak paths, thermal) that stand between lab demos and deployment.

---

## Architecture / Method (if applicable)

- Materials: oxide-based (filamentary VCM/ECM; moderate analog, HIGH endurance; key challenge = variability), organic (high analog, LOW endurance), halide-perovskite, phase-change (Ge2Sb2Te5). TaOx/TiOx flagged as thermally stable. 
- Crossbar MVM: conductance G stores weights; input as word-line voltages; device current I = V·G (Ohm) summed on the bit-line (Kirchhoff) → analog MVM in one step. Passive 0T1R (sneak-path prone) vs 1T1R / 1S1R (per-cell access, sneak-path suppression).
- Synapse/neuron emulation: memristor conductance = synaptic weight; STDP / LTP / LTD via pulse engineering. Threshold-switching memristors act as LIF neurons.
- Integration: three generations from discrete 1T1R → crossbar-over-CMOS → monolithic co-integrated (e.g., NeuRRAM). 1T1R gives current-compliance control that also improves endurance.

---

## Results

*What numbers matter? Fill in what they measured.*

| Metric | Their Result | Compared To |
|---|---|---|
| Computation	| clock-driven continuous MVM	| event-driven spatiotemporal spikes |
| Synaptic precision	| High (≥6–8 bits) |	Moderate (1–4 bits sufficient) |
| Conductance update |	linear, symmetric, low noise |	pulse-sensitive, history-dependent, possibly stochastic |
| Endurance	| High (frequent updates) |	Moderate (sparse updates) |
| Retention	| long-term nonvolatile	| long-term desirable; short-term plasticity useful |
| Neuron function |	analog activation (ReLU/sigmoid)	| integrate-and-fire w/ leak + threshold |
| Energy	| high precision at low power	| ultra-low per spike (<pJ/spike) |

---

## How This Connects to My Project

- SNNs need only moderate endurance because weight updates are sparse and inference is read-dominated — so RRAM's endurance limit (10^6–10^12 from Zahoor) is far less binding for SNN inference than for frequently retrained ANNs. 
- ≤2 bits/cell recommendation. SNNs work at "1–4 bits sufficient" vs ANN "≥6–8 bits."  
- Device-to-device + die-to-die variability, resistance drift, and filament instability "corrupt stored weights" — the precise mechanism my noise-injection sweep models. Endurance spread 10^5–10^9.
- neuromorphic models must increasingly account for non-idealities at the device level
- TaOx/TiOx is thermally stable 
- Proof variability-aware training works: NeuRRAM's noise-/relaxation-aware training → ~software accuracy in hardware. A concrete baseline

---


## 2-Sentence Explanation

This review shows the memristor is the single device that does both jobs my project cares about — analog crossbar matrix-multiply for in-memory computing and spiking-synapse learning - and it catalogs the non-idealities (variability, drift, limited endurance, sneak paths, thermal) that currently keep it in the lab. Its key table tells me SNNs tolerate low precision (1–4 bits) and only moderate endurance because they update sparsely, which both backs my ≤2 bits/cell design choice and answers whether read-dominated SNN inference sidesteps RRAM's endurance limit 

