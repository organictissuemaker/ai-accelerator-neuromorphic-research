# Paper Notes Template

> Copy this file and rename it to the paper's short title, e.g. `sze2017_efficient_processing.md`

---

## Paper Info

- **Title:** Exploring Neuromorphic Computing Based on Spiking Neural Networks: Algorithms to Hardware
- **Authors:** Nitin Rathi, Indranil Chakraborty, Adarsh Kosta, Abhronil Sengupta, Aayush Ankit, Priyadarshini Panda, Kaushik Roy (Purdue / Penn State / Yale)
- **Year:** 2023 
- **Venue:** ACM Computing Surveys, Vol. 55, No. 12, Article 243 (49 pp.)
- **Link:** https://dl.acm.org/doi/full/10.1145/3571155

---

## What Problem Does It Solve?

Neuromorphic computing needs progress on two fronts at once — new spiking algorithms and new hardware to run them — and the two are usually surveyed separately. This paper covers the full stack (device → circuit → algorithm) in one place and lays out where the open challenges are, especially for scaling SNNs onto emerging non-volatile-memory hardware.

---

## Key Contribution

A comprehensive, co-design survey that maps the SNN algorithm space (neuron models, input encodings, training methods, architectures) onto the neuromorphic hardware space (CMOS neuromorphic chips and NVM/crossbar in-memory computing), and names device variability + analog-compute error as the central barrier to scaling NVM-based SNN systems. 

---

## Architecture / Method (if applicable)

Neuron model: LIF/IF (Eq. 2 = the discrete update rule my lif_neuron.py implements: u_i^t = λ·u_i^(t-1) + Σ w_ij o_j^t − v·o_i^(t-1)). Also stochastic/probabilistic neuron model (fires with a sigmoid probability of weighted input).

Input encoding: rate coding (info in mean firing rate; simple but slow) · temporal coding (info in spike timing; fewer spikes, harder) · trained encoding layer.

Training methods (the core tradeoff table):

- ANN-to-SNN conversion: map ReLU→IF by threshold balancing; matches ANN accuracy; costs ~1,000 timesteps latency (percentile-threshold trick → 60–80). Loses timing info.
- Surrogate-gradient / spike-based backprop (BPTT): approximates the non-differentiable spike derivative; uses temporal info → lower latency; but memory/compute scales with timesteps → limited to small datasets (MNIST/CIFAR, few layers).
- Hybrid: convert first (good init), then surrogate-gradient fine-tune (train leak + threshold too). Solves both high latency and high training cost. ← likely my Part 2.
- STDP (unsupervised, local): hardware-friendly, but shallow/simple tasks only; two+ STDP layers give little benefit. Stochastic STDP enables binary-weight training.

Hardware — NVM in-memory computing:

- Motivation: CMOS hits on-chip density + area limits emulating neuron/synapse functions.
- Devices: RRAM (HfOx, TiOx, SrTiO3), PCM (chalcogenide), spintronics (MTJ/DWM), FeFET. Each can act as an analog synapse or a LIF/stochastic neuron directly in device physics.
- Crossbar MVM: conductance G stores the weight matrix; input applied as row voltages; device current I_ij = V_i · G_ij (Ohm); currents sum on the bitline (Kirchhoff) → matrix-vector multiply in place, no weight movement.
- Device figures of merit: bit resolution, programming energy/speed, ON/OFF conductance ratio, reliability, endurance. Low ON/OFF ratio limits achievable accuracy.
Key design point: RRAM switching now down to ~ns via material research

---

## Results

| Metric | Their Result | Compared To |
|---|---|---|

| Conversion latency	| ~1,000 timesteps (percentile trick: 60–80) | rate-coded ANN-to-SNN |
| Surrogate-gradient latency	| lower than conversion	| but training memory ∝ timesteps |
| STDP depth |	shallow only; >2 layers little gain	| vs supervised deep SNNs |
| Crossbar MVM | cost	I = V·G summed on bitline (analog, 1 step) |	vs digital MAC arrays |
| RRAM switching speed |	~ns (recent)	| earlier slow/high-power RRAM |

---

## How This Connects to My Project

- Crossbar MVM = I_ij = V_i·G_ij summed on the bitline (Ohm + Kirchhoff). This is the "eliminate movement" box of my design-space chart.
- "Device variability and reliability poses a big challenge in realizing scalable systems using NVM-based neural primitives"; analog NVM compute is "erroneous in nature and requires modeling and sufficient mitigation."
- Crossbar demonstrations are "limited to ANN workloads," though they "can be used to map SNNs trained using supervised learning." SNN-on-RRAM-crossbar is explicitly less explored than ANN-on-crossbar 
- HfOx is listed as an RRAM oxide — same material as my Weibull HfO2 variability data, tightening the failure-analysis ↔ RRAM bridge.
- The hybrid method (convert → surrogate-gradient fine-tune, train leak+threshold) is the practical way to get a trained SNN with a measurable accuracy.

---

## 2-Sentence Explanation

This survey covers the whole neuromorphic stack: how you train spiking networks (ANN conversion, surrogate-gradient backprop, hybrid, STDP — each trading accuracy against latency and training cost) and how you run them (CMOS chips, or RRAM/PCM/spintronic crossbars that do matrix-vector multiply in place via Ohm's and Kirchhoff's laws).

