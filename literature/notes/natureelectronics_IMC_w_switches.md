# Paper Notes Template

> Copy this file and rename it to the paper's short title, e.g. `sze2017_efficient_processing.md`

---

## Paper Info

- **Title:** In-memory computing with resistive switching devices
- **Authors:** Daniele Ielmini (Politecnico di Milano), H.-S. Philip Wong (Stanford)
- **Year:** 2018
- **Venue:** Nature Electronics, Vol. 1, June 2018, 333–343 (Review Article)
- **Link:** https://www.nature.com/articles/s41928-018-0092-2

---

## What Problem Does It Solve?

The von Neumann "memory wall" — shuttling data between separate memory and processing units — caps modern computing. This foundational review lays out how in-memory computing (IMC) with resistive switching devices (RRAM, PCM, STT-MRAM) computes inside the memory to eliminate that data movement, and surveys the digital, analog, and stochastic schemes plus the physics and challenges. 

---

## Key Contribution

Defines and organizes the whole IMC-with-resistive-devices field: (1) digital in-memory logic (stateful/IMP gates), (2) analog crosspoint MVM (the crossbar accelerator), and (3) stochastic computing / true random number generation. Establishes the key design principle that analog crosspoint arrays compute only approximate results and must be restricted to error-tolerant tasks (inference, pattern recognition) — the frame my experiment operates in.
---

## Architecture / Method (if applicable)

- Resistive devices: two-terminal RRAM / PCM / STT-MRAM; resistance (not charge) stores state; multilevel possible. RRAM = filamentary (localized conductive path); PCM = bulk amorphous/crystalline.
- Digital IMC: stateful logic (IMP, R–R gates) where resistance states are logic inputs and results, computed in place.
- Analog crosspoint MVM (the core, Eq. 1): I_i = Σ_j G_ij·V_j. Conductance matrix G = weights; voltage vector V = input; row currents = result. Computed in ONE step via Ohm's law (I = V·G per device) + Kirchhoff's law (currents sum on the line). Cell area 4F².
- Stochastic computing / TRNG: exploits cycle-to-cycle switching variability (random set-delay, telegraph noise) to generate physical random numbers with 50% bit probability.
- Scaling paths: shrink cell diameter (raises cell-to-cell variability); interconnect scaling raises series resistance → IR drop; 3D horizontal/vertical stacked arrays for density; carbon-nanotube/graphene interconnects to cut line resistivity.
- Key design decision: computational-memory devices will differ from storage-memory devices — must be co-optimized against system metrics (accuracy, energy, switching speed). 

---

## Results

*What numbers matter? Fill in what they measured.*

| Aspeect | Statement |
|---|---|
| Crosspoint MVM |	I_i = Σ_j G_ij·V_j in 1 step (Ohm + Kirchhoff) vs multi-step digital MAC | 
| Cell size	| 4F² (F = lithographic feature size) |
| Accuracy	| crosspoint arrays compute APPROXIMATE results only | 
| Suitable tasks	| error-tolerant: pattern recognition, page ranking, data inference |
| Variability role	| hurts deterministic/digital IMC; helps stochastic/neural functions |
| Correction	| verify-and-correct works for storage, NOT easily for compute |
| RRAM vs PCM	| RRAM more unstable (atomic-scale filament sensitivity); PCM drifts over time |
---

## How This Connects to My Project

*Why did I read this? How does it inform my simulation or architecture design?*

---

## What I'd Explore Further

*What question did this paper raise that I want to investigate?*

---

## 2-Sentence Explanation

*Practice explaining this paper simply. 

