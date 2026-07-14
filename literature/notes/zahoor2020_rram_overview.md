# Paper Notes Template

> Copy this file and rename it to the paper's short title, e.g. `sze2017_efficient_processing.md`

---

## Paper Info

- **Title:** Resistive Random Access Memory (RRAM): an Overview of Materials, Switching Mechanism, Performance, Multilevel Cell (MLC) Storage, Modeling, and Applications
- **Authors:**  Furqan Zahoor, Tun Zainal Azni Zulkifli, Farooq Ahmad Khanday
- **Year:** 2020
- **Venue:** Nanoscale Research Letters (open-access review; 930+ citations)
- **Link:** https://link.springer.com/article/10.1186/s11671-020-03299-9

---

## What Problem Does It Solve?

All three conventional memories store charge (DRAM: capacitor, SRAM: cross-coupled
inverter nodes, Flash: floating gate), and charge storage breaks down below ~10 nm —
stored charge leaks away, killing reliability and noise margin. On top of that, DRAM
burns power on refresh every few ms, and Flash is slow (write/erase ~1 ms/0.1 ms),
wears out (~10⁶ cycles), and needs >10 V. This review maps RRAM, which stores
resistance instead of charge, as the leading candidate to escape those walls.

---

## Key Contribution

- Ideal memory checklist: <1 V operation, >10¹⁷ cycle endurance, >10-year
retention, fJ/bit energy, <10 nm scalable. Nothing meets all of these; every
memory is a compromise.
- Resistance-based contenders: PCM (crystalline↔amorphous), STT-MRAM (magnetic
tunnel junction), RRAM (conductive filament). RRAM wins on simplicity: a
metal-insulator-metal (MIM) sandwich, CMOS-compatible, easy to fabricate.
- Positioning: RRAM is pitched as a Flash replacement (faster, lower voltage) —
NOT yet a DRAM replacement. RRAM beats DRAM on non-volatility, density, and scalability, but loses badly on endurance (10⁶–10¹² cycles vs DRAM's effectively unlimited) and suffers variability. Working memory gets rewritten constantly — endurance is disqualifying for now.

---

## Architecture / Method (if applicable)

- Cell structure: MIM — insulating oxide (HfOx, TaOx, TiOx...) between two metal
electrodes. Deposited by ALD. HRS = logic 0, LRS = logic 1.
- Switching = filament formation/rupture: Forming (first-time controlled soft
breakdown of the oxide) creates a nanoscale conductive filament → LRS. RESET
partially ruptures it → HRS. SET re-forms it. A compliance current (via a
selector transistor, "1T1R") prevents permanent hard breakdown.
- Two filament chemistries: CBRAM/ECM — filament of metal atoms (Ag/Cu
dissolve from active electrode, migrate, reduce at inert electrode).
OxRRAM/VCM — filament of oxygen vacancies (O²⁻ knocked to the anode
"oxygen reservoir", vacancies left behind form the path). OxRRAM has much better
endurance because vacancies drift/diffuse less easily than metal atoms.
- Two switching modes: unipolar (same polarity, Joule heating ruptures filament)
vs bipolar (opposite polarity, field-driven ion migration; most practical devices).
- Key design decision (field-wide): filament chemistry + oxide/electrode choice
set the endurance/retention/uniformity trade-off; TaOx currently best endurance.

---

## Results

*What numbers matter? Fill in what they measured.*

| Metric | Their Result | Compared To |
|---|---|---|
| Power | | |
| Latency | | |
| Accuracy | | |
| Write speed | ns-scale switching| Flash: ~1 ms write, 0.1 ms erase |
| Endurance | TaOx-based: >10¹² cycles (typical HfOx arrays: 10⁶) | Flash: ~10⁶ · DRAM: ~unlimited | 
| Retention | 10 years (extrapolated, 85–150 °C stress) | DRAM: ms (needs refresh) |
| Operating voltage | SET <1 V (forming <3 V) | Flash: >10 V |
| Density | MLC: multiple bits/cell via multiple resistance levels | beyond 1-bit charge cells |
| Weakness | cycle-to-cycle + device-to-device variability (stochastic filament) | the main barrier to scale-up |

---

## How This Connects to My Project

- This is the device behind IMC. GB10 notes ended with "what removes the hose?"
— RRAM crossbars are the answer candidate: store weights as conductances, apply
voltages, and Ohm's + Kirchhoff's laws compute the matrix-vector multiply in the
memory itself. Zero weight movement. 
- RRAM forming is deliberately induced soft breakdown. Retention testing uses the workflow: stress at
high temp (150 °C) → Arrhenius extrapolation to 10-year lifetime. Endurance
distributions are analyzed with Weibull plots.
- Analog resistance levels (MLC) map naturally to synaptic weights for SNNs — my N-neuron simulation. Corrected framing for slides: RRAM ≠ DRAM. 
  - (1) Flash replacement,
  - (2) storage-class memory between DRAM and Flash
  - (3) analog compute substrate
for AI 

---

## 2-Sentence Explanation

RRAM stores bits as the resistance of a metal-insulator-metal cell, using a voltage
to grow or rupture a nanometer-scale conductive filament (metal atoms or oxygen
vacancies) — non-volatile, nanosecond-fast, <1 V, and scalable below 10 nm where
charge-based memories fail. Its weaknesses are write endurance and the stochastic
filament's variability, which is why it's positioned as a Flash successor and analog
in-memory-compute substrate rather than a DRAM replacement.
