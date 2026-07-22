# Failure-Analysis Assignments — Results & RRAM Connections (Week 4)

Scripts: `simulations/failure_analysis/arrhenius_tddb.py`, `weibull_tddb.py`
Figures: `results/figures/arrhenius_tddb.png`, `weibull_tddb.png`

---

## 1. Arrhenius — TDDB vs Temperature

### Model
`TDDB(T) = A · exp(Ea / kB·T)`  →  `ln(TDDB) = ln(A) + (Ea/kB)·(1/T)`.
Straight line on ln(TDDB) vs 1/T; slope = Ea/kB, so **Ea = slope × kB**.

### Results
| Dataset | Ea (eV) | Prefactor A (s) | R² |
|--------|---------|-----------------|-----|
| Set 1  | **0.717** | 7.1e-6 | 0.993 |
| Set 2  | **0.702** | 1.1e-5 | 0.993 |

- Two independent stress runs give the **same Ea to within 0.015 eV** → the extracted
  activation energy is a real material property, not fitting noise. Reproducibility is
  itself the headline.
- TDDB falls **~245× (Set 1) / 200× (Set 2)** from 100 °C → 225 °C. Failure rate
  (∝ 1/TDDB) therefore *rises* steeply with temperature.

### (a) Failure rate variation with temperature
Failure rate ∝ 1/TDDB. Because TDDB is exponential in 1/T, a modest temperature rise
compresses lifetime dramatically — the left panel drops nearly 2.5 decades over 125 °C.

### (b) TDDB variation with temperature
Monotonic exponential decay; linear once plotted as ln(TDDB) vs 1/T (right panel,
R² > 0.99). The straight-line quality confirms a **single, thermally-activated
degradation mechanism** dominates across the whole 100–225 °C range (no change in slope
= no change in mechanism).

### (c) Activation energy
**Ea ≈ 0.71 eV.** Sits in the accepted band for oxide TDDB (~0.6–0.9 eV), consistent
with thermally-activated bond breakage / defect (oxygen-vacancy) generation in the
dielectric. The tight two-set agreement lets us quote it with confidence.

### (d) Reliability interpretation
The clean Arrhenius line is what licenses **lifetime extrapolation**: stress the part
hot to fail it in minutes-to-hours, fit Ea, then project TDDB down to the operating
temperature. High Ea = strongly temperature-sensitive = accelerated tests are very
efficient, but also means hot operation is punishing. Any lifetime spec must state its
temperature.

### → RRAM connection (closing paragraph)
> This Arrhenius workflow is exactly the methodology behind RRAM's "10-year retention"
> claim (Zahoor 2020). An RRAM cell's retention — how long its programmed conductance
> state survives — is thermally activated in the same form: the filament/vacancy
> configuration relaxes over time, faster when hot. Measuring retention at room
> temperature would take a decade, so labs bake cells at elevated temperature, extract
> an activation energy from an Arrhenius fit of state-loss time vs 1/T, and extrapolate
> to the 85 °C spec. The MOSFET-TDDB Ea I just extracted and an RRAM retention Ea are
> the *same kind of number produced by the same fit*. My failure-analysis toolkit is
> therefore not a side task — it is the reliability toolkit for the memory device at the
> center of my project.

---

## 2. Weibull — TDDB at 15 HfO₂ locations

### Model
`F(t) = 1 − exp(−(t/η)^β)`  →  `ln(−ln(1−F)) = β·ln(t) − β·ln(η)`.
Slope on the linearised plot = **β (shape = spread)**; η = scale = t at F = 63.2 %.
- β < 1: infant-mortality / defect-driven · β ≈ 1: random · β > 1: wear-out (bigger β = tighter).

### Results — the data is BIMODAL (this is the finding)
| Fit | β | η (s) | R² | note |
|-----|-----|-------|------|------|
| Single Weibull (LSQ) | 1.76 | 84 | **0.83** | poor — wrong model |
| Single Weibull (MLE) | 2.20 | 82 | — | poor |
| **Early** population (7 sites) | **7.2** | 37 | — | tight |
| **Late** population (8 sites) | **19.6** | 108 | — | very tight |

Sorted TDDB (s): `25 30 32 35 36 41 42 | 98 100 101 103 105 107 110 115`
There is a **clean empty gap between 42 s and 98 s** — no location breaks down in
between. The single-Weibull line (grey dashed on the plot) runs straight through that
empty gap, fitting *neither* cluster: the visual proof that one distribution is wrong.

### (a) Why Weibull for TDDB
Oxide breakdown is a **weakest-link / extreme-value** process — the dielectric fails at
its single most defective spot, not its average. Weibull is the extreme-value
distribution for "first defect to percolate," so it's the natural (and standard) model
for time-to-breakdown.

### (b) Physical meaning of the parameters
- **β (shape)** = how *tight* the breakdown-time distribution is = the device-to-device
  spread. Each of the two sub-populations here is internally tight (β = 7 and 20), but
  *pooled together they masquerade as a broad, low-β population* (β ≈ 1.8). β is the
  parameter that quantifies variability.
- **η (scale)** = characteristic life (63.2 % failed). The two populations sit at
  η ≈ 37 s and ≈ 108 s — a **~2.9× separation** in characteristic lifetime.

### (c) Reliability + microstructure interpretation
Two populations on a polycrystalline HfO₂ film map cleanly onto its **microstructure**:
- **Early / weak (η ≈ 37 s):** locations on or near **grain boundaries**. Grain
  boundaries are leakage/diffusion highways with higher trap density — breakdown
  percolates there first.
- **Late / robust (η ≈ 108 s):** locations in **grain interiors**, closer to the
  intrinsic dielectric strength.

Reliability consequence: the device is only as good as its *weak* population. A spec
based on the pooled mean (η ≈ 84 s) would be dangerously optimistic — 7 of 15 sites fail
by 42 s. The microstructure sets a floor that averaging hides. This is the reliability
argument *for* controlling grain structure in high-K dielectrics.

### → RRAM connection (closing paragraph)
> This dataset is device-to-device **variability data on HfO₂, the #1 RRAM switching
> oxide** — the same material, and RRAM *forming* is a deliberately induced soft
> breakdown of exactly this kind. The bimodal β is the microscopic origin of the "RRAM
> variability" that Zahoor 2020 calls the technology's core weakness: the same grain
> boundaries that break down early here are the sites where a filament forms with a
> different conductance, cell to cell. So this plot is direct evidence that HfO₂
> breakdown is spatially non-uniform at the grain scale — the physical root of the weight
> noise my SNN-on-RRAM experiment injects. **Caveat (keep stating this):** a spread in
> breakdown *times* is not numerically a spread in programmed *resistance* — I cannot
> plug β or η straight in as a weight-noise σ. What this gives me is (i) the toolkit
> (Weibull fitting), (ii) qualitative proof the variability is real and microstructural,
> and (iii) a bridge to ask Dr. Kalya for the lab's own resistance-distribution data,
> which *would* give me a calibrated σ.

---

## Quick slide points (drop-in)

**Arrhenius slide**
- Ea ≈ **0.71 eV**, both datasets agree to 0.015 eV, R² > 0.99
- TDDB collapses ~**245×** over 100→225 °C; failure rate ∝ 1/TDDB rises with T
- Single straight line = single degradation mechanism across the range
- *This is the exact method behind RRAM's 10-yr retention spec (bake hot, fit Ea, extrapolate)*

**Weibull slide**
- Data is **bimodal**: β_early ≈ 7 (η ≈ 37 s), β_late ≈ 20 (η ≈ 108 s); single fit β ≈ 1.8 is a poor descriptor (R² = 0.83)
- Two populations = **grain boundary (weak) vs grain interior (robust)** in polycrystalline HfO₂
- Device is limited by its weak population — pooled mean hides the floor
- *This IS device-to-device variability on the #1 RRAM oxide; RRAM forming = deliberate soft breakdown*
- **Caveat on the slide:** breakdown-time spread ≠ resistance spread → evidence + toolkit + bridge, not a plug-in σ

**Numbers to know cold**
- kB = 8.617e-5 eV/K; Ea = slope × kB
- Weibull: β = spread/shape, η = 63.2 % life; β>1 wear-out, bigger β = tighter
- Weakest-link → Weibull is the natural TDDB model
