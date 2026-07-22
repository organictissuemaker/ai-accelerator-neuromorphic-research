"""
Arrhenius Plot Analysis of MOSFET TDDB Reliability
--------------------------------------------------
Failure-analysis assignment 1 (Dr. Kalya).

Model:  TDDB(T) = A * exp(Ea / (k_B * T))
        => ln(TDDB) = ln(A) + (Ea/k_B) * (1/T)
A straight line on ln(TDDB) vs 1/T. Slope = Ea/k_B, so Ea = slope * k_B.

Two independent stress datasets are fit separately to check reproducibility of
the extracted activation energy.

House plot style: Georgia bold titles/axis labels, 8 pt bold ticks, dots s=50.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../results/figures"))
os.makedirs(FIGURES_DIR, exist_ok=True)

K_B = 8.617333e-5  # Boltzmann constant, eV/K

# ── Dataset (from assignment sheet) ──────────────────────────────────────────
TC = np.array([100, 125, 150, 175, 200, 225])      # deg C
TK = np.array([373, 398, 423, 448, 473, 498], float)  # K (as given on sheet)
TDDB1 = np.array([27000, 10200, 3000, 900, 300, 110], float)  # SET 1, seconds
TDDB2 = np.array([26000,  9800, 3100, 880, 280, 130], float)  # SET 2, seconds


def arrhenius_fit(tddb, T):
    """Fit ln(TDDB) = ln(A) + (Ea/k_B)*(1/T). Returns dict of fit results."""
    x = 1.0 / T
    y = np.log(tddb)
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    r2 = 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    return {
        "slope": slope,
        "intercept": intercept,
        "Ea": slope * K_B,          # eV
        "A": np.exp(intercept),     # seconds
        "r2": r2,
    }


FONT_TITLE = {"family": "Georgia", "weight": "bold", "size": 13}
FONT_LABEL = {"family": "Georgia", "weight": "bold", "size": 11}


def style_ticks(ax):
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("Georgia"); lbl.set_fontweight("bold"); lbl.set_fontsize(8)


def main():
    f1 = arrhenius_fit(TDDB1, TK)
    f2 = arrhenius_fit(TDDB2, TK)

    print("=== Arrhenius fit ===")
    for name, f in [("SET 1", f1), ("SET 2", f2)]:
        print(f"{name}: Ea = {f['Ea']:.3f} eV | A = {f['A']:.3e} s | R^2 = {f['r2']:.4f}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ── Panel A: raw TDDB vs temperature (log-y) ──
    # Failure rate ~ 1/TDDB, so a falling TDDB curve = rising failure rate with T.
    axA.scatter(TC, TDDB1, s=50, color="#2F5496", zorder=3, label="TDDB Set 1")
    axA.scatter(TC, TDDB2, s=50, color="#C00000", marker="s", zorder=3, label="TDDB Set 2")
    axA.plot(TC, TDDB1, color="#2F5496", linewidth=1, alpha=0.5)
    axA.plot(TC, TDDB2, color="#C00000", linewidth=1, alpha=0.5)
    axA.set_yscale("log")
    axA.set_xlabel("Temperature (" + r"$\degree$" + "C)", fontdict=FONT_LABEL)
    axA.set_ylabel("Time-to-Breakdown (s)", fontdict=FONT_LABEL)
    axA.set_title("TDDB Collapses with Temperature\n" +
                  r"($\bf{245\times}$" + " faster breakdown, 100" + r"$\rightarrow$" + "225" + r"$\degree$" + "C)",
                  fontdict=FONT_TITLE)
    axA.grid(True, which="both", alpha=0.25)
    axA.legend(prop={"family": "Georgia", "size": 9})
    style_ticks(axA)

    # ── Panel B: Arrhenius plot ln(TDDB) vs 1000/T ──
    x = 1000.0 / TK  # 1000/T for readable axis
    axB.scatter(x, np.log(TDDB1), s=50, color="#2F5496", zorder=3, label="Set 1 data")
    axB.scatter(x, np.log(TDDB2), s=50, color="#C00000", marker="s", zorder=3, label="Set 2 data")
    xline = np.linspace(x.min(), x.max(), 100)
    # convert: fit slope is per (1/T); here axis is 1000/T, so slope_plot = slope/1000
    axB.plot(xline, f1["slope"] * (xline / 1000.0) + f1["intercept"],
             color="#2F5496", linewidth=1.6,
             label=r"Set 1 fit: $\bf{E_a=%.2f}$ eV" % f1["Ea"])
    axB.plot(xline, f2["slope"] * (xline / 1000.0) + f2["intercept"],
             color="#C00000", linewidth=1.6, linestyle="--",
             label=r"Set 2 fit: $\bf{E_a=%.2f}$ eV" % f2["Ea"])
    axB.set_xlabel(r"$1000\,/\,T$  (K$^{-1}$)", fontdict=FONT_LABEL)
    axB.set_ylabel("ln( TDDB / s )", fontdict=FONT_LABEL)
    axB.set_title("Arrhenius Fit " + r"$\rightarrow$" + " Activation Energy\n" +
                  r"(slope $= E_a/k_B$;  $\bf{R^2 > 0.99}$)", fontdict=FONT_TITLE)
    axB.grid(True, alpha=0.25)
    axB.legend(prop={"family": "Georgia", "size": 9})
    style_ticks(axB)

    # secondary top axis with temperature in C
    def invK_to_C(v):  # v = 1000/T
        return 1000.0 / v - 273.15
    secax = axB.secondary_xaxis("top", functions=(lambda v: v, lambda v: v))
    tick_C = np.array([100, 150, 200, 225])
    secax.set_xticks(1000.0 / (tick_C + 273.15))
    secax.set_xticklabels([f"{c}" for c in tick_C])
    secax.set_xlabel("Temperature (" + r"$\degree$" + "C)", fontdict={"family": "Georgia", "weight": "bold", "size": 9})
    for lbl in secax.get_xticklabels():
        lbl.set_fontfamily("Georgia"); lbl.set_fontweight("bold"); lbl.set_fontsize(8)

    plt.suptitle("MOSFET TDDB — Arrhenius Reliability Analysis",
                 fontfamily="Georgia", fontweight="bold", fontsize=14)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "arrhenius_tddb.png")
    plt.savefig(out, dpi=150)
    plt.show()
    print("Saved:", out)

    # acceleration factor 100 -> 225 C
    print(f"\nAcceleration factor (100->225C): Set1 {TDDB1[0]/TDDB1[-1]:.0f}x, Set2 {TDDB2[0]/TDDB2[-1]:.0f}x")


if __name__ == "__main__":
    main()
