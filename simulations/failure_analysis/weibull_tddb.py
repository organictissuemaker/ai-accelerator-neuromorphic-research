"""
Weibull Plot Analysis of MOSFET TDDB at 15 locations on polycrystalline HfO2
----------------------------------------------------------------------------
Failure-analysis assignment 2 (Dr. Kalya).

Weibull CDF:  F(t) = 1 - exp( -(t/eta)^beta )
Linearised:   ln( -ln(1 - F) ) = beta * ln(t) - beta * ln(eta)
=> plotting ln(-ln(1-F)) vs ln(t) gives a straight line of slope beta (shape)
   and the scale eta is the t at F = 63.2%.

beta is the headline: it measures the SPREAD of breakdown times.
  beta < 1 : infant-mortality / defect-driven (failure rate falls with time)
  beta ~ 1 : random (constant failure rate)
  beta > 1 : wear-out (failure rate rises with time); larger beta = tighter spread.

This dataset is BIMODAL, so a single Weibull is a poor descriptor — that is
itself the physical result (two breakdown populations from the microstructure).

House plot style: Georgia bold titles/axis labels, 8 pt bold ticks, dots s=50.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../results/figures"))
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Dataset: TDDB (s) at 15 dielectric locations ─────────────────────────────
TDDB = np.array([25, 100, 105, 35, 42, 98, 32, 107, 115, 30, 41, 103, 101, 36, 110], float)


def weibull_mle(data):
    """Maximum-likelihood Weibull shape (beta) and scale (eta)."""
    b = 1.0
    for _ in range(200):
        lb = np.log(data)
        S0 = np.sum(data ** b)
        S1 = np.sum(data ** b * lb)
        S2 = np.sum(data ** b * lb * lb)
        f  = S1 / S0 - 1.0 / b - lb.mean()
        df = (S2 * S0 - S1 * S1) / S0 ** 2 + 1.0 / b ** 2
        bn = b - f / df
        if abs(bn - b) < 1e-12:
            b = bn; break
        b = bn
    eta = (np.sum(data ** b) / len(data)) ** (1.0 / b)
    return b, eta


def median_rank_points(data):
    """Return sorted t, plotting positions (x=ln t, y=ln(-ln(1-F))) via Bernard's approx."""
    ts = np.sort(data)
    n = len(ts)
    i = np.arange(1, n + 1)
    F = (i - 0.3) / (n + 0.4)
    return ts, np.log(ts), np.log(-np.log(1 - F)), F


FONT_TITLE = {"family": "Georgia", "weight": "bold", "size": 13}
FONT_LABEL = {"family": "Georgia", "weight": "bold", "size": 11}


def style_ticks(ax):
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("Georgia"); lbl.set_fontweight("bold"); lbl.set_fontsize(8)


def main():
    ts, x, y, F = median_rank_points(TDDB)

    # single-population fit (least squares on the linearised plot)
    beta_lsq, c = np.polyfit(x, y, 1)
    eta_lsq = np.exp(-c / beta_lsq)
    yhat = beta_lsq * x + c
    r2 = 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    beta_mle, eta_mle = weibull_mle(ts)

    # bimodal split at the visible gap (42 s | 98 s)
    low  = ts[ts < 70]
    high = ts[ts >= 70]
    bL, eL = weibull_mle(low)
    bH, eH = weibull_mle(high)

    print("=== Weibull fit ===")
    print(f"Single (LSQ): beta={beta_lsq:.2f}  eta={eta_lsq:.1f}s  R^2={r2:.3f}")
    print(f"Single (MLE): beta={beta_mle:.2f}  eta={eta_mle:.1f}s")
    print(f"LOW  cluster (n={len(low)}):  beta={bL:.1f}  eta={eL:.1f}s")
    print(f"HIGH cluster (n={len(high)}): beta={bH:.1f}  eta={eH:.1f}s")

    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    # colour points by cluster to make the two populations obvious
    is_low = ts < 70
    ax.scatter(x[is_low],  y[is_low],  s=50, color="#C00000", zorder=3,
               label="Early population (grain-boundary paths)")
    ax.scatter(x[~is_low], y[~is_low], s=50, color="#2F5496", zorder=3,
               label="Late population (grain interior)")

    # single-population fit line
    xl = np.linspace(x.min() - 0.05, x.max() + 0.05, 100)
    ax.plot(xl, beta_lsq * xl + c, color="grey", linewidth=1.4, linestyle="--",
            label=r"Single Weibull: $\bf{\beta=%.2f}$ ($R^2=%.2f$, poor)" % (beta_lsq, r2))

    # annotate the gap
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)  # F = 63.2% line
    ax.text(0.02, 0.06, r"$F = 63.2\%$  ($t=\eta$)", transform=ax.transAxes,
            family="Georgia", fontsize=8, color="black", alpha=0.6)

    ax.set_xlabel(r"ln( TDDB / s )", fontdict=FONT_LABEL)
    ax.set_ylabel(r"ln( $-$ln( 1 $-$ F ) )", fontdict=FONT_LABEL)
    ax.set_title(
        "Weibull Plot — TDDB at 15 HfO" + r"$_2$" + " Locations\n" +
        r"Bimodal: two breakdown populations ($\bf{\beta_{early}=%.0f}$, $\bf{\beta_{late}=%.0f}$)"
        % (bL, bH),
        fontdict=FONT_TITLE)
    ax.grid(True, alpha=0.25)
    ax.legend(prop={"family": "Georgia", "size": 9}, loc="lower right")
    style_ticks(ax)

    # annotation box with the story
    txt = (f"Early: {len(low)} sites, "
           r"$\eta\approx$" + f"{eL:.0f}s\n"
           f"Late:  {len(high)} sites, "
           r"$\eta\approx$" + f"{eH:.0f}s\n"
           r"$\Delta\eta\approx$" + f"{eH/eL:.1f}" + r"$\times$")
    ax.text(0.03, 0.72, txt, transform=ax.transAxes, family="Georgia", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#F2F2F2", edgecolor="grey", alpha=0.9))

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "weibull_tddb.png")
    plt.savefig(out, dpi=150)
    plt.show()
    print("Saved:", out)


if __name__ == "__main__":
    main()
