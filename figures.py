"""Manuscript figures (300 dpi PNG + PDF), generated from the result tables."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .network import build_network, route_table, RouteGeometry  # noqa: E402
from .physics import TIER_NAMES  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
GRAY, INK, MUTED = "#8a8985", "#0b0b0b", "#52514e"
TIER_COLORS = ["#cfe0f5", "#7fb0e8", "#2a78d6", "#123f78"]   # sequential blue, light -> dark

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.5,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white",
})
SINGLE, DOUBLE = 3.5, 7.16


def _save(fig, out, name):
    fig.savefig(os.path.join(out, f"{name}.png"))
    fig.savefig(os.path.join(out, f"{name}.pdf"))
    plt.close(fig)


def fig_network(out):
    """Schematic (not to scale) track diagram of the simulated network."""
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.5))
    L = lambda xs, ys, c=INK, lw=1.6: ax.plot(xs, ys, color=c, lw=lw, solid_capstyle="round", zorder=2)
    L([0, 20], [0, 0]); L([0, 20], [1, 1])                                   # Up / Down main
    for x0 in (4.6, 14.6):                                                    # station loops
        L([x0, x0 + 0.3, x0 + 0.5, x0 + 0.8], [0, -0.8, -0.8, 0], GRAY, 1.2)
        ax.add_patch(plt.Rectangle((x0 + 0.05, -1.25), 0.7, 0.3, color="#e6e5e1", zorder=1))
    L([10, 10.4], [0, 1], ORANGE, 1.6)                                        # junction crossover J -> DJ
    L([10.4, 11.2, 12.0], [1, 2.2, 3.0], BLUE)                                # curved branch (schematic)
    L([12.0, 12.0], [3.0, 7.6], BLUE)                                         # single line
    L([12.0, 12.4, 12.4, 12.0], [4.6, 5.0, 5.8, 6.2], GRAY, 1.2)              # passing loop
    pts = {"U0": (0, 0), "U6": (20, 0), "D0": (0, 1), "D6": (20, 1), "J": (10, 0), "DJ": (10.4, 1),
           "Pw": (12.0, 4.6), "Pe": (12.0, 6.2), "T": (12.0, 7.6)}
    for n, (x, y) in pts.items():
        ax.plot(x, y, "o", ms=3.5, color="white", mec=INK, mew=0.8, zorder=3)
        left = n in ("U6", "D6", "Pw", "Pe", "T")
        ax.text(x + (-0.18 if left else 0.15), y + (0.0 if n in ("Pw", "Pe", "T") else 0.18), n, fontsize=6.3,
                color=MUTED, ha="right" if left else "left", va="center" if n in ("Pw", "Pe", "T") else "bottom")
    ax.annotate("", xy=(3.2, 0), xytext=(1.6, 0), arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    ax.annotate("", xy=(1.6, 1), xytext=(3.2, 1), arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    ax.text(6.2, -0.38, "Up main (eastbound)", fontsize=6.3, color=MUTED)
    ax.text(3.4, 1.2, "Down main (westbound)", fontsize=6.3, color=MUTED)
    notes = [(5.0, -1.55, "Station 1 loop: occupied-route entry,\ninterlocking/route mismatch (Balasore-type)"),
             (15.0, -1.55, "Station 2 loop"),
             (7.2, 2.7, "Flat junction J/DJ:\ncrossing conflict, SPAD"),
             (12.7, 5.2, "Passing loop on single line:\nhead-on, low-adhesion overrun\n(Talerddig-type)"),
             (16.0, 3.3, "Automatic-block section:\nrear-end, stationary train,\ndegraded written authority\n(Rangapani-type)")]
    for x, y, t in notes:
        ax.text(x, y, t, fontsize=6.2, color=INK, ha="center" if x < 12.5 or x > 15.5 else "left", va="top")
    ax.set_xlim(-0.5, 21); ax.set_ylim(-2.6, 8.2); ax.axis("off")
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], color=c, lw=1.6) for c in (INK, GRAY, BLUE, ORANGE)],
              ["Double-track main line (20 km)", "Loop", "Single-line branch (R = 1500 m)", "Junction crossover"],
              loc="upper left", frameon=False, ncol=1)
    _save(fig, out, "fig_network")


def fig_labels(out, df):
    ct = pd.crosstab(df["scenario"], df["label"]).reindex(columns=[0, 1, 2, 3], fill_value=0)
    ct = ct.loc[ct.sum(1).sort_values().index]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.9))
    left = np.zeros(len(ct))
    for k in range(4):
        ax.barh(ct.index.str.replace("_", " "), ct[k], left=left, color=TIER_COLORS[k], edgecolor="white",
                linewidth=0.8, label=TIER_NAMES[k], height=0.72)
        left += ct[k].to_numpy()
    ax.set_xlabel("Encounters (seed 0)"); ax.grid(axis="y", visible=False)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.4, 1.0), frameon=False)
    _save(fig, out, "fig_labels_by_scenario")


SHOW = ["Fixed-threshold", "DRCRI index", "Nominal physics", "Physics shield", "LR", "RF", "HGB", "MLP",
        "HGB+CS", "HGB+CP", "HGB+Shield", "PSCL(HGB)"]


def fig_tradeoff(out, R):
    g = R.groupby("method")[["false_emergency", "critical_miss"]].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(SINGLE, 2.8))
    style = {"baseline": (GRAY, "s"), "ml": (BLUE, "o"), "ml+cost": (VIOLET, "D"), "ml+cp": (AQUA, "^"),
             "ml+shield": (YELLOW, "v"), "proposed": (ORANGE, "*")}
    fam = R.drop_duplicates("method").set_index("method")["family"]
    for m in SHOW:
        c, mk = style[fam[m]]
        x, xs = g.loc[m, ("false_emergency", "mean")] * 100, g.loc[m, ("false_emergency", "std")] * 100
        y, ys = g.loc[m, ("critical_miss", "mean")] * 100, g.loc[m, ("critical_miss", "std")] * 100
        ax.errorbar(x, max(y, 0.05), xerr=xs, yerr=ys, fmt=mk, color=c, ms=9 if mk == "*" else 5,
                    mec="white", mew=0.6, elinewidth=0.7, capsize=1.5)
        off = {"HGB": (5, 2), "RF": (-14, -9), "MLP": (-20, 4), "HGB+CS": (5, 2), "Physics shield": (5, 2),
               "HGB+CP": (5, 4), "HGB+Shield": (5, -9), "LR": (5, 2)}.get(m, (4, 2))
        ax.annotate(m, (x, max(y, 0.05)), xytext=off, textcoords="offset points", fontsize=5.8, color=INK)
    ax.set_yscale("log")
    ax.set_xlabel("False-emergency rate, P(decision ≥ High | Low) [%]")
    ax.set_ylabel("Missed-Critical rate [%] (log)")
    _save(fig, out, "fig_tradeoff")


def fig_confusion(out, cm_a, cm_b):
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE * 0.62, 2.2))
    for ax, cm, t in zip(axs, (cm_a, cm_b), ("HGB (arg-max)", "PSCL(HGB)")):
        cm = np.asarray(cm, float); rn = cm / cm.sum(1, keepdims=True)
        ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{int(cm[i, j])}\n{rn[i, j]*100:.1f}%", ha="center", va="center", fontsize=5.8,
                        color="white" if rn[i, j] > 0.55 else INK)
        ax.set_xticks(range(4), TIER_NAMES, rotation=30)
        ax.set_yticks(range(4), TIER_NAMES if ax is axs[0] else [""] * 4)
        ax.set_xlabel("Decided tier"); ax.set_title(t); ax.grid(False)
    axs[0].set_ylabel("True tier")
    _save(fig, out, "fig_confusion")


def fig_conformal(out, S):
    g = S.groupby("alpha").mean(numeric_only=True)
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE * 0.7, 2.3))
    a = g.index.to_numpy()
    axs[0].plot(1 - a, 1 - a, color=GRAY, lw=1, ls="--", label="Target 1 − α")
    for col, c, lab in [("cov_moderate", AQUA, "Moderate"), ("cov_high", BLUE, "High"), ("cov_critical", ORANGE, "Critical")]:
        axs[0].plot(1 - a, g[col], marker="o", ms=4, lw=1.6, color=c, label=lab)
    axs[0].set_xlabel("Nominal level 1 − α"); axs[0].set_ylabel("P(decision ≥ Y | Y = k)")
    axs[0].legend(frameon=False)
    axs[1].plot(1 - a, g["false_emergency"] * 100, marker="o", ms=4, lw=1.6, color=VIOLET)
    axs[1].set_xlabel("Nominal level 1 − α"); axs[1].set_ylabel("False-emergency rate [%]")
    _save(fig, out, "fig_conformal")


def fig_shift(out, H):
    order = ["Nominal physics", "Physics shield", "HGB", "HGB+CP", "HGB+CP (recalibrated)", "PSCL(HGB)",
             "PSCL(HGB) recalibrated"]
    g = H.groupby("method")[["critical_miss", "severe_miss", "false_emergency"]].agg(["mean", "std"]).loc[order]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.6))
    x = np.arange(len(order)); w = 0.27
    for i, (col, c, lab) in enumerate([("critical_miss", ORANGE, "Missed Critical"),
                                       ("severe_miss", BLUE, "Severe miss"),
                                       ("false_emergency", GRAY, "False emergency")]):
        ax.bar(x + (i - 1) * w, g[(col, "mean")] * 100, w * 0.92, yerr=g[(col, "std")] * 100, color=c,
               label=lab, error_kw=dict(lw=0.6, capsize=1.2))
    ax.set_xticks(x, [o.replace(" (recalibrated)", "\n(recal.)").replace(" recalibrated", "\n(recal.)") for o in order],
                  rotation=35, ha="right")
    ax.set_ylabel("Rate under shift [%]"); ax.legend(frameon=False, ncol=3, loc="upper center",
                                                      bbox_to_anchor=(0.5, 1.12)); ax.grid(axis="x", visible=False)
    _save(fig, out, "fig_shift")


def fig_scenarios(out, SC):
    rows = []
    for s, g in SC.groupby("scenario"):
        sev = g["y"] >= 2
        if not sev.any():
            continue
        rows.append([s] + [(g.loc[sev, m] <= 1).mean() * 100 for m in ["Nominal physics", "HGB", "PSCL(HGB)"]])
    T = pd.DataFrame(rows, columns=["scenario", "Nominal physics", "HGB", "PSCL(HGB)"]).set_index("scenario")
    T = T.sort_values("HGB")
    fig, ax = plt.subplots(figsize=(SINGLE, 3.0))
    y = np.arange(len(T)); h = 0.27
    for i, (m, c) in enumerate([("Nominal physics", GRAY), ("HGB", BLUE), ("PSCL(HGB)", ORANGE)]):
        ax.barh(y + (i - 1) * h, T[m], h * 0.92, color=c, label=m)
    ax.set_yticks(y, T.index.str.replace("_", " ")); ax.set_xlabel("Severe-miss rate, P(decision ≤ Moderate | High/Critical) [%]")
    ax.set_title("Scenario families with High/Critical encounters", fontsize=7, color=MUTED)
    ax.legend(frameon=False, loc="lower right"); ax.grid(axis="y", visible=False)
    _save(fig, out, "fig_scenarios")


def fig_importance(out, imp, k=15):
    t = imp.head(k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.9))
    ax.barh(t["feature"], t["mean"], xerr=t["std"], color=BLUE, height=0.7, error_kw=dict(lw=0.6, capsize=1.2))
    ax.set_xlabel("Permutation importance (Δ macro-F1)"); ax.grid(axis="y", visible=False)
    _save(fig, out, "fig_importance")


def make_all(out, R, A, S, H, SC, extra):
    fig_network(out)
    fig_labels(out, extra["df"])
    fig_tradeoff(out, R)
    fig_confusion(out, extra["cm_hgb"], extra["cm_pscl"])
    fig_conformal(out, S)
    fig_shift(out, H)
    fig_scenarios(out, SC)
    fig_importance(out, extra["importance"])
