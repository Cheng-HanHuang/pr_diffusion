"""Fig. kcurves (FAST-DIPS panel): observed K-curves vs frozen TH1 v1 forecasts.

Data: the executor's audited aggregate return
docs/track_r/executors/xp/returns/XP_A8_FAST_V1_20261028T090445Z/A8_FAST_V1_COMPARISON.json
(XP branch codex/prdiff-track-r-xp-a8-fast-20261028 @ 07d5460). Aggregates only.
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

src, out = sys.argv[1], sys.argv[2]
rows = json.load(open(src))["forecast_comparison"]
T = {(r["cohort"], r["panel"], r["K"], r["policy"], r["model_variant"]): r for r in rows}

BLUE, ORANGE = "#2a78d6", "#eb6834"  # validated pair (dataviz validator, light mode)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7, "axes.titlesize": 7.5,
    "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
})

Ks = [8, 12, 16, 24, 32, 48, 64]
fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.95), constrained_layout=True)
panels = [("B22", "Cohort A (random, 100)"), ("DEV80", "Cohort B (hard-enriched, 80)")]
for ax, (coh, title) in zip(axes, panels):
    def col(policy, variant, field):
        return [T[(coh, "primary", k, policy, variant)][field] for k in Ks]
    # Frozen race forecast for E0: 95% predictive band + point line.
    ax.fill_between(Ks, col("E0", "race", "lo95"), col("E0", "race", "hi95"),
                    color=BLUE, alpha=0.16, linewidth=0, zorder=1)
    ax.plot(Ks, col("E0", "race", "point"), color=BLUE, lw=1.0, ls=(0, (4, 2)), zorder=2)
    # Frozen S2 reference for E0: point only.
    ax.plot(Ks, col("E0", "S2", "point"), color=BLUE, lw=1.0, ls=(0, (1, 1.6)), zorder=2)
    # Frozen O_orbit forecast (identical across variants): point only.
    ax.plot(Ks, col("O_orbit", "race", "point"), color=ORANGE, lw=1.0, ls=(0, (4, 2)), zorder=2)
    # Observed (exact all-subset estimates on the realized 64-run panel).
    ax.plot(Ks, col("O_orbit", "race", "observed"), color=ORANGE, lw=1.3, marker="s", ms=3.4,
            mfc="white", mec=ORANGE, mew=1.0, zorder=3)
    ax.plot(Ks, col("E0", "race", "observed"), color=BLUE, lw=1.3, marker="o", ms=3.4,
            mfc=BLUE, mec="white", mew=0.6, zorder=4)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(k) for k in Ks])
    ax.minorticks_off()
    ax.set_xlim(7.3, 70)
    ax.set_title(title, color=INK, pad=3)
    ax.set_xlabel("$K$ (runs per image)")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylim(0.90, 1.003)
axes[0].set_yticks([0.90, 0.92, 0.94, 0.96, 0.98, 1.00])
axes[1].set_ylim(0.74, 1.006)
axes[1].set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
axes[0].set_ylabel("Share of images with a GOOD pick")
for ax in axes:
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100*v:.0f}%"))

handles = [
    Line2D([], [], color=BLUE, lw=1.3, marker="o", ms=3.4, mfc=BLUE, mec="white", label=r"$E_0$ observed"),
    Line2D([], [], color=ORANGE, lw=1.3, marker="s", ms=3.4, mfc="white", mec=ORANGE, label=r"$O_{\mathrm{orbit}}$ observed"),
    (Patch(facecolor=BLUE, alpha=0.16, lw=0), Line2D([], [], color=BLUE, lw=1.0, ls=(0, (4, 2)))),
    Line2D([], [], color=ORANGE, lw=1.0, ls=(0, (4, 2)), label=r"$O_{\mathrm{orbit}}$ frozen forecast"),
    Line2D([], [], color=BLUE, lw=1.0, ls=(0, (1, 1.6)), label=r"$E_0$ frozen S2 reference"),
]
labels = [r"$E_0$ observed", r"$O_{\mathrm{orbit}}$ observed", r"$E_0$ frozen race (95% band)",
          r"$O_{\mathrm{orbit}}$ frozen forecast", r"$E_0$ frozen S2 reference"]
fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False,
           handlelength=2.4, columnspacing=1.2, handletextpad=0.5, labelspacing=0.3)
fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight", pad_inches=0.01)
print("wrote", out)
