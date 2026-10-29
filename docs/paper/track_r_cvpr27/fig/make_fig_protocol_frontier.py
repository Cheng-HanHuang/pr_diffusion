"""Fig. frontier: GT-free protocol v1 vs fixed best-of-K at equal compute (FAST-DIPS, FFHQ, 180 images x 64 runs).

Data: docs/track_r/protocol/dev_v1/fast64_frontier_n64.json (protocol v1 development output, planner c598176).
Usage: python3 make_fig_protocol_frontier.py <dev_v1 dir> <out.pdf>
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

d, out = sys.argv[1], sys.argv[2]
r = json.load(open(os.path.join(d, "fast64_frontier_n64.json")))
E0 = r["fixed_curves"]["E0"]; OO = r["fixed_curves"]["O_orbit"]
pts = r["frontier"]

BLUE, ORANGE = "#2a78d6", "#eb6834"  # validated pair (dataviz validator, light mode)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.linewidth": 0.6, "pdf.fonttype": 42})

fig, ax = plt.subplots(figsize=(3.35, 1.85), constrained_layout=True)
Ks = list(range(1, 65))
ax.plot(Ks, OO, color=MUTED, lw=0.9, ls=(0, (4, 2)), zorder=1)
ax.plot(Ks, E0, color=BLUE, lw=1.4, zorder=2)
mk = [2, 4, 8, 16, 32, 64]
ax.plot(mk, [E0[k - 1] for k in mk], ls="none", marker="o", ms=3.2, mfc=BLUE, mec="white", mew=0.6, zorder=3)
for k in (16, 64):
    ax.annotate(f"best-of-{k}", (k, E0[k - 1]), xytext=(0, -10), textcoords="offset points", ha="center",
                fontsize=6, color=MUTED)
main = sorted([p for p in pts if p["tau_mult"] == 1.0], key=lambda p: p["M"])
ax.plot([p["runs_mean"] for p in main], [p["success_all"] for p in main], color=ORANGE, lw=1.4, zorder=4)
ax.plot([p["runs_mean"] for p in main], [p["success_all"] for p in main], ls="none", marker="D", ms=3.4,
        mfc=ORANGE, mec="white", mew=0.6, zorder=5)
alt = [p for p in pts if p["tau_mult"] != 1.0]
ax.plot([p["runs_mean"] for p in alt], [p["success_all"] for p in alt], ls="none", marker="D", ms=2.8,
        mfc="white", mec=ORANGE, mew=0.8, zorder=5)
ax.set_xscale("log", base=2)
ax.set_xticks(mk); ax.set_xticklabels([str(k) for k in mk]); ax.minorticks_off()
ax.set_xlim(1.8, 72); ax.set_ylim(0.72, 0.975)
ax.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95])
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100*v:.0f}%"))
ax.set_xlabel("Mean runs per image")
ax.set_ylabel("Images with a GOOD output")
ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
h = [Line2D([], [], color=ORANGE, lw=1.4, marker="D", ms=3.4, mfc=ORANGE, mec="white"),
     Line2D([], [], ls="none", marker="D", ms=2.8, mfc="white", mec=ORANGE, mew=0.8),
     Line2D([], [], color=BLUE, lw=1.4, marker="o", ms=3.2, mfc=BLUE, mec="white"),
     Line2D([], [], color=MUTED, lw=0.9, ls=(0, (4, 2)))]
lab = [r"Protocol, $M=1,\dots,4$ (left to right)", r"Protocol, $\tau\times(1\pm0.003)$",
       r"Fixed best-of-$K$, residual pick ($E_0$)", r"Ground-truth orbit oracle"]
ax.legend(h, lab, loc="lower right", frameon=False, handlelength=1.8, borderaxespad=0.2)
fig.savefig(out)
print("wrote", out)
