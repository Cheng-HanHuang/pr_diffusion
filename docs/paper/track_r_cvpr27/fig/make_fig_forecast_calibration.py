"""Fig. calibration: registered forecasts against outcomes, five prospective tests (primary variant = race; S2 reference).

Sources (committed comparison JSONs; aggregates only):
  test 1  FAST-DIPS FFHQ vs v1      XP 07d5460  A8_FAST_V1_COMPARISON.json              (forecast_comparison rows)
  test 2  FAST-DIPS ImageNet vs v2/v2 E1, and
  test 5  FAST-DIPS ImageNet vs v4  XP ef1cf10  CP2_V2_V2E1_V4_FORECAST_COMPARISON.json  (keys 'v2', 'v2 E1', 'v4')
  test 3  SITCOM FFHQ vs v1         XP 8798462  XP_B16_V1_V3_PROSPECTIVE_COMPARISON.json (forecast_source == v1)
  test 4  DAPS FFHQ vs v1           XP 29ef1f5  XP_A13_F8_PROSPECTIVE_V1_V3.json         (version == v1)
E1 rows use the primary decision model (beta_class) where two are registered. Primary panels only, except test 5 (both panels).
Usage: python3 make_fig_forecast_calibration.py <dir with the four JSONs> <out.pdf>
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

d, out = sys.argv[1], sys.argv[2]
J = lambda f: json.load(open(os.path.join(d, f)))
pts = []  # (test, variant, policy, point, lo, hi, observed)
for r in J("test1_fast_ffhq_v1.json")["forecast_comparison"]:
    if r["panel"] == "primary":
        pts.append((1, r["model_variant"], r["policy"], r["point"], r["lo95"], r["hi95"], r["observed"]))
cp2 = J("test2_5_imagenet_cp2.json")
for key, test in (("v2", 2), ("v2 E1", 2), ("v4", 5)):
    for r in cp2[key]["comparisons"]:
        if r.get("decision_model") not in (None, "beta_class"):
            continue
        pts.append((test, r["model_variant"], r["policy"], r["forecast"], r["lo95"], r["hi95"], r["observed"]))
for r in J("test3_sitcom_b16.json")["forecast_comparisons"]:
    if r["forecast_source"] == "v1" and r["panel"] == "primary":
        pts.append((3, r["model_variant"], r["policy"], r["point"], r["lo95"], r["hi95"], r["observed"]))
for r in J("test4_daps_f8.json")["comparisons"]:
    if r["version"] == "v1":
        pts.append((4, r["model_variant"], r["policy"], r["point"], r["lo95"], r["hi95"], r["observed"]))

COL = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100", 5: "#e87ba4"}  # validated (dataviz), shapes as secondary encoding
MK = {1: "o", 2: "s", 3: "^", 4: "D", 5: "v"}
NAME = {1: "1 FAST-DIPS, FFHQ", 2: "2 FAST-DIPS, ImageNet", 3: "3 SITCOM, FFHQ", 4: "4 DAPS, FFHQ", 5: "5 FAST-DIPS, ImageNet (16 runs seen)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 2, figsize=(3.35, 2.15), constrained_layout=True, sharey=True)
panels = [(axes[0], ("E0", "E1_CLIP"), "Selected run ($E_0$, $E_1$)"), (axes[1], ("O_raw", "O_orbit"), "Oracles ($O_{\\mathrm{raw}}$, $O_{\\mathrm{orbit}}$)")]
lo_ax, hi_ax = 0.28, 1.01
for ax, pols, title in panels:
    ax.plot([lo_ax, hi_ax], [lo_ax, hi_ax], color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    for t, var, pol, p, lo, hi, obs in pts:
        if pol not in pols:
            continue
        if var == "race":
            ax.errorbar(p, obs, xerr=[[p - lo], [hi - p]], fmt=MK[t], ms=3.0, color=COL[t], mfc=COL[t], mec="white", mew=0.4,
                        elinewidth=0.5, capsize=0, alpha=0.9, zorder=3)
        elif var == "S2" and pols[0] == "E0":
            ax.plot(p, obs, MK[t], ms=3.0, mfc="none", mec=COL[t], mew=0.7, alpha=0.9, zorder=2)
    ax.set_xlim(lo_ax, hi_ax); ax.set_ylim(lo_ax, hi_ax)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0]); ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_title(title, fontsize=7, color=INK, pad=3)
    ax.set_xlabel("Registered forecast")
    ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("Observed")
axes[1].text(0.97, 0.33, "below the line:\ncoverage overestimated", ha="right", va="bottom", fontsize=6, color=MUTED)
h = [Line2D([], [], ls="none", marker=MK[t], ms=3.2, color=COL[t], mec="white", mew=0.4) for t in sorted(NAME)]
h += [Line2D([], [], ls="none", marker="o", ms=3.2, mfc="none", mec=MUTED, mew=0.7)]
labels = [NAME[t] for t in sorted(NAME)] + ["exchangeable ranks (S2), hollow"]
fig.legend(h, labels, loc="outside lower center", ncol=2, frameon=False, handletextpad=0.3, columnspacing=0.8)
fig.savefig(out)
print("wrote", out, "points:", len(pts))
