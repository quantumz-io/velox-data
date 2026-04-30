import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def latex_plot(scale=1, fontsize=12):
    fig_width_pt = 246.0
    inches_per_pt = 1.0 / 72.27
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0
    fig_width = fig_width_pt * inches_per_pt * scale
    fig_height = fig_width * golden_mean
    fig_size = [fig_width, fig_height]
    eps_with_latex = {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": [],
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": fontsize,
        "font.size": fontsize,
        "legend.fontsize": fontsize,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.figsize": fig_size,
    }
    mpl.rcParams.update(eps_with_latex)


def load_rows(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["num_var"] = int(float(r["num_var"]))
        r["alpha"] = float(r["alpha"])
        for k in [
            "autotune_gap_mean", "custom1_gap_mean", "custom2_gap_mean", "sbm_gap_mean", "pa_gap_mean",
            "autotune_runtime_mean", "custom1_runtime_mean", "custom2_runtime_mean", "sbm_runtime_mean", "pa_runtime_mean",
        ]:
            r[k] = float(r[k])
    return rows


def series(rows, alpha, col):
    rr = sorted([r for r in rows if abs(r["alpha"] - alpha) < 1e-12], key=lambda r: r["num_var"])
    x = np.array([r["num_var"] for r in rr], dtype=float)
    y = np.array([r[col] for r in rr], dtype=float)
    return x, y


latex_plot()
fig, axd = plt.subplot_mosaic([["upper left", "right"], ["lower left", "right"]], figsize=(8, 4), gridspec_kw=dict(hspace=0))
axs = [item[1] for item in axd.items()]

cache_path = Path(__file__).resolve().parents[1] / "results" / "planted_wishart.csv"
rows = load_rows(cache_path)

colors = ["blue", "green", "red", "purple", "orange", "brown"]
fontsize_label = 17
fontsize_text = 17
fontsize_ticks = 17
fontsize_legend = 13

ax1 = axs[2]
ax2 = axs[0]
a1 = 0.20
a2 = 1.00
for a, ax, tag in [(a1, ax1, "b"), (a2, ax2, "a")]:
    ax.axhline(0, color="black", linewidth=0.5)
    x, y = series(rows, a, "custom1_gap_mean"); ax.plot(x, y, label="Custom VeloxQ 1", marker="s", markersize=5, linestyle=":", linewidth=1.0, color=colors[1])
    x, y = series(rows, a, "custom2_gap_mean"); ax.plot(x, y, label="Custom VeloxQ 2", marker="D", markersize=4, linestyle=":", linewidth=1.0, color=colors[4])
    x, y = series(rows, a, "autotune_gap_mean"); ax.plot(x, y, label="AutoTune VeloxQ", marker="o", markersize=4, linestyle=":", linewidth=1.0, color=colors[0])
    x, y = series(rows, a, "sbm_gap_mean"); ax.plot(x, y, label="SBM", marker=">", markersize=5, linestyle=":", linewidth=1.0, color=colors[2])
    x, y = series(rows, a, "pa_gap_mean"); ax.plot(x, y, label="PA", marker="v", markersize=5, linestyle=":", linewidth=1.0, color=colors[3])
    ax.text(0.05, 0.8, rf"{tag}) $\alpha = {a:.1f}$", transform=ax.transAxes, fontsize=fontsize_text + 4)

ax1.set_xscale("log")
ax2.set_xscale("log")
xticks = [1e2, 1e3, 5e3]
xticklabels = [r"$10^2$", r"$10^3$", r"$5\times 10^3$"]
ax1.set_xticks(xticks)
ax1.set_xticklabels(xticklabels)
ax2.set_xticks([])
ax1.set_xlabel("QUBO variables", fontsize=fontsize_label)

ax2.set_ylim(-2, 20)
ax2.set_yticks([0, 5, 10, 15, 20], minor=False)
ax2.set_yticks([2.5, 7.5, 12.5, 17.5], minor=True)
ax1.set_ylim(-0.5, 9)
ax1.set_yticks([0, 2, 4, 6, 8], minor=False)
ax1.set_yticks([1, 3, 5, 7], minor=True)
yl = ax2.set_ylabel(r"Optimality gap $g$ [\%]", fontsize=fontsize_label)
yl.set_position((-0.4, 0.0))
ax1.tick_params(axis="both", which="both", labelsize=fontsize_ticks)
ax2.tick_params(axis="both", which="both", labelsize=fontsize_ticks)

ax3 = axs[1]
x, y = series(rows, a1, "autotune_runtime_mean"); ax3.plot(x, y, label="AutoTune VeloxQ", marker="o", markersize=5, linestyle=":", linewidth=1.0, color=colors[0])
x, y = series(rows, a1, "custom1_runtime_mean"); ax3.plot(x, y, label="Custom VeloxQ 1", marker="s", markersize=5, linestyle=":", linewidth=1.0, color=colors[1])
x, y = series(rows, a1, "custom2_runtime_mean"); ax3.plot(x, y, label="Custom VeloxQ 2", marker="D", markersize=4, linestyle=":", linewidth=1.0, color=colors[4])
x, y = series(rows, a1, "sbm_runtime_mean"); ax3.plot(x, y, label="SBM", marker=">", markersize=5, linestyle=":", linewidth=1.0, color=colors[2])
x, y = series(rows, a1, "pa_runtime_mean"); ax3.plot(x, y, label="PA", marker="v", markersize=5, linestyle=":", linewidth=1.0, color=colors[3])
ax3.text(0.05, 0.4, "c)", transform=ax3.transAxes, fontsize=fontsize_text + 4)
ax3.legend(loc="upper left", fontsize=fontsize_legend, frameon=False)
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_ylabel("Runtime [s]", fontsize=fontsize_label)
ax3.set_xticks(xticks)
ax3.set_xticklabels(xticklabels)
ax3.set_xlabel("QUBO variables", fontsize=fontsize_label)
ax3.tick_params(axis="both", which="both", labelsize=fontsize_ticks)

plt.tight_layout()
plt.savefig("wishart_ensemble.pdf", bbox_inches="tight")
