import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator


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


def load_cache(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(float(r["num_var"])))

    def arr(col):
        return np.array([float(r[col]) for r in rows], dtype=float)

    return {
        "num_var": arr("num_var"),
        "autotune_gap_mean": arr("autotune_gap_mean"),
        "custom1_gap_mean": arr("custom1_gap_mean"),
        "custom2_gap_mean": arr("custom2_gap_mean"),
        "sbm_gap_mean": arr("sbm_gap_mean"),
        "pa_gap_mean": arr("pa_gap_mean"),
        "autotune_runtime_mean": arr("autotune_runtime_mean"),
        "custom1_runtime_mean": arr("custom1_runtime_mean"),
        "custom2_runtime_mean": arr("custom2_runtime_mean"),
        "sbm_runtime_mean": arr("sbm_runtime_mean"),
        "pa_runtime_mean": arr("pa_runtime_mean"),
    }


latex_plot()
fig, axs = plt.subplots(1, 2, figsize=(8, 4))

cache_path = (
    Path(__file__).resolve().parents[1] / "results" / "planted_3r3x.csv"
)
df = load_cache(cache_path)

colors = ["blue", "green", "red", "purple", "orange", "brown"]
fontsize_label = 17
fontsize_text = 17
fontsize_ticks = 17
fontsize_legend = 13

ax1 = axs[0]
ax1.axhline(0, color="black", linewidth=1.0)
ax1.plot(df["num_var"], df["custom1_gap_mean"], label="Custom VeloxQ 1", marker="s", markersize=5, linestyle=":", linewidth=1.0, color=colors[1])
ax1.plot(df["num_var"], df["custom2_gap_mean"], label="Custom VeloxQ 2", marker="D", markersize=4, linestyle=":", linewidth=1.0, color=colors[4])
ax1.plot(df["num_var"], df["autotune_gap_mean"], label="AutoTune VeloxQ", marker="o", markersize=4, linestyle=":", linewidth=1.0, color=colors[0])
ax1.plot(df["num_var"], df["sbm_gap_mean"], label="SBM", marker=">", markersize=5, linestyle=":", linewidth=1.0, color=colors[2])
ax1.plot(df["num_var"], df["pa_gap_mean"], label="PA", marker="v", markersize=5, linestyle=":", linewidth=1.0, color=colors[3])
ax1.text(0.05, 0.5, "a)", transform=ax1.transAxes, fontsize=fontsize_text + 8)
ax1.set_xscale("log")
xticks = [10**i for i in range(1, 6)]
ax1.set_xticks(xticks)
ax1.set_xticklabels([f"$10^{{{i}}}$" for i in range(1, 6)])
ax1.xaxis.set_minor_locator(LogLocator(numticks=999, subs="auto"))
ax1.set_xlabel("QUBO variables", fontsize=fontsize_label)
ax1.set_yticks([0, 5, 10, 15])
ax1.set_yticks([2.5, 7.5, 12.5], minor=True)
ax1.set_ylabel(r"Optimality gap $g$ [\%]", fontsize=fontsize_label)
ax1.tick_params(axis="both", which="both", labelsize=fontsize_ticks)

ax2 = axs[1]
ax2.plot(df["num_var"], df["autotune_runtime_mean"], label="AutoTune VeloxQ", marker="o", markersize=5, linestyle=":", linewidth=1.0, color=colors[0])
ax2.plot(df["num_var"], df["custom1_runtime_mean"], label="Custom VeloxQ 1", marker="s", markersize=5, linestyle=":", linewidth=1.0, color=colors[1])
ax2.plot(df["num_var"], df["custom2_runtime_mean"], label="Custom VeloxQ 2", marker="D", markersize=4, linestyle=":", linewidth=1.0, color=colors[4])
ax2.plot(df["num_var"], df["sbm_runtime_mean"], label="SBM", marker=">", markersize=5, linestyle=":", linewidth=1.0, color=colors[2])
ax2.plot(df["num_var"], df["pa_runtime_mean"], label="PA", marker="v", markersize=5, linestyle=":", linewidth=1.0, color=colors[3])
ax2.text(0.05, 0.4, "b)", transform=ax2.transAxes, fontsize=fontsize_text + 8)
ax2.legend(loc="upper left", fontsize=fontsize_legend, frameon=False)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_ylabel("Runtime [s]", fontsize=fontsize_label)
ax2.set_xlabel("QUBO variables", fontsize=fontsize_label)
ax2.set_xticks(xticks)
ax2.set_xticklabels([f"$10^{{{i}}}$" for i in range(1, 6)])
ax2.xaxis.set_minor_locator(LogLocator(numticks=999, subs="auto"))
ax2.tick_params(axis="both", which="both", labelsize=fontsize_ticks)

plt.tight_layout()
plt.savefig("3R3X.pdf", bbox_inches="tight")
