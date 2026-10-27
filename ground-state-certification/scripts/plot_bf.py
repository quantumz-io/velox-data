import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from pathlib import Path


def latex_plot(scale=1, fontsize=12):
    """Changes the size of a figure and fonts for the publication-quality plots."""
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


latex_plot()
fig, axs = plt.subplots(1, 1, figsize=(8, 5))

cache_path = (
    Path(__file__).resolve().parents[3]
    / "velox-data"
    / "ground-state-certification"
    / "results"
    / "bf_all-to-all.csv"
)
df_cache = pd.read_csv(cache_path, sep=",").sort_values("size")

colors = ["blue", "green", "red", "purple", "orange", "black"]

fontsize_label = 17
fontsize_text = 17
fontsize_ticks = 17

# ====================== runtimes ======================================
ax3 = axs
ax3.plot(
    df_cache["size"],
    df_cache["runtime_bf"],
    label="Brute Force",
    marker="^",
    markersize=8,
    linestyle=":",
    linewidth=1.0,
    color=colors[2],
)
ax3.plot(
    df_cache["size"],
    df_cache["velox_autotune_runtime_mean"],
    label="AutoTune VeloxQ",
    marker="o",
    markersize=8,
    linestyle=":",
    linewidth=1.0,
    color=colors[0],
)
ax3.plot(
    df_cache["size"],
    df_cache["velox_custom_runtime_mean"],
    label="Custom VeloxQ",
    marker="s",
    markersize=8,
    linestyle=":",
    linewidth=1.0,
    color=colors[1],
)
ax3.plot(
    df_cache["size"],
    df_cache["cplex_runtime_mean"],
    label="CPLEX",
    marker="d",
    markersize=8,
    linestyle=":",
    linewidth=1.0,
    color=colors[3],
)
ax3.legend(loc="center right", fontsize=13, frameon=False, ncol=1, bbox_to_anchor=(1.0, 0.57))
ax3.set_yscale("log")
ax3.set_ylabel("Runtime [s]", fontsize=fontsize_label)
ax3.set_xlabel("QUBO variables", fontsize=fontsize_label)

xxticks = df_cache["size"].values
ax3.set_xticks(xxticks[::2], minor=False)
ax3.set_xticks(xxticks, minor=True)

ax3.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
ax3.set_ylim(top=1e6)

ax3.axhline(y=60, color="black", linestyle="--", linewidth=1.0)
ax3.axhline(y=3600, color="black", linestyle="--", linewidth=1.0)
ax3.axhline(y=86400, color="black", linestyle="--", linewidth=1.0)

ax3.text(40, 1.5 * 60, "1 minute", fontsize=fontsize_text, transform=ax3.transData)
ax3.text(40, 1.5 * 3600, "1 hour", fontsize=fontsize_text, transform=ax3.transData)
ax3.text(40, 1.5 * 86400, "1 day", fontsize=fontsize_text, transform=ax3.transData)

plt.savefig("bf_runtime.pdf", bbox_inches="tight")
