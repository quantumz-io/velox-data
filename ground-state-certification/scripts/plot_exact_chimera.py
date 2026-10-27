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
    / "chimera_exact_runtime.csv"
)
df = pd.read_csv(cache_path, sep=",").sort_values("num_var")

colors = ["blue", "green", "red", "purple", "orange", "black"]
fontsize_label = 17
fontsize_ticks = 17

ax3 = axs
ax3.plot(
    df["num_var"],
    df["beit_runtime_mean"],
    label="BEIT",
    marker="x",
    markersize=8,
    linestyle=":",
    linewidth=1.0,
    color=colors[2],
)
ax3.plot(
    df["num_var"],
    df["velox_autotune_runtime_mean"],
    label="AutoTune VeloxQ",
    marker="o",
    markersize=8,
    linestyle=":",
    linewidth=1.0,
    color=colors[0],
)
ax3.plot(
    df["num_var"],
    df["velox_custom_runtime_mean"],
    label="Custom VeloxQ",
    marker="s",
    markersize=8,
    linestyle=":",
    linewidth=1.0,
    color=colors[1],
)
ax3.plot(
    df["num_var"],
    df["ttn_cpu_energy_only_runtime_mean"],
    label="TTN on CPU",
    marker=">",
    markersize=8,
    linestyle=":",
    linewidth=1.0,
    color=colors[3],
)
ax3.plot(
    df["num_var"],
    df["ttn_cpu_state_runtime_mean"],
    label="",
    marker=">",
    markersize=8,
    linestyle=":",
    linewidth=1.0,
    color=colors[3],
    markerfacecolor="none",
)
ax3.plot(
    df["num_var"],
    df["ttn_gpu_energy_only_runtime_mean"],
    label="TTN on GPU",
    marker="^",
    markersize=8,
    linestyle=":",
    linewidth=1.0,
    color=colors[4],
)
ax3.plot(
    df["num_var"],
    df["ttn_gpu_state_runtime_mean"],
    label="",
    marker="^",
    markersize=8,
    linestyle=":",
    linewidth=1.0,
    color=colors[4],
    markerfacecolor="none",
)

ax3.legend(loc="upper right", fontsize=14, frameon=False, ncol=1)
ax3.set_yscale("log")
ax3.set_ylabel("Runtime [s]", fontsize=fontsize_label)
ax3.set_xlabel("QUBO variables", fontsize=fontsize_label)

xxticks = df["num_var"].values
ax3.set_xticks(xxticks[::2], minor=False)
ax3.set_xticks(xxticks, minor=True)

ax3.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
ax3.set_ylim(top=1e6)

plt.savefig("gs_certif_runtime.pdf", bbox_inches="tight")
