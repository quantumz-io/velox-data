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

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
FIGURE_DIR = SCRIPT_DIR.parent / "plots"


fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs = axs.flatten()

df_plot = pd.read_csv(RESULTS_DIR / "random3body_plot.csv")
df_autotune_stats = df_plot[df_plot["solver_variant"] == "AutoTune VeloxQ"].copy()
df_custom_stats = df_plot[df_plot["solver_variant"] == "Custom VeloxQ"].copy()

df_autotune_stats.rename(
    columns={
        "runtime_velox_mean": "avg_velox_runtime",
        "std_velox_runtime": "std_velox_runtime",
        "runtime_sa_mean": "avg_sa_runtime",
        "runtime_sa_std": "std_sa_runtime",
        "runtime_cplex_mean": "avg_cplex_runtime",
        "runtime_cplex_std": "std_cplex_runtime",
        "gap_sa_mean": "avg_sa_gap",
        "gap_sa_std": "std_sa_gap",
        "gap_cplex_mean": "avg_cplex_gap",
        "gap_cplex_std": "std_cplex_gap",
    },
    inplace=True,
)
df_custom_stats.rename(
    columns={
        "runtime_velox_mean": "avg_velox_runtime",
        "std_velox_runtime": "std_velox_runtime",
        "runtime_sa_mean": "avg_sa_runtime",
        "runtime_sa_std": "std_sa_runtime",
        "runtime_cplex_mean": "avg_cplex_runtime",
        "runtime_cplex_std": "std_cplex_runtime",
        "gap_sa_mean": "avg_sa_gap",
        "gap_sa_std": "std_sa_gap",
        "gap_cplex_mean": "avg_cplex_gap",
        "gap_cplex_std": "std_cplex_gap",
    },
    inplace=True,
)

df_autotune_stats = df_autotune_stats.sort_values("size").reset_index(drop=True)
df_custom_stats = df_custom_stats.sort_values("size").reset_index(drop=True)

# for cplex_gap, replace negative with NaN, and also replace the runtime with NaN
# if the gap is negative
df_autotune_stats["avg_cplex_gap"] = df_autotune_stats["avg_cplex_gap"].apply(
    lambda x: np.nan if x < 0 else x
)
df_custom_stats["avg_cplex_gap"] = df_custom_stats["avg_cplex_gap"].apply(
    lambda x: np.nan if x < 0 else x
)

df_autotune_stats["avg_cplex_runtime"] = df_autotune_stats.apply(
    lambda x: np.nan if np.isnan(x["avg_cplex_gap"]) else x["avg_cplex_runtime"], axis=1
)
df_custom_stats["avg_cplex_runtime"] = df_custom_stats.apply(
    lambda x: np.nan if np.isnan(x["avg_cplex_gap"]) else x["avg_cplex_runtime"], axis=1
)

print(df_autotune_stats)
print(df_custom_stats)
df_autotune_stats["size"] = (df_autotune_stats["size"] + 2) / 2
df_custom_stats["size"] = (df_custom_stats["size"] + 2) / 2

colors = ["blue", "green", "red", "purple", "orange", "brown"]

fontsize_label = 17
fontsize_text = 17
fontsize_ticks = 17
fontsize_legend = 13

# # ====================== panel a)  ======================
ax1 = axs[0]
ax1.axhline(0, color="black", linewidth=1.0)
ax1.errorbar(
    df_autotune_stats["size"],
    df_autotune_stats["avg_sa_gap"],
    yerr=df_autotune_stats["std_sa_gap"],
    elinewidth=0.5,
    capsize=2,
    marker="o",
    markersize=5,
    linestyle=":",
    linewidth=1.0,
    color=colors[0],
)
ax1.errorbar(
    df_custom_stats["size"],
    df_custom_stats["avg_sa_gap"],
    yerr=df_custom_stats["std_sa_gap"],
    elinewidth=0.5,
    capsize=2,
    marker="o",
    markersize=5,
    linestyle=":",
    linewidth=1.0,
    color=colors[1],
)

ax1.errorbar(
    df_autotune_stats["size"],
    df_autotune_stats["avg_cplex_gap"],
    yerr=df_autotune_stats["std_cplex_gap"],
    elinewidth=0.5,
    capsize=2,
    marker="s",
    markersize=5,
    linestyle=":",
    linewidth=1.0,
    color=colors[0],
)

ax1.errorbar(
    df_custom_stats["size"],
    df_custom_stats["avg_cplex_gap"],
    yerr=df_custom_stats["std_cplex_gap"],
    elinewidth=0.5,
    capsize=2,
    marker="s",
    markersize=5,
    linestyle=":",
    linewidth=1.0,
    color=colors[1],
)

# custom legend for the gap
ax1.plot([], [], color="black", marker="o", linestyle="", label="Ref. solver: SA")
ax1.plot([], [], color="black", marker="s", linestyle="", label="Ref. solver: CPLEX")

# custom legend with colored patches
import matplotlib.patches as mpatches

old_handles, old_labels = ax1.get_legend_handles_labels()
legend1 = ax1.legend(
    handles=[
        old_handles[0],
        old_handles[1],
        mpatches.Patch(color=colors[0]),
        mpatches.Patch(color=colors[1]),
    ],
    labels=[
        old_labels[0],
        old_labels[1],
        "AutoTune VeloxQ",
        "Custom VeloxQ",
    ],
    loc="lower left",
    fontsize=fontsize_legend-3,
    frameon=False,
    ncols=1,
    bbox_to_anchor=(0.1, 0.0)
)

ax1.text(0.85, 0.85, "a)", transform=ax1.transAxes, fontsize=fontsize_text+8)
ax1.tick_params(axis="both", which="major", labelsize=fontsize_ticks)

# Two hardcoded red KIPU points
ax1.scatter([433], [0.019 * 100], color="r", marker="D", s=50, label="", zorder=10)
ax1.scatter([156], [0.037 * 100], color="r", marker="^", s=50, label="")

ax1.set_xscale("log")
xxticks = [
    10**i for i in range(1, int(np.ceil(np.log10(df_autotune_stats["size"].max())) + 1))
]
xxtickslabels = [
    f"$10^{i}$"
    for i in range(1, int(np.ceil(np.log10(df_autotune_stats["size"].max())) + 1))
]
ax1.set_xticks(xxticks)
ax1.set_xticklabels(xxtickslabels, fontsize=fontsize_ticks)
ax1.set_xlabel("HUBO variables", fontsize=fontsize_label)
ax1.set_xlim(50, 2*10**8)

ax1.set_ylim(-0.1, 4)
ax1.set_yticks([0, 1, 2, 3, 4])
ax1.minorticks_on()
ax1.set_ylabel(r"Reference gap $g$ [\%]", fontsize=fontsize_label)

ax1.axvline(7*10**6, color='k', linestyle="--")

# # ====================== panel c) runtimes for panel a) ======================================
ax3 = axs[1]
ax3.plot(
    df_autotune_stats["size"],
    df_autotune_stats["avg_velox_runtime"],
    label="AutoTune VeloxQ",
    marker=">",
    markersize=5,
    linestyle=":",
    linewidth=1.0,
    color=colors[0],
)
ax3.plot(
    df_custom_stats["size"],
    df_custom_stats["avg_velox_runtime"],
    label="Custom VeloxQ",
    marker="<",
    markersize=5,
    linestyle=":",
    linewidth=1.0,
    color=colors[1],
)
ax3.plot(
    df_autotune_stats["size"],
    df_autotune_stats["avg_cplex_runtime"],
    label="CPLEX",
    marker="s",
    markersize=5,
    linestyle=":",
    linewidth=1.0,
    color=colors[4],
)
ax3.plot(
    df_autotune_stats["size"],
    df_autotune_stats["avg_sa_runtime"],
    label="SA",
    marker="o",
    markersize=5,
    linestyle=":",
    linewidth=1.0,
    color=colors[3],
)
ax3.text(0.85, 0.1, "b)", transform=ax3.transAxes, fontsize=fontsize_text+8)

ax3.legend(loc="upper left", fontsize=12, frameon=False)
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_ylabel("Runtime [s]", fontsize=fontsize_label)
ax3.set_xlabel("HUBO variables", fontsize=fontsize_label)
ax3.set_xticks(xxticks)
ax3.set_xticklabels(xxtickslabels, fontsize=fontsize_ticks)
ax3.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
ax3.set_xlim(50, 2*10**8)

miny = int(np.floor(np.log10(df_autotune_stats["avg_velox_runtime"].min())))
maxy = int(np.ceil(np.log10(df_autotune_stats["avg_sa_runtime"].max())))
yyticks = [10**i for i in range(miny, maxy + 1)]
yytickslabels = [f"$10^{{{i}}}$" for i in range(miny, maxy + 1)]
ax3.set_yticks(yyticks)
ax3.set_yticklabels(yytickslabels, fontsize=fontsize_ticks)
ax3.axvline(7*10**6, color='k', linestyle="--")

plt.tight_layout()
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURE_DIR / "random3Body.pdf", bbox_inches="tight")
