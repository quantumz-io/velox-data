from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def latex_plot(scale: float = 1.0, fontsize: int = 12) -> None:
    fig_width_pt = 246.0
    inches_per_pt = 1.0 / 72.27
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0
    fig_width = fig_width_pt * inches_per_pt * scale
    fig_height = fig_width * golden_mean
    mpl.rcParams.update(
        {
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
            "figure.figsize": [fig_width, fig_height],
        }
    )


def draw_error_lines(ax, df: pd.DataFrame, value: str, solvers: list[str], labels: bool = True) -> None:
    markers = {"AutoTune VeloxQ": "o", "Custom VeloxQ": "s", "D-Wave Adv2 1.12": ">", "D-Wave Kerberos": ">", "Simulated Annealing": "D"}
    colors = {"AutoTune VeloxQ": "blue", "Custom VeloxQ": "green", "D-Wave Adv2 1.12": "red", "D-Wave Kerberos": "red", "Simulated Annealing": "orange"}
    for solver in solvers:
        part = df[df["solver"] == solver].sort_values("graph_size")
        if part.empty:
            continue
        ax.errorbar(part["num_var"], part[f"{value}_mean"], yerr=part[f"{value}_std"], label=solver if labels else None, marker=markers.get(solver, "o"), linestyle=":", linewidth=1.0, markersize=5, capsize=2, elinewidth=0.5, color=colors.get(solver))


def main() -> None:
    latex_plot()
    df = pd.read_csv(RESULTS_DIR / "zephyr_native.csv")
    small = df[df["regime"] == "small"].copy()
    large = df[df["regime"] == "large"].copy()
    ultra = df[df["regime"] == "ultralarge"].copy()

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs = axs.flatten()
    colors = ["blue", "green", "red"]
    fontsize_label = 17
    fontsize_text = 15
    fontsize_ticks = 17

    ax = axs[0]
    ax.axhline(0, color="black", linewidth=1.0)
    draw_error_lines(ax, small, "gap", ["AutoTune VeloxQ", "Custom VeloxQ", "D-Wave Adv2 1.12"])
    ax.text(0.05, 0.75, "a)", transform=ax.transAxes, fontsize=fontsize_text + 8)
    ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    ax.set_ylim(-0.005, 0.1)
    ax.set_ylabel(r"Reference gap $g$ [\%]", fontsize=fontsize_label)
    ax1_ticks = list(small[small["solver"] == "AutoTune VeloxQ"]["num_var"])
    ax.set_xticks(ax1_ticks)
    xticklabels = [r"$1$", r"", r"", r"$4$", r"", r"$6$", r"$7$", r"$8$", r"$9$", r"${10}$", r"${11}$", r"${12}$"]
    ax.set_xticklabels((xticklabels + [""] * len(ax1_ticks))[: len(ax1_ticks)], fontsize=fontsize_ticks)

    inset_source = df[(df["solver"] == "AutoTune VeloxQ") & (df["regime"].isin(["small", "large"]))].drop_duplicates("graph_size")
    inset = ax.inset_axes([0.0, 0.7, 0.35, 0.3])
    inset.plot(inset_source["graph_size"], inset_source["num_var"], marker="o", linestyle=":", color="black", markersize=3)
    x = np.asarray(inset_source["graph_size"], dtype=float)
    inset.plot(x, 30 * x**2, color="red", linewidth=1.5)
    inset.text(0.05, 0.3, r"$\sim 30 m^2$", transform=inset.transAxes, fontsize=fontsize_text, color="red", rotation=43)
    inset.set_xscale("log")
    inset.set_yscale("log")
    inset.set_ylabel(r"\#qubits in $Z_{m}$", fontsize=fontsize_label - 6)
    inset.yaxis.tick_right()
    inset.yaxis.set_label_position("right")
    zephyr_index = np.asarray(inset_source["graph_size"], dtype=float)
    xticks_major = [1, 10, 150]
    xticks_minor = [item for item in zephyr_index if item not in xticks_major]
    inset.set_xticks(xticks_major, minor=False)
    inset.set_xticks(xticks_minor, minor=True)
    inset.set_xticklabels([rf"${item}$" for item in xticks_major], fontsize=fontsize_ticks - 3, minor=False)
    inset.set_xticklabels(["" for _ in xticks_minor], fontsize=fontsize_ticks - 3, minor=True)
    inset.set_ylim(10, 1e6)
    inset.set_yticks([10, 1000, 100000])
    inset.set_yticklabels([r"$10^1$", r"$10^3$", r"$10^5$"], fontsize=fontsize_ticks - 3)

    ax = axs[1]
    ax.axhline(0, color="black", linewidth=1.0)
    draw_error_lines(ax, large, "gap", ["AutoTune VeloxQ", "Custom VeloxQ", "D-Wave Kerberos"])
    ax.text(0.05, 0.75, "b)", transform=ax.transAxes, fontsize=fontsize_text + 8)
    ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    ax.set_xscale("log")
    ax.set_ylim(-0.01, 0.2)
    ax2_ticks = list(large[large["solver"] == "AutoTune VeloxQ"]["num_var"])
    ax.set_xticks(ax2_ticks)
    xticklabels = [r"$30$", r"$40$", r"$50$", r"$60$", r"", r"$80$", r"", r"", r"$110$", r"", r"", r"", r"${150}$"]
    ax.set_xticklabels((xticklabels + [""] * len(ax2_ticks))[: len(ax2_ticks)], fontsize=fontsize_ticks, minor=False)
    ax.set_xticks([], minor=True)
    ax.annotate("", xy=(0.05, 0.65), xytext=(0.95, 0.65), xycoords="axes fraction", textcoords="axes fraction", arrowprops=dict(arrowstyle="<-", color="black", lw=1.5))
    ax.text(0.15, 0.7, r"$\sim 10^3 \times$ larger than $Z_{15}$", transform=ax.transAxes, fontsize=fontsize_text)

    ax = axs[2]
    for solver, marker, color, label in [
        ("AutoTune VeloxQ", "o", colors[0], "AutoTune VeloxQ"),
        ("Custom VeloxQ", "s", colors[1], "Custom VeloxQ"),
        ("D-Wave Adv2 1.12", ">", colors[2], "D-Wave Adv2 1.12"),
    ]:
        part = small[small["solver"] == solver].sort_values("graph_size")
        ax.plot(
            part["num_var"],
            part["runtime_mean"],
            label=label,
            marker=marker,
            markersize=5,
            linestyle=":",
            linewidth=1.0,
            color=color,
        )
    ax.text(0.05, 0.6, "c)", transform=ax.transAxes, fontsize=fontsize_text + 8)
    ax.legend(loc="upper left", fontsize=fontsize_text - 2, frameon=False)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax3_ticks = list(small[small["solver"] == "AutoTune VeloxQ"]["num_var"])
    ax.set_xticks(ax3_ticks, minor=False)
    xticklabels = [r"$1$", r"$2$", r"$3$", r"$4$", r"$5$", r"$6$", r"$7$", r"$8$", r"", r"${10}$", r"", r"$12$"]
    ax.set_xticklabels((xticklabels + [""] * len(ax3_ticks))[: len(ax3_ticks)], fontsize=fontsize_ticks, minor=False)
    ax.set_xticks([], minor=True)
    ax.set_xlabel(r"Zephyr graph size parameter $m$", fontsize=fontsize_label)
    ax.set_ylabel(r"Runtime $[s]$", fontsize=fontsize_label)
    ax.set_ylim(1e-2, 1e1)
    ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)

    ax = axs[3]
    for solver, marker, color, label in [
        ("AutoTune VeloxQ", "o", colors[0], None),
        ("Custom VeloxQ", "s", colors[1], None),
        ("D-Wave Kerberos", ">", colors[2], "D-Wave Kerberos"),
        ("Simulated Annealing", "D", "orange", "Simulated Annealing"),
    ]:
        part = large[large["solver"] == solver].sort_values("graph_size")
        ax.plot(
            part["num_var"],
            part["runtime_mean"],
            label=label,
            marker=marker,
            markersize=5,
            linestyle=":",
            linewidth=1.0,
            color=color,
        )
    ax.text(0.9, 0.55, "d)", transform=ax.transAxes, fontsize=fontsize_text + 8)
    ax.legend(loc="center left", fontsize=fontsize_text - 2, frameon=False, bbox_to_anchor=(0.0, 0.52))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax4_ticks = list(large[large["solver"] == "AutoTune VeloxQ"]["num_var"])
    ax.set_xticks(ax4_ticks)
    xticklabels = [r"$30$", r"$40$", r"$50$", r"$60$", r"", r"$80$", r"", r"", r"$110$", r"", r"", r"", r"${150}$"]
    ax.set_xticklabels((xticklabels + [""] * len(ax4_ticks))[: len(ax4_ticks)], fontsize=fontsize_ticks, minor=False)
    ax.set_xticks([], minor=True)
    ax.set_ylim(1e-2, 1e12)
    ax.set_yticks([1e-2, 1e0, 1e2, 1e4, 1e6, 1e8, 1e10])
    ax.yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    ax.set_xlabel(r"Zephyr graph size parameter $m$", fontsize=fontsize_label)
    ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    if not ultra.empty:
        inset = ax.inset_axes([0.15, 0.7, 0.7, 0.3])
        inset.set_xticks([])
        inset.set_yticks([])
        for solver, marker, color in [("AutoTune VeloxQ", "o", "blue"), ("Custom VeloxQ", "s", "green")]:
            part = ultra[ultra["solver"] == solver].sort_values("num_var")
            inset.plot(part["num_var"], part["runtime_mean"], marker=marker, linestyle=":", color=color, linewidth=1.0, markersize=5)
        inset.set_xscale("log")
        inset.set_yscale("log")
        auto = ultra[ultra["solver"] == "AutoTune VeloxQ"].sort_values("num_var")
        inset.set_xticks(auto["num_var"])
        inset.set_xticklabels([r"$250$", r"$500$", r"", r"$1000$", r"", r"", r"$1750$"], fontsize=fontsize_ticks - 4, minor=False)
        inset.set_xticks([], minor=True)
        inset.set_yticks([1e0, 1e2, 1e4], minor=False)
        inset.set_yticklabels([r"$10^0$", r"$10^2$", r"$10^4$"], fontsize=fontsize_ticks - 4, minor=False)

    plt.tight_layout()
    out = Path(__file__).with_name("zephyr_native.pdf")
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
