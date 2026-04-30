from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
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


def main() -> None:
    latex_plot()
    df = pd.read_csv(RESULTS_DIR / "pegasus_embedded.csv")
    colors = {"AutoTune VeloxQ": "blue", "Custom VeloxQ": "green", "D-Wave Advantage6.4": "red"}
    markers = {"AutoTune VeloxQ": "o", "Custom VeloxQ": "s", "D-Wave Advantage6.4": ">"}

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    ax1, ax2 = axs
    fontsize_label = 17
    fontsize_text = 15
    fontsize_ticks = 17
    marker_size = 7

    ax1.axhline(0, color="black", linewidth=1.0)
    for solver in ["AutoTune VeloxQ", "Custom VeloxQ", "D-Wave Advantage6.4"]:
        part = df[df["solver"] == solver].sort_values("clique")
        ax1.errorbar(part["clique"], part["gap_mean"], yerr=part["gap_std"], label=solver, marker=markers[solver], linestyle=":", linewidth=1.0, markersize=marker_size, capsize=2, elinewidth=0.5, color=colors[solver])
    ax1.text(0.88, 0.9, "a)", transform=ax1.transAxes, fontsize=fontsize_text + 8)
    ax1.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    ax1.set_ylabel(r"Reference gap $g$ [\%]", fontsize=fontsize_label)
    ax1.set_xlabel(r"Complete graph vertices $m$", fontsize=fontsize_label)
    x_ticks = [20, 40, 60, 80, 100, 120, 140, 160]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([rf"${x}$" for x in x_ticks], fontsize=fontsize_ticks - 3)
    ax1.set_xlim(15, 190)

    dwave = df[df["solver"] == "D-Wave Advantage6.4"].sort_values("clique")
    inset = ax1.inset_axes([0.25, 0.6, 0.37, 0.35])
    inset.plot(dwave["clique"], dwave["num_qubits_mean"], marker="o", linestyle=":", color="black", markersize=3, linewidth=1.0)
    inset.set_xscale("log")
    inset.set_yscale("log")
    inset.set_ylabel(r"\#qubits", fontsize=fontsize_label - 3)
    inset.set_xticks(dwave["clique"].values)
    inset.set_xticklabels([r"${20}$", r"${40}$", r"", r"${80}$", r"", r"", r"", r"${160}$"], fontsize=fontsize_ticks - 3)
    inset.set_xticks([], minor=True)
    inset.set_ylim(10, 10000)
    inset.set_yticks([10, 100, 1000, 10000])
    inset.set_yticklabels([r"$10^1$", r"$10^2$", r"$10^3$", r"$10^4$"], fontsize=fontsize_ticks - 3)

    for solver in ["AutoTune VeloxQ", "Custom VeloxQ"]:
        part = df[df["solver"] == solver].sort_values("clique")
        ax2.errorbar(part["clique"], part["runtime_mean"], yerr=part["runtime_std"], label=solver, marker=markers[solver], linestyle=":", linewidth=1.0, markersize=marker_size, capsize=2, color=colors[solver])
    ax2.errorbar(dwave["clique"] * 1.015, dwave["runtime_mean"], yerr=dwave["runtime_std"], label="D-Wave Adv. 6.4", marker=">", linestyle=":", linewidth=1.0, markersize=marker_size, capsize=2, color="red", zorder=2)
    ax2.errorbar(dwave["clique"], dwave["runtime_qpu_mean"], yerr=dwave["runtime_qpu_std"], label="D-Wave Adv. 6.4\n(excl. embedding)", marker=">", markerfacecolor="none", linestyle=":", linewidth=1.0, markersize=marker_size, capsize=2, color="red")
    ax2.text(0.88, 0.9, "b)", transform=ax2.transAxes, fontsize=fontsize_text + 8)
    ax2.legend(loc="lower left", fontsize=fontsize_text - 5, frameon=False, ncol=1, bbox_to_anchor=(0.02, 0.03))
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([r"$20$", r"$40$", r"$60$", r"$80$", r"", r"$120$", r"", r"$160$"], fontsize=fontsize_ticks)
    ax2.set_xticks([], minor=True)
    ax2.set_xlabel(r"Complete graph vertices $m$", fontsize=fontsize_label)
    ax2.set_ylabel("Runtime [s]", fontsize=fontsize_label)
    ax2.tick_params(axis="both", which="major", labelsize=fontsize_ticks)

    plt.tight_layout()
    out_pdf = Path(__file__).with_name("pegasus_embedded_v2.pdf")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
