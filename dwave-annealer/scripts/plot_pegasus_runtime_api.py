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
    df = pd.read_csv(RESULTS_DIR / "pegasus_api_comparison.csv")
    styles = {
        "D-Wave qpu access time": ("red", "s", "-"),
        "D-Wave wall time": ("red", "^", "--"),
        "VeloxQ solver time": ("blue", "s", "-"),
        "VeloxQ wall time": ("blue", "^", "--"),
        "VeloxQ job time": ("blue", "D", ":"),
    }

    fig, ax = plt.subplots(1, 1, figsize=(4.7, 4.3))
    fontsize_label = 17
    fontsize_text = 15
    fontsize_ticks = 15
    for series, (color, marker, linestyle) in styles.items():
        part = df[df["series"] == series].sort_values("graph_size")
        kwargs = {
            "label": series,
            "marker": marker,
            "markersize": 5,
            "linestyle": linestyle,
            "linewidth": 1.2,
            "elinewidth": 0.8,
            "capsize": 2,
            "color": color,
        }
        if series == "VeloxQ job time":
            kwargs["markerfacecolor"] = "white"
            kwargs["markeredgewidth"] = 1.2
        ax.errorbar(part["num_var"], part["runtime_mean"], yerr=part["runtime_std"], **kwargs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    tick_source = df.drop_duplicates("graph_size").sort_values("graph_size")
    ax.set_xticks(list(tick_source["num_var"]))
    ax.set_xticklabels(
        [
            rf"${int(x)}$" if int(x) in {2, 3, 4, 5, 6, 8, 10, 13, 16} else ""
            for x in tick_source["graph_size"]
        ],
        fontsize=fontsize_ticks,
    )
    ax.set_xticks([], minor=True)
    ax.set_xlabel(r"Pegasus graph size parameter $m$", fontsize=fontsize_label)
    ax.set_ylabel(r"Runtime $[s]$", fontsize=fontsize_label)
    ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    ax.set_ylim(4e-2, 15)
    ax.legend(loc="upper left", fontsize=fontsize_text - 6, frameon=False, ncol=2)
    plt.tight_layout()
    out_pdf = Path(__file__).with_name("pegasus_runtime_api.pdf")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
