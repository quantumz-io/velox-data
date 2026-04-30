from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


class MinorSymLogLocator(mticker.Locator):
    def __init__(self, linthresh: float, base: float = 10.0, linear_divisions: int = 5) -> None:
        self.linthresh = linthresh
        self.base = base
        self.linear_divisions = linear_divisions

    def __call__(self) -> list[float]:
        axis = self.axis
        if axis is None:
            return []
        vmin, vmax = axis.get_view_interval()
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        ticks: list[float] = []
        if vmin < self.linthresh and vmax > -self.linthresh:
            step = (2.0 * self.linthresh) / float(self.linear_divisions)
            start = np.floor(max(vmin, -self.linthresh) / step) * step
            stop = min(vmax, self.linthresh)
            for tick in np.arange(start, stop + step, step):
                value = float(tick)
                if abs(value) > 1e-15 and abs(value) < self.linthresh:
                    ticks.append(value)
        if vmax > self.linthresh:
            pos_min = max(vmin, self.linthresh)
            for exponent in range(int(np.floor(np.log10(pos_min))), int(np.ceil(np.log10(vmax))) + 1):
                decade = self.base**exponent
                for sub in range(2, 10):
                    tick = float(sub * decade)
                    if pos_min <= tick <= vmax:
                        ticks.append(tick)
        if vmin < -self.linthresh:
            neg_max = min(vmax, -self.linthresh)
            for exponent in range(int(np.floor(np.log10(abs(neg_max)))), int(np.ceil(np.log10(abs(vmin)))) + 1):
                decade = self.base**exponent
                for sub in range(2, 10):
                    tick = float(-sub * decade)
                    if vmin <= tick <= neg_max:
                        ticks.append(tick)
        return list(self.raise_if_exceeds(sorted(set(ticks))))


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
    latex_plot(scale=2)
    df = pd.read_csv(RESULTS_DIR / "zephyr_velox_vs_cplex.csv")
    runtime = df.pivot(index="graph_size", columns="solver", values="runtime_mean").sort_index()
    gap = df.pivot(index="graph_size", columns="solver", values="gap_mean").sort_index()
    plot_df = pd.DataFrame(
        {
            "zephyr": runtime.index,
            "avg_runtime_velox": runtime["VeloxQ AutoTune"],
            "avg_runtime_cplex": runtime["CPLEX"],
            "avg_gap_velox": gap["VeloxQ AutoTune"],
            "avg_gap_cplex": gap["CPLEX"],
        }
    ).reset_index(drop=True)
    x = np.arange(len(plot_df)) * 3.0
    h = 0.9

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7, 3.6), sharey=True)
    colors = ["blue", "red"]
    fontsize_label = 17
    fontsize_text = 15
    fontsize_ticks = 14
    ax_left.barh(x - h / 2, plot_df["avg_runtime_velox"], height=h, label="VeloxQ AutoTune", color=colors[0])
    ax_left.barh(x + h / 2, plot_df["avg_runtime_cplex"], height=h, label="CPLEX", color=colors[1])
    ax_left.set_xscale("log")
    ax_left.set_xlabel(r"Runtime $[s]$", fontsize=fontsize_label)
    ax_left.set_ylabel(r"Zephyr graph index $m$", fontsize=fontsize_label)
    ax_left.set_yticks(x)
    ax_left.set_yticklabels([rf"${m}$" for m in plot_df["zephyr"].astype(int)], fontsize=fontsize_ticks)
    ax_left.set_ylim(-1.0, x[-1] + 1.0)
    ax_left.xaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0,), numticks=12))
    ax_left.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=[i / 10.0 for i in range(2, 10)], numticks=100))
    ax_left.xaxis.set_minor_formatter(mticker.NullFormatter())
    for runtime_value, label in [(1.0, r"$1\,\mathrm{s}$"), (10.0, r"$10\,\mathrm{s}$"), (60.0, r"$1\,\mathrm{min}$"), (600.0, r"$10\,\mathrm{min}$")]:
        ax_left.axvline(runtime_value, color="0.35", linestyle="--", linewidth=0.8, zorder=0)
        ax_left.text(runtime_value * 0.6, 0.02, label, rotation=90, va="bottom", ha="center", transform=ax_left.get_xaxis_transform(), fontsize=fontsize_ticks - 1, color="0.25")
    ax_left.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    ax_left.tick_params(axis="x", which="major", length=6)
    ax_left.tick_params(axis="x", which="minor", length=3)
    ax_left.text(0.91, 0.05, "a)", transform=ax_left.transAxes, fontsize=fontsize_text + 5)

    ax_right.axvline(0, color="black", linewidth=1.0)
    ax_right.barh(x - h / 2, plot_df["avg_gap_velox"], height=h, label="VeloxQ AutoTune", color=colors[0])
    ax_right.barh(x + h / 2, plot_df["avg_gap_cplex"], height=h, label="CPLEX", color=colors[1])
    ax_right.set_xscale("symlog", linthresh=1e-2)
    ax_right.set_xlabel(r"Reference gap $g$ [\%]", fontsize=fontsize_label)
    ax_right.set_yticks(x)
    ax_right.tick_params(axis="y", labelleft=False, left=False)
    ax_right.set_ylim(-1.0, x[-1] + 1.0)
    ax_right.xaxis.set_major_locator(mticker.SymmetricalLogLocator(base=10.0, linthresh=1e-2, subs=(1.0,)))
    ax_right.xaxis.set_minor_locator(MinorSymLogLocator(linthresh=1e-2, base=10.0, linear_divisions=20))
    ax_right.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_right.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    ax_right.tick_params(axis="x", which="major", length=6)
    ax_right.tick_params(axis="x", which="minor", length=3)
    ax_right.text(0.91, 0.05, "b)", transform=ax_right.transAxes, fontsize=fontsize_text + 5)
    ax_right.legend(loc="lower right", frameon=False, fontsize=fontsize_text - 1, bbox_to_anchor=(0.95, -0.05))
    ax_right.text(0.5, -0.35, "(smaller is better)", transform=ax_right.transAxes, fontsize=10, ha="center", va="bottom")

    plt.tight_layout()
    out = Path(__file__).with_name("zephyr_velox_vs_cplex.pdf")
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
