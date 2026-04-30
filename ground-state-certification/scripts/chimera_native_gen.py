#!/usr/bin/env python3
"""Generate perfect-Chimera random Ising instances in Velox-compatible HDF5.

- Samples integer QUBO coefficients in [-31, 31] for diagonal and native edges.
- Converts QUBO -> Ising with dimod.
- Saves HDF5 with Ising/biases and Ising/J_coo datasets.
- No live QPU calls, no embedding logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dimod
import dwave_networkx as dnx
import h5py
import numpy as np

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "instances" / "chimera"


def parse_list(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    vals = [int(p) for p in parts]
    for v in vals:
        if v < 2:
            raise ValueError("chimera cols must be >= 2")
    return vals


def build_qubo(graph, rng: np.random.Generator) -> dict[tuple[int, int], float]:
    q = {}
    for node in graph.nodes:
        q[(node, node)] = float(rng.integers(-31, 32))
    for u, v in graph.edges:
        key = (u, v) if u < v else (v, u)
        q[key] = float(rng.integers(-31, 32))
    return q


def qubo_to_ising(qubo):
    h_raw, j_raw, _ = dimod.qubo_to_ising(qubo)
    h = {int(k): float(v) for k, v in h_raw.items()}
    j = {}
    for (u, v), val in j_raw.items():
        a, b = (int(u), int(v)) if u < v else (int(v), int(u))
        j[(a, b)] = float(val)
    return h, j


def regular_index(nodes: list[int]) -> dict[int, int]:
    return {node: i + 1 for i, node in enumerate(sorted(nodes))}


def to_coo(nodes, j_terms, idx):
    rows = []
    cols = []
    vals = []

    for n in nodes:
        i = idx[n]
        rows.append(i)
        cols.append(i)
        vals.append(0.0)

    for (u, v), val in j_terms.items():
        iu, iv = idx[u], idx[v]
        rows.extend([iu, iv])
        cols.extend([iv, iu])
        vals.extend([val, val])

    order = np.lexsort((np.array(rows), np.array(cols)))
    rows = np.array(rows, dtype=np.int64)[order]
    cols = np.array(cols, dtype=np.int64)[order]
    vals = np.array(vals, dtype=np.float64)[order]
    return np.vstack([rows, cols, vals]).T


def write_h5(path: Path, h: dict[int, float], j_terms: dict[tuple[int, int], float]):
    nodes = sorted(h.keys())
    idx = regular_index(nodes)
    biases = np.array([h[n] for n in nodes], dtype=np.float64)
    j_coo = to_coo(nodes, j_terms, idx)

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.create_group("Ising")
        g.create_dataset("biases", data=biases)
        g.create_dataset("J_coo", data=j_coo)


def main():
    parser = argparse.ArgumentParser(description="Generate perfect-Chimera Ising instances in HDF5")
    parser.add_argument("--rows", type=int, default=8, help="Chimera rows (default: 8)")
    parser.add_argument("--cols", type=str, default="2,3,4,5,6,7,8,9,10,11,12,13,14,15,16", help="Comma-separated Chimera columns")
    parser.add_argument("--rep", type=int, default=1, help="Number of reps per size")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    cols_list = parse_list(args.cols)
    rng = np.random.default_rng(args.seed)

    for c in cols_list:
        g = dnx.chimera_graph(args.rows, c, 4)
        for inst in range(1, args.rep + 1):
            qubo = build_qubo(g, rng)
            h, j_terms = qubo_to_ising(qubo)
            out = args.out_dir / f"random_ising_chimera_C{c}_inst{inst}_linear.h5"
            write_h5(out, h, j_terms)
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
