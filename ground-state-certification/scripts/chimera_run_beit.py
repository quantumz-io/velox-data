#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any


INSTANCE_RE = re.compile(r"random_qubo_chimera_C(?P<cols>\d+)_inst(?P<inst>\d+)_linear\.txt$")
BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INSTANCES_DIR = BASE_DIR / "instances" / "chimera"
DEFAULT_RESULTS_DIR = BASE_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BEIT on Chimera QUBO instances.")
    parser.add_argument(
        "--instances-dir",
        type=Path,
        default=DEFAULT_INSTANCES_DIR,
    )
    parser.add_argument("--instance-glob", default="random_qubo_chimera_C*_inst*_linear.txt")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "beit_chimera_runs.csv",
    )
    parser.add_argument(
        "--responses-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "beit_responses",
    )
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--api-key", default=None, help="API key; if omitted, reads BEIT_API_KEY env var")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_instance(path: Path) -> tuple[int, int]:
    match = INSTANCE_RE.match(path.name)
    if not match:
        raise ValueError(f"Unsupported Chimera QUBO filename: {path.name}")
    return int(match.group("cols")), int(match.group("inst"))


def load_qubo_txt(path: Path) -> dict[tuple[int, int], float]:
    qubo: dict[tuple[int, int], float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            i_s, j_s, v_s = line.split()
            i = int(i_s)
            j = int(j_s)
            v = float(v_s)
            qubo[(i, j)] = v
    return qubo


def embed_to_c816(qubo: dict[tuple[int, int], float], cols: int):
    import dimod
    import dwave_networkx as dnx
    g_sub = dnx.chimera_graph(8, cols, 4)
    g_full = dnx.chimera_graph(8, 16, 4)
    mappings = list(dnx.chimera_sublattice_mappings(g_sub, g_full))
    if not mappings:
        raise RuntimeError(f"No sublattice mapping found for cols={cols}")
    mapping = mappings[0]

    mapped_qubo: dict[tuple[int, int], float] = {}
    for (i, j), v in qubo.items():
        mapped_qubo[(mapping(i), mapping(j))] = v

    bqm = dimod.BinaryQuadraticModel.from_qubo(mapped_qubo)
    return bqm, len(list(bqm.variables))


def response_best_energy(response: Any) -> float:
    if hasattr(response, "record") and hasattr(response.record, "energy"):
        energies = response.record.energy
        if len(energies) > 0:
            return float(min(energies))

    if hasattr(response, "to_serializable"):
        data = response.to_serializable()
        if isinstance(data, dict):
            info = data.get("info", {})
            if isinstance(info, dict):
                # Some serializers keep vectors under info-like keys; fallback-friendly.
                for key in ("energies", "energy", "spectrum_energies"):
                    vals = info.get(key)
                    if isinstance(vals, list) and vals:
                        return float(min(vals))
    raise RuntimeError("Could not derive best energy from BEIT response")


def build_solver(api_key: str):
    from beit.qubo_solver.solver_connection import AWSSolverConnection
    from beit.qubo_solver.beit_solver import BEITSolver

    connection = AWSSolverConnection(api_key)
    return BEITSolver(connection)


def ensure_csv_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "instance",
                "cols",
                "run_id",
                "num_var",
                "status",
                "best_energy",
                "runtime_ns",
                "runtime_s",
                "response_json_path",
                "error",
            ]
        )


def append_csv_row(
    path: Path,
    instance: str,
    cols: int,
    run_id: int,
    num_var: int,
    status: str,
    best_energy: float | None,
    runtime_ns: int | None,
    response_json_path: str | None,
    error: str | None,
) -> None:
    runtime_s = None if runtime_ns is None else runtime_ns / 1e9
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                instance,
                cols,
                run_id,
                num_var,
                status,
                "" if best_energy is None else best_energy,
                "" if runtime_ns is None else runtime_ns,
                "" if runtime_s is None else runtime_s,
                "" if response_json_path is None else response_json_path,
                "" if error is None else error,
            ]
        )


def response_dump_name(instance_name: str, run_id: int) -> str:
    stem = instance_name.removesuffix(".txt")
    return f"response_{stem}_run{run_id}.json"


def main() -> None:
    args = parse_args()
    instances = sorted(
        (p for p in args.instances_dir.glob(args.instance_glob) if p.is_file()),
        key=lambda p: parse_instance(p)[0],
    )

    if not instances:
        raise SystemExit(
            f"No instances found in {args.instances_dir} with glob {args.instance_glob}"
        )

    if args.dry_run:
        print(f"Dry run: {len(instances)} Chimera QUBO instances")
        for path in instances:
            cols, inst = parse_instance(path)
            print(f"  cols={cols} inst={inst} file={path}")
        return

    api_key = args.api_key
    if not api_key:
        import os

        api_key = os.getenv("BEIT_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key: pass --api-key or set BEIT_API_KEY")

    solver = build_solver(api_key)
    ensure_csv_header(args.output_csv)
    args.responses_dir.mkdir(parents=True, exist_ok=True)

    for instance_path in instances:
        cols, _inst_id = parse_instance(instance_path)
        instance_name = instance_path.name
        print(f"Processing {instance_name}")

        num_var = 0
        try:
            qubo = load_qubo_txt(instance_path)
            bqm, num_var = embed_to_c816(qubo, cols)

            start_ns = time.time_ns()
            response = solver.sample(bqm)
            runtime_ns = time.time_ns() - start_ns

            best_energy = response_best_energy(response)

            dump_path = args.responses_dir / response_dump_name(instance_name, args.run_id)
            with dump_path.open("w", encoding="utf-8") as handle:
                json.dump(response.to_serializable(), handle)

            append_csv_row(
                args.output_csv,
                instance=instance_name,
                cols=cols,
                run_id=args.run_id,
                num_var=num_var,
                status="ok",
                best_energy=best_energy,
                runtime_ns=runtime_ns,
                response_json_path=str(dump_path),
                error=None,
            )
            print(f"  ok energy={best_energy} runtime_s={runtime_ns / 1e9:.6f}")

        except Exception as exc:  # noqa: BLE001
            append_csv_row(
                args.output_csv,
                instance=instance_name,
                cols=cols,
                run_id=args.run_id,
                num_var=num_var,
                status="failed",
                best_energy=None,
                runtime_ns=None,
                response_json_path=None,
                error=str(exc),
            )
            print(f"  failed: {exc}")

    print("Done.")


if __name__ == "__main__":
    main()
