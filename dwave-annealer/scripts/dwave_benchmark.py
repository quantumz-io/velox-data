#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import dimod
import h5py
from dwave.system import DWaveSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run native D-Wave benchmarks.")
    parser.add_argument("--topology", choices=["pegasus", "zephyr"], required=True)
    parser.add_argument("--instances-dir", type=Path, required=True)
    parser.add_argument("--instance-glob", default="*")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-reads", type=int, default=2**10)
    parser.add_argument("--num-shots", type=int, default=10)
    parser.add_argument("--anneal-times", default="100")
    parser.add_argument("--solver", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_anneal_times(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("No anneal times provided")
    return values


def canonical_instance_name(path: Path) -> str:
    name = path.name
    for suffix in [".coo_real.h5", ".h5", ".txt", ".coo"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _add_coupling(target: dict[tuple[int, int], float], i: int, j: int, value: float) -> None:
    if i == j:
        return
    a, b = (i, j) if i < j else (j, i)
    target[(a, b)] = target.get((a, b), 0.0) + float(value)


def _normalize_index_base(i_idx, j_idx, nvars: int):
    if len(i_idx) == 0:
        return i_idx, j_idx
    min_idx = min(int(i_idx.min()), int(j_idx.min()))
    max_idx = max(int(i_idx.max()), int(j_idx.max()))
    if min_idx >= 1 and max_idx <= nvars:
        return i_idx - 1, j_idx - 1
    return i_idx, j_idx


def load_text_ising(path: Path) -> dimod.BinaryQuadraticModel:
    h: dict[int, float] = {}
    j: dict[tuple[int, int], float] = {}
    with path.open("r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Expected three columns in {path}: {line!r}")
            i = int(parts[0])
            jj = int(parts[1])
            value = float(parts[2])
            if i == jj:
                h[i] = h.get(i, 0.0) + value
            else:
                _add_coupling(j, i, jj, value)
    return dimod.BinaryQuadraticModel.from_ising(h, j)


def _flatten_biases(dataset) -> list[float]:
    arr = dataset[...]
    if getattr(arr, "ndim", 1) == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    elif getattr(arr, "ndim", 1) == 2 and arr.shape[0] == 1:
        arr = arr[0, :]
    elif getattr(arr, "ndim", 1) > 1:
        arr = arr.reshape(-1)
    return [float(x) for x in arr]


def load_h5_ising(path: Path) -> dimod.BinaryQuadraticModel:
    with h5py.File(path, "r") as f:
        if "Ising" not in f:
            raise ValueError(f"Unsupported HDF5 instance format, missing Ising group: {path}")
        group = f["Ising"]
        h = _flatten_biases(group["biases"]) if "biases" in group else []
        nvars = len(h)
        j: dict[tuple[int, int], float] = {}

        if "J_coo" in group:
            coo = group["J_coo"]
            i_idx = coo["I"][...].astype(int)
            j_idx = coo["J"][...].astype(int)
            vals = coo["V"][...]
            i_idx, j_idx = _normalize_index_base(i_idx, j_idx, nvars)
            for i, jj, value in zip(i_idx, j_idx, vals):
                ii = int(i)
                kk = int(jj)
                vv = float(value)
                if ii == kk:
                    h[ii] += vv
                else:
                    _add_coupling(j, ii, kk, vv)
        elif "couplings" in group:
            couplings = group["couplings"]
            if isinstance(couplings, h5py.Group):
                i_idx = couplings["I"][...].astype(int)
                j_idx = couplings["J"][...].astype(int)
                vals = couplings["V"][...]
                i_idx, j_idx = _normalize_index_base(i_idx, j_idx, nvars)
                for i, jj, value in zip(i_idx, j_idx, vals):
                    ii = int(i)
                    kk = int(jj)
                    vv = float(value)
                    if ii == kk:
                        h[ii] += vv
                    else:
                        _add_coupling(j, ii, kk, vv)
            else:
                arr = couplings[...]
                if arr.ndim == 3 and arr.shape[2] == 1:
                    arr = arr[:, :, 0]
                if not h:
                    h = [0.0] * arr.shape[0]
                for i in range(arr.shape[0]):
                    h[i] += float(arr[i, i])
                    for jj in range(i + 1, arr.shape[1]):
                        value = float(arr[i, jj])
                        if value != 0.0:
                            _add_coupling(j, i, jj, value)

    return dimod.BinaryQuadraticModel.from_ising(h, j)


def load_bqm(path: Path) -> dimod.BinaryQuadraticModel:
    if path.suffix == ".txt" or path.suffix == ".coo":
        return load_text_ising(path)
    if path.suffix == ".h5":
        return load_h5_ising(path)
    raise ValueError(f"Unsupported instance extension: {path}")


def get_sampler(args: argparse.Namespace) -> DWaveSampler:
    kwargs: dict = {}
    token = os.getenv("DWAVE_API_TOKEN")
    if token:
        kwargs["token"] = token
    kwargs["solver"] = args.solver if args.solver else {"topology__type": args.topology, "qpu": True}
    return DWaveSampler(**kwargs)


def sample_once(
    sampler: DWaveSampler,
    bqm: dimod.BinaryQuadraticModel,
    anneal_time: float,
    num_reads: int,
) -> tuple[dict[float, int], list[dict], dict]:
    start_ns = time.time_ns()
    response = sampler.sample(
        bqm,
        annealing_time=anneal_time,
        num_reads=num_reads,
        answer_mode="histogram",
    )
    elapsed_us = (time.time_ns() - start_ns) / 1e3

    variables = sorted(bqm.variables)
    spectrum: dict[float, int] = {}
    states: list[dict] = []
    for sample, energy, count in response.data(["sample", "energy", "num_occurrences"]):
        e = float(energy)
        c = int(count)
        spectrum[e] = spectrum.get(e, 0) + c
        states.append(
            {
                "state": [int(sample[v]) for v in variables],
                "energy": e,
                "num_occurrences": c,
                "variables": variables,
            }
        )

    timing = response.info.get("timing", {}) if hasattr(response, "info") else {}
    timing_info = {
        "qpu_access_time": timing.get("qpu_access_time", 0),
        "qpu_anneal_time_per_sample": timing.get("qpu_anneal_time_per_sample", 0),
        "qpu_readout_time_per_sample": timing.get("qpu_readout_time_per_sample", 0),
        "qpu_programming_time": timing.get("qpu_programming_time", 0),
        "qpu_delay_time_per_sample": timing.get("qpu_delay_time_per_sample", 0),
        "post_processing_overhead_time": timing.get("post_processing_overhead_time", 0),
        "user_time": elapsed_us,
    }
    return spectrum, states, timing_info


def save_shot_result(
    output_file: Path,
    bqm: dimod.BinaryQuadraticModel,
    instance_name: str,
    topology: str,
    shot: int,
    anneal_time: float,
    spectrum: dict[float, int],
    states: list[dict],
    timing: dict,
    num_reads: int,
    num_shots: int,
) -> None:
    with h5py.File(output_file, "a") as f:
        f.attrs["instance_name"] = instance_name
        f.attrs["num_variables"] = len(bqm.variables)
        f.attrs["num_edges"] = len(bqm.quadratic)
        f.attrs["NUM_READS"] = num_reads
        f.attrs["NUM_SHOTS"] = num_shots
        f.attrs["topology"] = topology

        anneal_group = f.require_group(f"{anneal_time:.3f}")
        anneal_group.attrs["annealing_time"] = anneal_time
        shot_key = f"shot_{shot:03d}"
        if shot_key in anneal_group:
            del anneal_group[shot_key]
        shot_group = anneal_group.create_group(shot_key)
        shot_group.attrs["shot"] = shot
        shot_group.create_dataset("spectrum_energies", data=list(spectrum.keys()))
        shot_group.create_dataset("spectrum_counts", data=list(spectrum.values()))

        states_group = shot_group.create_group("states")
        for idx, state_data in enumerate(states):
            state_group = states_group.create_group(f"state_{idx:03d}")
            state_group.create_dataset("spin_config", data=state_data["state"])
            state_group.attrs["energy"] = state_data["energy"]
            state_group.attrs["num_occurrences"] = state_data["num_occurrences"]
            if idx == 0:
                states_group.attrs["variable_names"] = state_data["variables"]

        timing_group = shot_group.create_group("timing")
        for key, value in timing.items():
            timing_group.attrs[key] = value


def main() -> None:
    args = parse_args()
    anneal_times = parse_anneal_times(args.anneal_times)
    instances = sorted(p for p in args.instances_dir.glob(args.instance_glob) if p.is_file())
    if not instances:
        raise SystemExit(f"No instances found in {args.instances_dir} with glob {args.instance_glob}")

    if args.dry_run:
        print(f"Dry run: topology={args.topology}, num_instances={len(instances)}")
        for path in instances[:20]:
            print(f"  {path}")
        return

    sampler = get_sampler(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for instance_path in instances:
        bqm = load_bqm(instance_path)
        name = canonical_instance_name(instance_path)
        out_h5 = args.output_dir / f"{name}.h5"
        print(f"Processing {instance_path.name} -> {out_h5.name}")
        for anneal_time in anneal_times:
            for shot in range(args.num_shots):
                spectrum, states, timing = sample_once(sampler, bqm, anneal_time, args.num_reads)
                save_shot_result(
                    output_file=out_h5,
                    bqm=bqm,
                    instance_name=name,
                    topology=args.topology,
                    shot=shot,
                    anneal_time=anneal_time,
                    spectrum=spectrum,
                    states=states,
                    timing=timing,
                    num_reads=args.num_reads,
                    num_shots=args.num_shots,
                )

    print("Done.")


if __name__ == "__main__":
    main()
