#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import dimod
import h5py
from dwave.embedding import chain_breaks, embed_bqm, unembed_sampleset
from dwave.embedding.pegasus import find_clique_embedding
from dwave.system import DWaveSampler


INSTANCE_RE = re.compile(r"random_ising_clique_L(?P<L>\d+)_inst(?P<inst>\d+)_linear\.h5$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clique-embedded benchmark on Pegasus.")
    parser.add_argument("--instances-dir", type=Path, required=True)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instance-glob", default="*.h5")
    parser.add_argument("--num-reads", type=int, default=2**10)
    parser.add_argument("--num-shots", type=int, default=10)
    parser.add_argument("--anneal-times", default="100")
    parser.add_argument("--reuse-embeddings", action="store_true")
    parser.add_argument("--solver", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_anneal_times(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("No anneal times provided")
    return values


def parse_instance(path: Path) -> tuple[int, int]:
    match = INSTANCE_RE.match(path.name)
    if not match:
        raise ValueError(f"Unsupported clique filename: {path.name}")
    return int(match.group("L")), int(match.group("inst"))


def canonical_name(path: Path) -> str:
    l_value, inst = parse_instance(path)
    return f"random_ising_clique_L{l_value}_inst{inst}_qpu"


def _add_coupling(target: dict[tuple[int, int], float], i: int, j: int, value: float) -> None:
    if i == j:
        return
    a, b = (i, j) if i < j else (j, i)
    target[(a, b)] = target.get((a, b), 0.0) + float(value)


def load_h5_ising(path: Path) -> dimod.BinaryQuadraticModel:
    with h5py.File(path, "r") as f:
        if "Ising" not in f:
            raise ValueError(f"Unsupported HDF5 instance format, missing Ising group: {path}")
        group = f["Ising"]
        h = [float(x) for x in group["biases"][...].reshape(-1)]
        couplings = group["couplings"][...]
        if couplings.ndim == 3 and couplings.shape[2] == 1:
            couplings = couplings[:, :, 0]

    j: dict[tuple[int, int], float] = {}
    for i in range(couplings.shape[0]):
        h[i] += float(couplings[i, i])
        for jj in range(i + 1, couplings.shape[1]):
            value = float(couplings[i, jj])
            if value != 0.0:
                _add_coupling(j, i, jj, value)
    return dimod.BinaryQuadraticModel.from_ising(h, j)


def load_embedding(path: Path) -> dict[int, list[int]] | None:
    if not path.exists():
        return None
    with path.open("r") as f:
        data = json.load(f)
    return {int(k): [int(x) for x in v] for k, v in data.items()}


def save_embedding(path: Path, embedding: dict[int, list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump({str(k): list(v) for k, v in embedding.items()}, f, indent=2, sort_keys=True)


def get_sampler(solver: str | None) -> DWaveSampler:
    kwargs: dict = {}
    token = os.getenv("DWAVE_API_TOKEN")
    if token:
        kwargs["token"] = token
    kwargs["solver"] = solver if solver else {"topology__type": "pegasus", "qpu": True}
    return DWaveSampler(**kwargs)


def append_embedding_time(csv_path: Path, instance: str, elapsed_us: float, reused: bool) -> None:
    header = "instance,embedding_time_us,reused\n"
    if not csv_path.exists():
        csv_path.write_text(header)
    with csv_path.open("a") as f:
        f.write(f"{instance},{elapsed_us:.3f},{int(reused)}\n")


def logical_response(
    response: dimod.SampleSet,
    embedding: dict[int, list[int]],
    logical_bqm: dimod.BinaryQuadraticModel,
) -> tuple[dict[float, int], list[dict], list[int], float]:
    unembedded = unembed_sampleset(
        response,
        embedding,
        logical_bqm,
        chain_break_method=chain_breaks.majority_vote,
        chain_break_fraction=True,
    )
    variables = sorted(int(v) for v in logical_bqm.variables)
    spectrum: dict[float, int] = {}
    states: list[dict] = []
    total_occurrences = 0
    weighted_chain_break = 0.0

    for sample, count, chain_break_fraction in unembedded.data(
        ["sample", "num_occurrences", "chain_break_fraction"]
    ):
        occurrences = int(count)
        energy = float(logical_bqm.energy(sample))
        cbf = float(chain_break_fraction)
        spectrum[energy] = spectrum.get(energy, 0) + occurrences
        total_occurrences += occurrences
        weighted_chain_break += cbf * occurrences
        states.append(
            {
                "state": [int(sample[v]) for v in variables],
                "energy": energy,
                "num_occurrences": occurrences,
                "chain_break_fraction": cbf,
            }
        )

    mean_chain_break = weighted_chain_break / total_occurrences if total_occurrences else 0.0
    return spectrum, states, variables, mean_chain_break


def save_result_h5(
    output_file: Path,
    instance_name: str,
    shot: int,
    anneal_time: float,
    spectrum: dict[float, int],
    states: list[dict],
    variable_names: list[int],
    timing: dict,
) -> None:
    with h5py.File(output_file, "a") as f:
        f.attrs["instance_name"] = instance_name
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
        states_group.attrs["variable_names"] = variable_names
        for idx, state_data in enumerate(states):
            state_group = states_group.create_group(f"state_{idx:03d}")
            state_group.create_dataset("spin_config", data=state_data["state"])
            state_group.attrs["energy"] = state_data["energy"]
            state_group.attrs["num_occurrences"] = state_data["num_occurrences"]
            state_group.attrs["chain_break_fraction"] = state_data["chain_break_fraction"]

        timing_group = shot_group.create_group("timing")
        for key, value in timing.items():
            timing_group.attrs[key] = value


def main() -> None:
    args = parse_args()
    instances = sorted(p for p in args.instances_dir.glob(args.instance_glob) if p.is_file())
    anneal_times = parse_anneal_times(args.anneal_times)
    if not instances:
        raise SystemExit(f"No instances found in {args.instances_dir} with glob {args.instance_glob}")

    if args.dry_run:
        print(f"Dry run: {len(instances)} clique instances")
        for path in instances[:20]:
            print(f"  {path}")
        return

    sampler = get_sampler(args.solver)
    target_graph = sampler.to_networkx_graph()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.embeddings_dir.mkdir(parents=True, exist_ok=True)
    timing_csv = args.output_dir / "embedding_times.csv"
    embedding_cache: dict[int, dict[int, list[int]]] = {}

    for instance_path in instances:
        l_value, _ = parse_instance(instance_path)
        name = canonical_name(instance_path)
        bqm = load_h5_ising(instance_path)
        embedding_path = args.embeddings_dir / f"embedding_clique_L{l_value}.json"

        embedding = embedding_cache.get(l_value)
        reused = embedding is not None
        start_embedding_ns = time.time_ns()
        if embedding is None and args.reuse_embeddings:
            embedding = load_embedding(embedding_path)
            reused = embedding is not None
        if embedding is None:
            embedding = find_clique_embedding(l_value, target_graph=target_graph)
            save_embedding(embedding_path, embedding)
        embedding_cache[l_value] = embedding
        embedding_elapsed_us = (time.time_ns() - start_embedding_ns) / 1e3
        append_embedding_time(timing_csv, name, embedding_elapsed_us, reused)

        embedded_bqm = embed_bqm(bqm, embedding, sampler.adjacency)
        out_h5 = args.output_dir / f"{name}.h5"
        print(f"Processing {instance_path.name} (L={l_value}) -> {out_h5.name}")

        for anneal_time in anneal_times:
            for shot in range(args.num_shots):
                start_ns = time.time_ns()
                response = sampler.sample(
                    embedded_bqm,
                    annealing_time=anneal_time,
                    num_reads=args.num_reads,
                    answer_mode="histogram",
                )
                elapsed_us = (time.time_ns() - start_ns) / 1e3
                spectrum, states, variable_names, mean_chain_break = logical_response(
                    response, embedding, bqm
                )
                timing = dict(response.info.get("timing", {})) if hasattr(response, "info") else {}
                timing["user_time"] = elapsed_us
                timing["energy_space"] = "logical_unembedded"
                timing["chain_break_method"] = "majority_vote"
                timing["mean_chain_break_fraction"] = mean_chain_break
                save_result_h5(out_h5, name, shot, anneal_time, spectrum, states, variable_names, timing)

    print("Done.")


if __name__ == "__main__":
    main()
