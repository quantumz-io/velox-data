#!/usr/bin/env julia
using CSV
using DataFrames
using SparseArrays
using LinearAlgebra
using Graphs
using CUDA
using GenericTensorNetworks
using TropicalGEMM
using VeloxQIO

const DEFAULT_INST_DIR = normpath(joinpath(@__DIR__, "..", "instances", "chimera"))
const DEFAULT_OUT_DIR = normpath(joinpath(@__DIR__, "..", "instances", "chimera"))

parse_cols(name::String) = parse(Int, match(r"_C(\d+)_", name).captures[1])

function to_dense(model)
    J = Matrix{Float32}(model.couplings)
    h = Vector{Float32}(model.biases)
    return J, h
end

function run_ttn(J::AbstractMatrix, h::AbstractVector; backend::Symbol, objective::Symbol)
    G = SimpleGraph(J)
    JG = [J[e.src, e.dst] for e in edges(G)]
    spinglass = SpinGlass(G, JG, h)
    problem = GenericTensorNetwork(spinglass; optimizer=TreeSA())

    if objective == :energy_only
        usecuda = backend == :gpu
        tts = @elapsed res = solve(problem, SizeMin(), usecuda=usecuda)
        Emin = Float64(res[1])
        return Emin, tts, nothing
    elseif objective == :state
        usecuda = backend == :gpu
        tts = @elapsed res = solve(problem, SingleConfigMin(), usecuda=usecuda)
        state = if backend == :gpu
            CUDA.@allowscalar read_config(res[])
        else
            read_config(res[])
        end
        Emin = Float64(energy(problem.problem, 1 .- 2 .* Int.(state)))
        return Emin, tts, string(state)
    else
        error("Unknown objective: $objective")
    end
end

function output_name(backend::Symbol, objective::Symbol)
    if backend == :cpu_tgemm && objective == :energy_only
        return "TTN_chimera_CPU_TGEMM_energy_only.csv"
    elseif backend == :cpu_tgemm && objective == :state
        return "TTN_chimera_CPU_TGEMM.csv"
    elseif backend == :gpu && objective == :energy_only
        return "TTN_chimera_GPU_energy_only.csv"
    elseif backend == :gpu && objective == :state
        return "TTN_chimera_GPU.csv"
    else
        error("Unsupported backend/objective combination")
    end
end

function parse_args(args)
    backend = :cpu_tgemm
    objective = :energy_only
    inst_dir = DEFAULT_INST_DIR
    out_dir = DEFAULT_OUT_DIR

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--backend"
            i += 1
            backend = Symbol(args[i])
        elseif a == "--objective"
            i += 1
            objective = Symbol(args[i])
        elseif a == "--instances"
            i += 1
            inst_dir = args[i]
        elseif a == "--out-dir"
            i += 1
            out_dir = args[i]
        else
            error("Unknown arg: $a")
        end
        i += 1
    end

    return backend, objective, inst_dir, out_dir
end

function main(args)
    backend, objective, inst_dir, out_dir = parse_args(args)
    mkpath(out_dir)

    files = sort(filter(f -> endswith(f, ".h5") && occursin("random_ising_chimera_C", f), readdir(inst_dir)); by=parse_cols)
    out_file = joinpath(out_dir, output_name(backend, objective))

    if objective == :energy_only
        rows = DataFrame(instance=String[], num_var=Int[], optimizer=String[], TTN_energy=Float64[], runtime=Float64[])
        for f in files
            model = load_model(Float64, joinpath(inst_dir, f))
            J, h = to_dense(model)
            en, tts, _ = run_ttn(J, h; backend=backend, objective=objective)
            push!(rows, (f, length(h), "TreeSA", en, tts))
            println("$(f): E=$(en), t=$(tts)")
        end
        CSV.write(out_file, rows)
    else
        rows = DataFrame(instance=String[], num_var=Int[], optimizer=String[], TTN_energy=Float64[], runtime=Float64[], state=String[])
        for f in files
            model = load_model(Float64, joinpath(inst_dir, f))
            J, h = to_dense(model)
            en, tts, state = run_ttn(J, h; backend=backend, objective=objective)
            push!(rows, (f, length(h), "TreeSA", en, tts, state))
            println("$(f): E=$(en), t=$(tts)")
        end
        CSV.write(out_file, rows)
    end

    println("Wrote $(out_file)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
