using Random
using LinearAlgebra
using SparseArrays
using HDF5

struct Random3BodyLinearGenerator{T<:AbstractFloat}
    size::Int
    rng::AbstractRNG
end

Random3BodyLinearGenerator{T}(size::Int, seed::Int) where {T<:AbstractFloat} =
    Random3BodyLinearGenerator{T}(size, MersenneTwister(seed))
Random3BodyLinearGenerator{T}(size::Int) where {T<:AbstractFloat} =
    Random3BodyLinearGenerator{T}(size, Random.default_rng())

function generate(g::Random3BodyLinearGenerator{T}) where {T}
    h = randn(g.rng, T, g.size)
    J = randn(g.rng, T, g.size - 1)
    K = randn(g.rng, T, g.size - 2)
    sgn_K = sign.(K)

    n_spins = g.size + (g.size - 2)
    couplings = spzeros(T, n_spins, n_spins)
    biases = zeros(T, n_spins)

    for i in 1:(g.size - 2)
        i1, i2, i3 = i, i + 1, i + 2
        aux_i = g.size + i

        biases[i1] += h[i1] + sgn_K[i] * abs(K[i])
        biases[i2] += sgn_K[i] * abs(K[i])
        biases[i3] += sgn_K[i] * abs(K[i])
        biases[aux_i] += 2 * sgn_K[i] * abs(K[i])

        couplings[i1, i2] += J[i] + abs(K[i])
        couplings[i1, i3] += abs(K[i])
        couplings[i2, i3] += abs(K[i])
        couplings[i1, aux_i] += 2 * abs(K[i])
        couplings[i2, aux_i] += 2 * abs(K[i])
        couplings[i3, aux_i] += 2 * abs(K[i])
    end

    biases[g.size - 1] += h[g.size - 1]
    biases[g.size] += h[g.size]
    couplings[g.size - 1, g.size] += J[g.size - 1]
    couplings = couplings + couplings'

    return Matrix(couplings), biases
end

function couplings_to_coo(couplings::AbstractMatrix{T}; tol::Real = 0.0) where {T<:Real}
    n = size(couplings, 1)
    entries = Vector{Tuple{Int64,Int64,T}}()

    for i in 1:n
        push!(entries, (Int64(i), Int64(i), zero(T)))
    end

    for j in 1:n
        for i in 1:n
            i == j && continue
            v = couplings[i, j]
            if abs(v) > tol
                push!(entries, (Int64(i), Int64(j), v))
            end
        end
    end

    sort!(entries, by = x -> (x[2], x[1]))
    I = Int64[e[1] for e in entries]
    J = Int64[e[2] for e in entries]
    V = T[e[3] for e in entries]
    return I, J, V
end

function write_sparse_ising_h5(path::AbstractString, couplings::AbstractMatrix{T}, biases::AbstractVector{T}) where {T<:Real}
    @assert size(couplings, 1) == size(couplings, 2) == length(biases)
    I, J, V = couplings_to_coo(couplings)
    n = Int64(length(biases))

    h5open(path, "w") do f
        ising = create_group(f, "Ising")
        attributes(ising)["sparsity"] = "sparse"
        attributes(ising)["type"] = "BinaryQuadraticModel"
        attributes(ising)["var_type"] = "SPIN"
        write(ising, "L", n)
        write(ising, "biases", collect(biases))
        c = create_group(ising, "couplings")
        attributes(c)["type"] = "SparseMatrixCSC"
        write(c, "I", I)
        write(c, "J", J)
        write(c, "V", V)
        write(c, "dims", Int64[n, n])
    end
end

to_gen = [(10, 10), (100, 10), (156, 10), (433, 10), (1000, 10)]
out_dir = joinpath(@__DIR__, "instances")
ispath(out_dir) || mkpath(out_dir)

for (s, rep) in to_gen
    gen = Random3BodyLinearGenerator{Float64}(s)
    for i in 1:rep
        println("Generating random3Body L=$s inst=$i")
        couplings, biases = generate(gen)
        out_name = "random3BodyIsing_sparse_L=$(s)_$(i).h5"
        write_sparse_ising_h5(joinpath(out_dir, out_name), couplings, biases)
    end
end
