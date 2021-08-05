using Test
using NeuralAttentionlib

using Random
using Flux
using NNlib

const tests = [
    "collapseddim",
    "matmul",
    "mha",
]

Random.seed!(0)

include("old_impl/old_impl.jl")

@testset "NeuralAttentionlib" begin
    for t in tests
        fp = joinpath(dirname(@__FILE__), "$t.jl")
        @info "Test $(uppercase(t))"
        include(fp)
    end
end

