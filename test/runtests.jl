using Test
using NeuralAttentionlib

using Random
using Flux

const tests = [
    "mta",
]

Random.seed!(0)

include("old_impl/old_impl.jl")

@testset "Transformers" begin
    for t in tests
        fp = joinpath(dirname(@__FILE__), "$t.jl")
        @info "Test $(uppercase(t))"
        include(fp)
    end
end

