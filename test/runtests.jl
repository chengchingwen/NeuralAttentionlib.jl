using Test
using NeuralAttentionlib

using Random
using Flux
using NNlib
using Static
using ChainRulesCore
using ChainRulesTestUtils

const tests = [
    "collapseddim",
    "matmul",
    "mask",
    "mha",
]

Random.seed!(0)

include("old_impl/old_impl.jl")
using .Old_Impl
using .Old_Impl: batched_triu!, batched_tril!

@testset "NeuralAttentionlib" begin
    for t in tests
        fp = joinpath(dirname(@__FILE__), "$t.jl")
        @info "Test $(uppercase(t))"
        include(fp)
    end
end

