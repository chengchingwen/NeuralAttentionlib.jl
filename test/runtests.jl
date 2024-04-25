using Test
using NeuralAttentionlib

using Random
using Flux
using CUDA
using AMDGPU
using NNlib
using Static
using ChainRulesCore
using ChainRulesTestUtils

const tests = [
    "collapseddims",
    "matmul",
    "mask",
    "functional",
    "mha",
]

Random.seed!(0)

include("old_impl/old_impl.jl")
using .Old_Impl
using .Old_Impl: batched_triu!, batched_tril!

function should_test_cuda()
    e = get(ENV, "JL_PKG_TEST_CUDA", false)
    e isa Bool && return e
    if e isa String
        x = tryparse(Bool, e)
        return isnothing(x) ? false : x
    else
        return false
    end
end

function should_test_amdgpu()
    e = get(ENV, "JL_PKG_TEST_AMDGPU", false)
    e isa Bool && return e
    if e isa String
        x = tryparse(Bool, e)
        return isnothing(x) ? false : x
    else
        return false
    end
end

const USE_CUDA = @show should_test_cuda()
const USE_AMDGPU = @show should_test_amdgpu()

if USE_CUDA
    CUDA.allowscalar(false)
end

if USE_AMDGPU
    AMDGPU.allowscalar(false)
end

device(x) = USE_CUDA || USE_AMDGPU ? gpu(x) : x

drandn(arg...) = randn(arg...) |> device
drand(arg...) = rand(arg...) |> device
dones(arg...) = ones(arg...) |> device
dzeros(arg...) = zeros(arg...) |> device

@testset "NeuralAttentionlib" begin
    for t in tests
        fp = joinpath(dirname(@__FILE__), "$t.jl")
        @info "Test $(uppercase(t))"
        include(fp)
    end
end
