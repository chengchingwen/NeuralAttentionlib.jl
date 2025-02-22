using Test
using NeuralAttentionlib

using Random
using Flux
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

function testing_gpu()
    e = get(ENV, "JL_PKG_TEST_GPU", nothing)
    isnothing(e) && return nothing
    if e isa String
        x = lowercase(e)
        if isempty(x)
            return nothing
        elseif x == "cuda"
            return :cuda
        elseif x == "amdgpu"
            return :amdgpu
        end
    end
    error("Unknown value for `JL_PKG_TEST_GPU`: $x")
end

const GPUBACKEND = testing_gpu()
if isnothing(GPUBACKEND)
    const USE_GPU = false
else
    const USE_GPU = true
    if GPUBACKEND == :cuda
        using CUDA, cuDNN
        CUDA.allowscalar(false)
    elseif GPUBACKEND == :amdgpu
        using AMDGPU
        AMDGPU.allowscalar(false)
    end
end
@show GPUBACKEND
@show USE_GPU

device(x) = USE_GPU ? gpu(x) : x

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
