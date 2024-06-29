using Test, Pkg
using Flux
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
        elseif x == "metal"
            return :metal
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
        Pkg.add(["CUDA"])
        using CUDA
        CUDA.allowscalar(false)
        Flux.gpu_backend!("CUDA")
    elseif GPUBACKEND == :amdgpu
        Pkg.add(["AMDGPU"])
        using AMDGPU
        AMDGPU.allowscalar(false)
        Flux.gpu_backend!("AMDGPU")
    elseif GPUBACKEND == :metal
        Pkg.add(["Metal"])
        using Metal
        Metal.allowscalar(false)
        Flux.gpu_backend!("Metal")
    end

end
@show GPUBACKEND
@show USE_GPU

using NeuralAttentionlib

using Random
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
