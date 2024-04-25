module NeuralAttentionlibGPUArraysExt

using NeuralAttentionlib
using NeuralAttentionlib: CollapsedDimsArray
using GPUArrays
using GPUArrays.GPUArraysCore

GPUArraysCore.backend(::Type{<:CollapsedDimsArray{E, A}}) where {E, A} = GPUArraysCore.backend(A)

function NeuralAttentionlib.batched_transpose_f!(f, B::AnyGPUArray{T, 3}, A::AnyGPUArray{T, 3}) where T
    axes(B,1) == axes(A,2) && axes(B,2) == axes(A,1) && axes(A,3) == axes(B,3) || throw(DimensionMismatch(string(f)))
    GPUArrays.gpu_call(B, A) do ctx, B, A
        idx = GPUArrays.@cartesianidx A
        @inbounds B[idx[2], idx[1], idx[3]] = f(A[idx[1], idx[2], idx[3]])
        return
    end
    return B
end

end
