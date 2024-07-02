module NeuralAttentionlibMetalExt

using NeuralAttentionlib
using NeuralAttentionlib.Adapt
using NeuralAttentionlib.NNlib
using Metal

const NAlib = NeuralAttentionlib

function mpsmatrix(arr::MtlArray{T}, lda, stride, batch) where T
    sz = sizeof(T)
    N = length(arr)
    n_cols_rows = iszero(stride) ? N : stride
    n_cols = lda
    n_rows = div(n_cols_rows, n_cols)
    n_matrices = batch
    row_bytes = sz * n_cols
    matrix_bytes = iszero(stride) ? 0 : row_bytes * n_rows
    desc = MPS.MPSMatrixDescriptor(n_rows, n_cols, n_matrices, row_bytes, matrix_bytes, T)
    offset = arr.offset * sz
    return MPS.MPSMatrix(arr, desc, offset), (n_cols, n_rows, n_matrices)
end

for elty in (:Float32, :Float16)
    @eval begin
        @inline function NAlib.gemm_strided_batched_impl!(
            transA::Char, transB::Char,
            m::Integer, n::Integer, k::Integer,
            alpha::($elty), A::MtlArray{$elty, N1}, lda::Integer, strideA::Integer,
            B::MtlArray{$elty, N2}, ldb::Integer, strideB::Integer, beta::($elty),
            C::MtlArray{$elty, N3}, ldc::Integer, strideC::Integer, batchCount::Integer
        ) where {N1, N2, N3}

            transpose_a = transA != 'N'
            transpose_b = transB != 'N'

            mps_a, shp_A = mpsmatrix(A, lda, strideA, batchCount)
            mps_b, shp_B = mpsmatrix(B, ldb, strideB, batchCount)
            mps_c, shp_C = mpsmatrix(C, ldc, strideC, batchCount)

            cols_a = shp_A[transpose_a ? 1 : 2]
            cols_c, rows_c = shp_C

            mat_mul_kernel = MPS.MPSMatrixMultiplication(device(), transpose_b, transpose_a, rows_c, cols_c, cols_a, alpha, beta)

            cmdbuf = Metal.MTLCommandBuffer(Metal.global_queue(device()))
            MPS.encode!(cmdbuf, mat_mul_kernel, mps_b, mps_a, mps_c)
            Metal.commit!(cmdbuf)
            return C
        end
    end
end

# TODO:
function NNlib._batched_gemm!(::Type{<:MtlArray}, transA::Char, transB::Char, α::Number, A, B, β::Number, C)
    a = if transA == 'T'
        batched_transpose(A)
    elseif transA == 'C'
        batched_adjoint(A)
    else
        A
    end
    b = if transB == 'T'
        batched_transpose(B)
    elseif transB == 'C'
        batched_adjoint(B)
    else
        B
    end
    return NNlib.batched_mul_generic!(C, a, b, α, β)
end

NAlib.use_gemm_strided_batched(A::MtlArray{ComplexF32}, B::MtlArray{ComplexF32}) = false
NAlib.use_gemm_strided_batched(A::NAlib.CollapsedDimsArray{ComplexF32, <:MtlMatrix{ComplexF32}}, B::NAlib.CollapsedDimsArray{ComplexF32, <:MtlMatrix{ComplexF32}}) = false
NAlib.ptrtypetag(::Metal.mtlArrayBackend) = MtlArray
NAlib.check_strided_gemm_type(A::MtlArray{Float16}) = true


end
