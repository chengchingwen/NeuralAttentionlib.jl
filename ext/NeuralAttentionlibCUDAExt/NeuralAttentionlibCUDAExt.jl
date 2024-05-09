module NeuralAttentionlibCUDAExt

using NeuralAttentionlib
using NeuralAttentionlib.Adapt
using NeuralAttentionlib: TypedPtr
using CUDA

import LinearAlgebra
import LinearAlgebra.BLAS
using LinearAlgebra.BLAS: get_num_threads, set_num_threads

const NAlib = NeuralAttentionlib

import CUDA.CUBLAS
for (fname, elty) in
    ((:cublasDgemmStridedBatched,:Float64),
     (:cublasSgemmStridedBatched,:Float32),
     (:cublasHgemmStridedBatched,:Float16),
     (:cublasZgemmStridedBatched,:ComplexF64),
     (:cublasCgemmStridedBatched,:ComplexF32))
    @eval begin
        @inline function NeuralAttentionlib.unsafe_gemm_strided_batched!(
            transA::Char, transB::Char,
            m::Integer, n::Integer, k::Integer,
            alpha::($elty), tptrA::TypedPtr{CuArray, $elty}, lda::Integer, strideA::Integer,
            tptrB::TypedPtr{CuArray, $elty}, ldb::Integer, strideB::Integer, beta::($elty),
            tptrC::TypedPtr{CuArray, $elty}, ldc::Integer, strideC::Integer, batchCount::Integer)

            ptrA = tptrA.ptr
            ptrB = tptrB.ptr
            ptrC = tptrC.ptr
            CUBLAS.$fname(CUBLAS.handle(),
                          transA, transB, m, n, k,
                          alpha, ptrA, lda, strideA,
                          ptrB, ldb, strideB, beta,
                          ptrC, ldc, strideC, batchCount)
            return nothing
        end
    end
end

NeuralAttentionlib.ptrtypetag(::CUDA.CuArrayBackend) = CuArray
NeuralAttentionlib.check_strided_gemm_type(A::CuArray{Float16}) = true

end
