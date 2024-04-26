module NeuralAttentionlibCUDAExt

using NeuralAttentionlib
using NeuralAttentionlib.Adapt
using NeuralAttentionlib: TypedPtr, AbstractArrayMask, Indexer, GetIndexer
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

Adapt.adapt(to::CUDA.KernelAdaptor, m::AbstractArrayMask) =
    Indexer{typeof(m)}(map(Base.Fix1(Adapt.adapt, to), GetIndexer(m).__fields))
Adapt.adapt(to::CUDA.KernelAdaptor, m::NAlib.FlipMask) = Indexer{typeof(m)}((mask = adapt(to, m.mask),))
Adapt.adapt(to::CUDA.KernelAdaptor, m::NAlib.CombinedMask) =
    Indexer{typeof(m)}((f = adapt(to, m.f), masks = map(Base.Fix1(adapt, to), m.masks)))
Adapt.adapt(to::CUDA.KernelAdaptor, m::NAlib.BatchedMask) =
    Indexer{typeof(m)}((mask = adapt(to, m.mask), batch_dim = static(m.batch_dim)))
Adapt.adapt(to::CUDA.KernelAdaptor, m::NAlib.RepeatMask) = Indexer{typeof(m)}((mask = adapt(to, m.mask), num = m.num))
Adapt.adapt(to::CUDA.KernelAdaptor, m::NAlib.BiSequenceMask) =
    Indexer{typeof(m)}((q_mask = adapt(to, m.q_mask), k_mask = adapt(to, m.k_mask)))

end
