module NeuralAttentionlibAMDGPUExt

using NeuralAttentionlib
using NeuralAttentionlib.Adapt
using NeuralAttentionlib: TypedPtr, AbstractArrayMask, Indexer, GetIndexer
using AMDGPU

import LinearAlgebra
import LinearAlgebra.BLAS
using LinearAlgebra.BLAS: get_num_threads, set_num_threads

const NAlib = NeuralAttentionlib

import AMDGPU.rocBLAS
for (fname, elty) in
    ((:rocblas_dgemm_strided_batched, :Float64),
     (:rocblas_sgemm_strided_batched, :Float32),
     (:rocblas_hgemm_strided_batched, :Float16),
     (:rocblas_zgemm_strided_batched, :ComplexF64),
     (:rocblas_cgemm_strided_batched, :ComplexF32))
    @eval begin
        @inline function NeuralAttentionlib.unsafe_gemm_strided_batched!(
            transA::Char, transB::Char,
            m::Integer, n::Integer, k::Integer,
            alpha::($elty), tptrA::TypedPtr{ROCArray, $elty}, lda::Integer, strideA::Integer,
            tptrB::TypedPtr{ROCArray, $elty}, ldb::Integer, strideB::Integer, beta::($elty),
            tptrC::TypedPtr{ROCArray, $elty}, ldc::Integer, strideC::Integer, batchCount::Integer)

            ptrA = tptrA.ptr
            ptrB = tptrB.ptr
            ptrC = tptrC.ptr
            rocBLAS.$fname(rocBLAS.handle(),
                           transA, transB, m, n, k,
                           alpha, ptrA, lda, strideA,
                           ptrB, ldb, strideB, beta,
                           ptrC, ldc, strideC, batchCount)
            return nothing
        end
    end
end

NeuralAttentionlib.ptrtypetag(::AMDGPU.ROCArrayBackend) = ROCArray
NeuralAttentionlib.check_strided_gemm_type(A::ROCArray{Float16}) = true

Adapt.adapt(to::AMDGPU.Runtime.Adaptor, m::AbstractArrayMask) =
    Indexer{typeof(m)}(map(Base.Fix1(Adapt.adapt, to), GetIndexer(m).__fields))
Adapt.adapt(to::AMDGPU.Runtime.Adaptor, m::NAlib.FlipMask) = Indexer{typeof(m)}((mask = adapt(to, m.mask),))
Adapt.adapt(to::AMDGPU.Runtime.Adaptor, m::NAlib.CombinedMask) =
    Indexer{typeof(m)}((f = adapt(to, m.f), masks = map(Base.Fix1(adapt, to), m.masks)))
Adapt.adapt(to::AMDGPU.Runtime.Adaptor, m::NAlib.BatchedMask) =
    Indexer{typeof(m)}((mask = adapt(to, m.mask), batch_dim = static(m.batch_dim)))
Adapt.adapt(to::AMDGPU.Runtime.Adaptor, m::NAlib.RepeatMask) = Indexer{typeof(m)}((mask = adapt(to, m.mask), num = m.num))
Adapt.adapt(to::AMDGPU.Runtime.Adaptor, m::NAlib.BiSequenceMask) =
    Indexer{typeof(m)}((q_mask = adapt(to, m.q_mask), k_mask = adapt(to, m.k_mask)))

end
