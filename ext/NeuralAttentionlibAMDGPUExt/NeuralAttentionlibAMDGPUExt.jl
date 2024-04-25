module NeuralAttentionlibAMDGPUExt

using NeuralAttentionlib
using NeuralAttentionlib.Adapt
using NeuralAttentionlib: AbstractArrayMask, Indexer, GetIndexer
using AMDGPU
using AMDGPU.GPUArrays
using AMDGPU.GPUArrays.GPUArraysCore

import LinearAlgebra
import LinearAlgebra.BLAS
using LinearAlgebra.BLAS: get_num_threads, set_num_threads

const NAlib = NeuralAttentionlib

GPUArraysCore.backend(T::Type{<:NAlib.CollapsedDimsArray{E,<:ROCArray}}) where {E} = GPUArraysCore.backend(ROCArray{E,3})

function NeuralAttentionlib.batched_transpose_f!(f, B::AnyGPUArray{T,3}, A::AnyGPUArray{T,3}) where {T}
    axes(B, 1) == axes(A, 2) && axes(B, 2) == axes(A, 1) && axes(A, 3) == axes(B, 3) || throw(DimensionMismatch(string(f)))
    GPUArrays.gpu_call(B, A) do ctx, B, A
        idx = GPUArrays.@cartesianidx A
        @inbounds B[idx[2], idx[1], idx[3]] = f(A[idx[1], idx[2], idx[3]])
        return
    end
    return B
end
using AMDGPU
AMDGPU.roc
import AMDGPU.rocBLAS
for (fname, elty) in
    ((:rocblas_dgemm_strided_batched, :Float64),
    (:rocblas_sgemm_strided_batched, :Float32),
    (:rocblas_zgemm_strided_batched, :ComplexF64),
    (:rocblas_cgemm_strided_batched, :ComplexF32))
    @eval begin

        @inline function NeuralAttentionlib.unsafe_gemm_strided_batched!(
            transA::Char, transB::Char,
            m::Int, n::Int, k::Int,
            alpha::($elty), ptrA::ROCArray{$elty}, lda::Int, strideA::Int,
            ptrB::ROCArray{$elty}, ldb::Int, strideB::Int, beta::($elty),
            ptrC::ROCArray{$elty}, ldc::Int, strideC::Int, batchCount::Int)

            rocBLAS.$fname(rocBLAS.handle(),
                transA, transB, m, n, k,
                alpha, ptrA, lda, strideA,
                ptrB, ldb, strideB, beta,
                ptrC, ldc, strideC, batchCount)
            return nothing
        end

    end
end

for (elty, array) in (
    (:Float16, :ROCArray),
)
    @eval begin
        @inline function NeuralAttentionlib.unsafe_gemm_strided_batched!(
            transA::Char, transB::Char,
            m::Int, n::Int, k::Int,
            alpha::($elty), ptrA::Ptr{$elty}, lda::Int, strideA::Int,
            ptrB::Ptr{$elty}, ldb::Int, strideB::Int, beta::($elty),
            ptrC::Ptr{$elty}, ldc::Int, strideC::Int, batchCount::Int)

            # https://github.com/FluxML/NNlib.jl/blob/cd3851d31e95020e77e67f80fb6402b5b87db1e6/src/gemm.jl#L91-L139
            n_threads = min(Threads.nthreads(), 1 + max(m * k * batchCount, n * k * batchCount) ÷ 8000)
            if n_threads > 1
                old_threads = get_num_threads()
                set_num_threads(1)
                Threads.@sync for bs in Iterators.partition(1:batchCount, cld(batchCount, n_threads))
                    Threads.@spawn for b in bs
                        ptrAi = ptrA + (b - 1) * strideA * sizeof($elty)
                        ptrBi = ptrB + (b - 1) * strideB * sizeof($elty)
                        ptrCi = ptrC + (b - 1) * strideC * sizeof($elty)

                        NeuralAttentionlib.unsafe_gemm!(transA, transB, m, n, k,
                            alpha, ptrAi, lda,
                            ptrBi, ldb, beta,
                            ptrCi, ldc)
                    end
                end
                set_num_threads(old_threads)
            else
                for i = 1:batchCount
                    ptrAi = ptrA + (i - 1) * strideA * sizeof($elty)
                    ptrBi = ptrB + (i - 1) * strideB * sizeof($elty)
                    ptrCi = ptrC + (i - 1) * strideC * sizeof($elty)
                    NeuralAttentionlib.unsafe_gemm!(transA, transB, m, n, k,
                        alpha, ptrAi, lda,
                        ptrBi, ldb, beta,
                        ptrCi, ldc)
                end
            end
            return nothing
        end

        @inline function NeuralAttentionlib.gemm_strided_batched_impl!(
            transA::Char, transB::Char,
            m::Int, n::Int, k::Int,
            alpha::($elty), A::$array{$elty}, lda::Int, strideA::Int,
            B::$array{$elty}, ldb::Int, strideB::Int, beta::($elty),
            C::$array{$elty}, ldc::Int, strideC::Int, batchCount::Int)

            ptrA = pointer(A)
            ptrB = pointer(B)
            ptrC = pointer(C)
            GC.@preserve A B C begin
                NeuralAttentionlib.unsafe_gemm_strided_batched!(
                    transA, transB, m, n, k,
                    alpha, ptrA, lda, strideA,
                    ptrB, ldb, strideB, beta,
                    ptrC, ldc, strideC, batchCount)
            end
            return C
        end

        @inline function NeuralAttentionlib.gemm_strided_batched!(
            transA::Char, transB::Char,
            alpha::($elty), A::$array{$elty,3},
            B::$array{$elty,3}, beta::($elty),
            C::$array{$elty,3})

            Base.require_one_based_indexing(A, B, C)
            BLAS.chkstride1(A, B, C)
            @assert size(A, 3) == size(C, 3) || size(A, 3) == 1 "batch size mismatch: A != C"
            @assert size(B, 3) == size(C, 3) || size(B, 3) == 1 "batch size mismatch: B != C"

            m = size(A, transA == 'N' ? 1 : 2)
            ka = size(A, transA == 'N' ? 2 : 1)
            kb = size(B, transB == 'N' ? 1 : 2)
            n = size(B, transB == 'N' ? 2 : 1)

            if m != size(C, 1) || n != size(C, 2) || ka != kb
                throw(DimensionMismatch("A has size ($m,$ka,$(size(A, 3))), B has size ($kb,$n,$(size(B, 3))), C has size $(size(C))"))
            end

            lda = max(1, stride(A, 2))
            ldb = max(1, stride(B, 2))
            ldc = max(1, stride(C, 2))

            strideA = size(A, 3) == 1 ? 0 : stride(A, 3)
            strideB = size(B, 3) == 1 ? 0 : stride(B, 3)
            strideC = stride(C, 3)
            batchCount = size(C, 3)

            NeuralAttentionlib.gemm_strided_batched_impl!(
                transA, transB, m, n, ka,
                alpha, A, lda, strideA,
                B, ldb, strideB, beta,
                C, ldc, strideC, batchCount)

            return C
        end

        function NeuralAttentionlib.gemm_strided_batched(
            transA::Char, transB::Char,
            alpha::($elty), A::$array{$elty,3},
            B::$array{$elty,3})
            C = similar(B, (size(A, transA == 'N' ? 1 : 2), size(B, transB == 'N' ? 2 : 1), max(size(A, 3), size(B, 3))))
            return NeuralAttentionlib.gemm_strided_batched!(transA, transB, alpha, A, B, zero($elty), C)
        end

        function NeuralAttentionlib.gemm_strided_batched(
            transA::Char, transB::Char,
            A::$array{$elty,3},
            B::$array{$elty,3})
            return NeuralAttentionlib.gemm_strided_batched(transA, transB, one($elty), A, B)
        end

        function NeuralAttentionlib.gemm_strided_batched!(
            transA::Char, transB::Char,
            alpha::($elty), A::$array{$elty,N1},
            B::$array{$elty,N2}, beta::($elty),
            C::$array{$elty,N3},
            Ai, Aj, Bi, Bj, Ci, Cj) where {N1,N2,N3}

            # (a1, a2, ..., ai-1, ai, ai+1, ..., aj-1, aj, ..., an)
            #  |______lda______|  |____K/M (Ai)_____|  |___Aj____|
            # (b1, b2, ..., bi-1, bi, bi+1, ..., bj-1, bj, ..., bn)
            #  |______ldb______|  |____K/N (Bi)_____|  |___Bj____|
            # (c1, c2, ..., ci-1, ci, ci+1, ..., cj-1, cj, ..., cn)
            #  |______ldc______|  |____K/N (Ci)_____|  |___Cj____|

            Base.require_one_based_indexing(A, B, C)
            BLAS.chkstride1(A, B, C)

            sa1, sa2, sa3 = NeuralAttentionlib.collapsed_size(A, Ai, Aj)
            sb1, sb2, sb3 = NeuralAttentionlib.collapsed_size(B, Bi, Bj)
            sc1, sc2, sc3 = NeuralAttentionlib.collapsed_size(C, Ci, Cj)

            @assert sa3 == sc3 || sa3 == 1 "batch size mismatch: A != C"
            @assert sb3 == sc3 || sb3 == 1 "batch size mismatch: B != C"

            m = transA == 'N' ? sa1 : sa2
            ka = transA == 'N' ? sa2 : sa1
            kb = transB == 'N' ? sb1 : sb2
            n = transB == 'N' ? sb2 : sb1

            if m != sc1 || n != sc2 || ka != kb
                throw(DimensionMismatch("A has size ($m,$ka,$sa3), B has size ($kb,$n,$sb3), C has size ($sc1, $sc2, $sc3)"))
            end

            lda = max(1, stride(A, N1 - Ai - Aj + 1))
            ldb = max(1, stride(B, N2 - Bi - Bj + 1))
            ldc = max(1, stride(C, N3 - Ci - Cj + 1))

            strideA = sa3 == 1 ? 0 : stride(A, N1 - Aj + 1)
            strideB = sb3 == 1 ? 0 : stride(B, N2 - Bj + 1)
            strideC = stride(C, N3 - Cj + 1)
            batchCount = sc3

            NeuralAttentionlib.gemm_strided_batched_impl!(
                transA, transB, m, n, ka,
                alpha, A, lda, strideA,
                B, ldb, strideB, beta,
                C, ldc, strideC, batchCount)

            return C
        end

        function NeuralAttentionlib.gemm_strided_batched(
            transA::Char, transB::Char,
            alpha::($elty), A::$array{$elty,N1},
            B::$array{$elty,N2},
            Ai, Aj, Bi, Bj) where {N1,N2}

            m = NeuralAttentionlib.noncollapsed_size(A, Ai, Aj, transA == 'N' ? 1 : 2)
            n = NeuralAttentionlib.noncollapsed_size(B, Bi, Bj, transB == 'N' ? 2 : 1)
            sc3 = NeuralAttentionlib.collapsed_size(A, Ai, Aj, 3) > NeuralAttentionlib.collapsed_size(B, Bi, Bj, 3) ?
                  NeuralAttentionlib.noncollapsed_size(A, Ai, Aj, 3) :
                  NeuralAttentionlib.noncollapsed_size(B, Bi, Bj, 3)

            Ci = length(n)
            Cj = length(sc3)
            C = similar(B, (m..., n..., sc3...))
            return NeuralAttentionlib.gemm_strided_batched!(
                transA, transB, alpha, A, B, zero($elty), C, Ai, Aj, Bi, Bj, Ci, Cj)
        end

        function NeuralAttentionlib.gemm_strided_batched(
            transA::Char, transB::Char,
            A::$array{$elty,N1},
            B::$array{$elty,N2},
            Ai, Aj, Bi, Bj) where {N1,N2}
            return NeuralAttentionlib.gemm_strided_batched(transA, transB, one($elty), A, B, Ai, Aj, Bi, Bj)
        end

    end
end

NeuralAttentionlib.check_strided_gemm_type(A::ROCArray{Float16}) = true
AMDGPU.rocconvert
Adapt.adapt(to::AMDGPU.Runtime.Adaptor, m::AbstractArrayMask) =
    Indexer{typeof(m)}(map(Base.Fix1(Adapt.adapt, to), GetIndexer(m).__fields))
Adapt.adapt(to::AMDGPU.Runtime.Adaptor, m::NAlib.FlipMask) = Indexer{typeof(m)}((mask=adapt(to, m.mask),))
Adapt.adapt(to::AMDGPU.Runtime.Adaptor, m::NAlib.CombinedMask) =
    Indexer{typeof(m)}((f=adapt(to, m.f), masks=map(Base.Fix1(adapt, to), m.masks)))
Adapt.adapt(to::AMDGPU.Runtime.Adaptor, m::NAlib.BatchedMask) =
    Indexer{typeof(m)}((mask=adapt(to, m.mask), batch_dim=static(m.batch_dim)))
Adapt.adapt(to::AMDGPU.Runtime.Adaptor, m::NAlib.RepeatMask) = Indexer{typeof(m)}((mask=adapt(to, m.mask), num=m.num))
Adapt.adapt(to::AMDGPU.Runtime.Adaptor, m::NAlib.BiSequenceMask) =
    Indexer{typeof(m)}((q_mask=adapt(to, m.q_mask), k_mask=adapt(to, m.k_mask)))

end
