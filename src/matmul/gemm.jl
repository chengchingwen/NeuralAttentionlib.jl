using NNlib

import LinearAlgebra
import LinearAlgebra.BLAS
using LinearAlgebra.BLAS: get_num_threads, set_num_threads

const libblas = Base.libblas_name

struct TypedPtr{T, ET, P}
    ptr::P
    TypedPtr{T}(ptr) where T = new{T, eltype(ptr), typeof(ptr)}(ptr)
end
@inline typedpointer(arr::AbstractArray) = TypedPtr{ptrtypetag(arr)}(pointer(arr))
@inline ptrtypetag(arr) = Array

@inline function gemm_strided_batched_impl!(
    transA::Char, transB::Char,
    m::Integer, n::Integer, k::Integer,
    alpha::ET, A::T1, lda::Integer, strideA::Integer,
    B::T2, ldb::Integer, strideB::Integer, beta::ET,
    C::T3, ldc::Integer, strideC::Integer, batchCount::Integer
) where {ET, N1, N2, N3, T1 <: AbstractArray{ET, N1}, T2 <: AbstractArray{ET, N2}, T3 <: AbstractArray{ET, N3}}
    ptrA = typedpointer(A)
    ptrB = typedpointer(B)
    ptrC = typedpointer(C)
    GC.@preserve A B C begin
        unsafe_gemm_strided_batched!(
            transA, transB, m, n, k,
            alpha, ptrA, lda, strideA,
            ptrB, ldb, strideB, beta,
            ptrC, ldc, strideC, batchCount)
    end
    return C
end

for (gemm, elty) in NNlib.gemm_datatype_mappings
    @eval begin
        @inline function unsafe_gemm!(
            transA::Char, transB::Char,
            m::Integer, n::Integer, k::Integer,
            alpha::($elty), tptrA::TypedPtr{Array, $elty}, lda::Integer,
            tptrB::TypedPtr{Array, $elty}, ldb::Integer, beta::($elty),
            tptrC::TypedPtr{Array, $elty}, ldc::Integer)

            ptrA = tptrA.ptr
            ptrB = tptrB.ptr
            ptrC = tptrC.ptr
            ccall((BLAS.@blasfunc($gemm), libblas), Nothing,
                  (Ref{UInt8}, Ref{UInt8}, Ref{BLAS.BlasInt}, Ref{BLAS.BlasInt}, Ref{BLAS.BlasInt},
                   Ref{$elty}, Ptr{$elty}, Ref{BLAS.BlasInt},
                   Ptr{$elty}, Ref{BLAS.BlasInt}, Ref{$elty},
                   Ptr{$elty}, Ref{BLAS.BlasInt}),
                  transA, transB, m, n, k,
                  alpha, ptrA, lda,
                  ptrB, ldb, beta,
                  ptrC, ldc)
            return nothing
        end

        @inline function unsafe_gemm_strided_batched!(
            transA::Char, transB::Char,
            m::Integer, n::Integer, k::Integer,
            alpha::($elty), tptrA::TypedPtr{Array, $elty}, lda::Integer, strideA::Integer,
            tptrB::TypedPtr{Array, $elty}, ldb::Integer, strideB::Integer, beta::($elty),
            tptrC::TypedPtr{Array, $elty}, ldc::Integer, strideC::Integer, batchCount::Integer)

            ptrA = tptrA.ptr
            ptrB = tptrB.ptr
            ptrC = tptrC.ptr
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
                        tptrAi = TypedPtr{Array}(ptrAi)
                        tptrBi = TypedPtr{Array}(ptrBi)
                        tptrCi = TypedPtr{Array}(ptrCi)

                        unsafe_gemm!(transA, transB, m, n, k,
                                     alpha, tptrAi, lda,
                                     tptrBi, ldb, beta,
                                     tptrCi, ldc)
                    end
                end
                set_num_threads(old_threads)
            else
                for i = 1:batchCount
                    ptrAi = ptrA + (i - 1) * strideA * sizeof($elty)
                    ptrBi = ptrB + (i - 1) * strideB * sizeof($elty)
                    ptrCi = ptrC + (i - 1) * strideC * sizeof($elty)
                    tptrAi = TypedPtr{Array}(ptrAi)
                    tptrBi = TypedPtr{Array}(ptrBi)
                    tptrCi = TypedPtr{Array}(ptrCi)

                    unsafe_gemm!(transA, transB, m, n, k,
                                 alpha, tptrAi, lda,
                                 tptrBi, ldb, beta,
                                 tptrCi, ldc)
                end
            end
            return nothing
        end
    end
end

gemm_strided_batched(
    transA::Char, transB::Char, A::T1, B::T2) where {ET, T1 <: AbstractArray{ET, 3}, T2 <: AbstractArray{ET, 3}} =
        gemm_strided_batched(transA, transB, A, B, static(1), static(1), static(1), static(1))
gemm_strided_batched(
    transA::Char, transB::Char, alpha::ET, A::T1, B::T2
) where {ET, T1 <: AbstractArray{ET, 3}, T2 <: AbstractArray{ET, 3}} =
    gemm_strided_batched(transA, transB, alpha, A, B, static(1), static(1), static(1), static(1))
gemm_strided_batched!(
    transA::Char, transB::Char, alpha::ET, A::T1, B::T2, beta::ET, C::T3
) where {ET, T1 <: AbstractArray{ET, 3}, T2 <: AbstractArray{ET, 3}, T3 <: AbstractArray{ET, 3}} =
    gemm_strided_batched!(
        transA, transB, alpha, A, B, beta, C, static(1), static(1), static(1), static(1), static(1), static(1))

gemm_strided_batched(
    transA::Char, transB::Char, A::T1, B::T2, Ai, Aj, Bi, Bj
) where {ET, N1, N2, T1 <: AbstractArray{ET, N1}, T2 <: AbstractArray{ET, N2}} =
    gemm_strided_batched(transA, transB, one(ET), A, B, Ai, Aj, Bi, Bj)
function gemm_strided_batched(
    transA::Char, transB::Char, alpha::ET, A::T1, B::T2, Ai, Aj, Bi, Bj
) where {ET, N1, N2, T1 <: AbstractArray{ET, N1}, T2 <: AbstractArray{ET, N2}}
    m = noncollapsed_size(A, Ai, Aj, transA == 'N' ? 1 : 2)
    n = noncollapsed_size(B, Bi, Bj, transB == 'N' ? 2 : 1)
    sc3 = collapsed_size(A, Ai, Aj, 3) > collapsed_size(B, Bi, Bj, 3) ?
        noncollapsed_size(A, Ai, Aj, 3) :
        noncollapsed_size(B, Bi, Bj, 3)
    Ci = length(n)
    Cj = length(sc3)
    C = similar(B, (m..., n..., sc3...))
    return gemm_strided_batched!(transA, transB, alpha, A, B, zero(ET), C, Ai, Aj, Bi, Bj, Ci, Cj)
end
function gemm_strided_batched!(
    transA::Char, transB::Char,
    alpha::ET, A::T1, B::T2, beta::ET, C::T3, Ai, Aj, Bi, Bj, Ci, Cj
) where {ET, N1, N2, N3, T1 <: AbstractArray{ET, N1}, T2 <: AbstractArray{ET, N2}, T3 <: AbstractArray{ET, N3}}
    # (a1, a2, ..., ai-1, ai, ai+1, ..., aj-1, aj, ..., an)
    #  |______lda______|  |____K/M (Ai)_____|  |___Aj____|
    # (b1, b2, ..., bi-1, bi, bi+1, ..., bj-1, bj, ..., bn)
    #  |______ldb______|  |____K/N (Bi)_____|  |___Bj____|
    # (c1, c2, ..., ci-1, ci, ci+1, ..., cj-1, cj, ..., cn)
    #  |______ldc______|  |____K/N (Ci)_____|  |___Cj____|
    Base.require_one_based_indexing(A, B, C)
    BLAS.chkstride1(A, B, C)

    sa1, sa2, sa3 = collapsed_size(A, Ai, Aj)
    sb1, sb2, sb3 = collapsed_size(B, Bi, Bj)
    sc1, sc2, sc3 = collapsed_size(C, Ci, Cj)
    @assert sa3 == sc3 || sa3 == 1 "batch size mismatch: A != C"
    @assert sb3 == sc3 || sb3 == 1 "batch size mismatch: B != C"

    m = transA == 'N' ? sa1 : sa2
    ka = transA == 'N' ? sa2 : sa1
    kb = transB == 'N' ? sb1 : sb2
    n = transB == 'N' ? sb2 : sb1
    if m != sc1 || n != sc2 || ka != kb
        throw(DimensionMismatch("A has size ($m, $ka, $sa3), B has size ($kb, $n, $sb3), C has size ($sc1, $sc2, $sc3)"))
    end

    lda = max(1, stride(A, N1 - Ai - Aj + 1))
    ldb = max(1, stride(B, N2 - Bi - Bj + 1))
    ldc = max(1, stride(C, N3 - Ci - Cj + 1))

    strideA = sa3 == 1 ? 0 : stride(A, N1 - Aj + 1)
    strideB = sb3 == 1 ? 0 : stride(B, N2 - Bj + 1)
    strideC = stride(C, N3 - Cj + 1)
    batchCount = sc3

    gemm_strided_batched_impl!(
        transA, transB, m, n, ka,
        alpha, A, lda, strideA,
        B, ldb, strideB, beta,
        C, ldc, strideC, batchCount)
    return C
end
