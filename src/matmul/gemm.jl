using NNlib

import LinearAlgebra
import LinearAlgebra.BLAS
using LinearAlgebra.BLAS: get_num_threads, set_num_threads

const libblas = Base.libblas_name

for (gemm, elty) in NNlib.gemm_datatype_mappings
    @eval begin

        @inline function unsafe_gemm!(
            transA::Char, transB::Char,
            m::Int, n::Int, k::Int,
            alpha::($elty), ptrA::Ptr{$elty}, lda::Int,
            ptrB::Ptr{$elty}, ldb::Int, beta::($elty),
            ptrC::Ptr{$elty}, ldc::Int)

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

                        unsafe_gemm!(transA, transB, m, n, k,
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

                    unsafe_gemm!(transA, transB, m, n, k,
                                 alpha, ptrAi, lda,
                                 ptrBi, ldb, beta,
                                 ptrCi, ldc)

                end
            end
            return nothing
        end

        @inline function gemm_strided_batched_impl!(
            transA::Char, transB::Char,
            m::Int, n::Int, k::Int,
            alpha::($elty), A::AbstractArray{$elty}, lda::Int, strideA::Int,
            B::AbstractArray{$elty}, ldb::Int, strideB::Int, beta::($elty),
            C::AbstractArray{$elty}, ldc::Int, strideC::Int, batchCount::Int)

            ptrA = pointer(A)
            ptrB = pointer(B)
            ptrC = pointer(C)

            GC.@preserve A B C begin
                unsafe_gemm_strided_batched!(
                    transA, transB, m, n, k,
                    alpha, ptrA, lda, strideA,
                    ptrB, ldb, strideB, beta,
                    ptrC, ldc, strideC, batchCount)
            end
            return C
        end

        @inline function gemm_strided_batched!(
            transA::Char, transB::Char,
            alpha::($elty), A::AbstractArray{$elty, 3},
            B::AbstractArray{$elty, 3}, beta::($elty),
            C::AbstractArray{$elty, 3})

            Base.require_one_based_indexing(A, B, C)
            BLAS.chkstride1(A, B, C)
            @assert size(A, 3) == size(C, 3) || size(A, 3) == 1 "batch size mismatch: A != C"
            @assert size(B, 3) == size(C, 3) || size(B, 3) == 1 "batch size mismatch: B != C"

            m = size(A, transA == 'N' ? 1 : 2)
            ka = size(A, transA == 'N' ? 2 : 1)
            kb = size(B, transB == 'N' ? 1 : 2)
            n = size(B, transB == 'N' ? 2 : 1)

            if m != size(C,1) || n != size(C,2) || ka != kb
                throw(DimensionMismatch("A has size ($m,$ka,$(size(A, 3))), B has size ($kb,$n,$(size(B, 3))), C has size $(size(C))"))
            end

            lda = max(1, stride(A,2))
            ldb = max(1, stride(B,2))
            ldc = max(1, stride(C,2))

            strideA = size(A, 3) == 1 ? 0 : stride(A, 3)
            strideB = size(B, 3) == 1 ? 0 : stride(B, 3)
            strideC = stride(C, 3)
            batchCount = size(C, 3)

            gemm_strided_batched_impl!(
                transA, transB, m, n, ka,
                alpha, A, lda, strideA,
                B, ldb, strideB, beta,
                C, ldc, strideC, batchCount)

            return C
        end

        function gemm_strided_batched(
            transA::Char, transB::Char,
            alpha::($elty), A::AbstractArray{$elty, 3},
            B::AbstractArray{$elty, 3})
            C = similar(B, (size(A, transA == 'N' ? 1 : 2), size(B, transB == 'N' ? 2 : 1), max(size(A, 3), size(B, 3))))
            return gemm_strided_batched!(transA, transB, alpha, A, B, zero($elty), C)
        end

        function gemm_strided_batched(
            transA::Char, transB::Char,
            A::AbstractArray{$elty, 3},
            B::AbstractArray{$elty, 3})
            return gemm_strided_batched(transA, transB, one($elty), A, B)
        end

        function gemm_strided_batched!(
            transA::Char, transB::Char,
            alpha::($elty), A::AbstractArray{$elty, N1},
            B::AbstractArray{$elty, N2}, beta::($elty),
            C::AbstractArray{$elty, N3},
            Ai, Aj, Bi, Bj, Ci, Cj) where {N1, N2, N3}

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
                throw(DimensionMismatch("A has size ($m,$ka,$sa3), B has size ($kb,$n,$sb3), C has size ($sc1, $sc2, $sc3)"))
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

        function gemm_strided_batched(
            transA::Char, transB::Char,
            alpha::($elty), A::AbstractArray{$elty, N1},
            B::AbstractArray{$elty, N2},
            Ai, Aj, Bi, Bj) where {N1, N2}

            m = noncollapsed_size(A, Ai, Aj, transA == 'N' ? 1 : 2)
            n = noncollapsed_size(B, Bi, Bj, transB == 'N' ? 2 : 1)
            sc3 = collapsed_size(A, Ai, Aj, 3) > collapsed_size(B, Bi, Bj, 3) ?
                noncollapsed_size(A, Ai, Aj, 3) :
                noncollapsed_size(B, Bi, Bj, 3)

            Ci = length(n)
            Cj = length(sc3)
            C = similar(B, (m..., n..., sc3...))
            return gemm_strided_batched!(transA, transB, alpha, A, B, zero($elty), C, Ai, Aj, Bi, Bj, Ci, Cj)
        end

        function gemm_strided_batched(
            transA::Char, transB::Char,
            A::AbstractArray{$elty, N1},
            B::AbstractArray{$elty, N2},
            Ai, Aj, Bi, Bj) where {N1, N2}
            return gemm_strided_batched(transA, transB, one($elty), A, B, Ai, Aj, Bi, Bj)
        end

    end
end
