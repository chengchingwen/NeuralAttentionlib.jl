using NNlib

import LinearAlgebra
import LinearAlgebra.BLAS

for (gemm, elty) in NNlib.gemm_datatype_mappings
    @eval begin

        @inline function unsafe_gemm!(
            transA::Char, transB::Char,
            m::Int, n::Int, k::Int,
            alpha::($elty), ptrA::Ptr{$elty}, lda::Int,
            ptrB::Ptr{$elty}, ldb::Int, beta::($elty),
            ptrC::Ptr{$elty}, ldc::Int)

            ccall((BLAS.@blasfunc($gemm), BLAS.libblas), Nothing,
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

            strA = strideA * sizeof($elty)
            strB = strideB * sizeof($elty)
            strC = strideC * sizeof($elty)

            for i = 1:batchCount
                unsafe_gemm!(transA, transB, m, n, k,
                             alpha, ptrA, lda,
                             ptrB, ldb, beta,
                             ptrC, ldc)

                ptrA += strA
                ptrB += strB
                ptrC += strC

            end
            return nothing
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

            ptrA = pointer(A)
            ptrB = pointer(B)
            ptrC = pointer(C)

            GC.@preserve A B C begin
                unsafe_gemm_strided_batched!(
                    transA, transB, m, n, ka,
                    alpha, ptrA, lda, strideA,
                    ptrB, ldb, strideB, beta,
                    ptrC, ldc, strideC, batchCount)
            end
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

            # (A1, A2, ..., Ai-1, Ai, Ai+1, ..., Aj-1, Aj, ..., An)
            #  |______lda______|  |_______K/M_______|  |__batch__|
            # (B1, B2, ..., Bi-1, Bi, Bi+1, ..., Bj-1, Bj, ..., Bn)
            #  |______ldb______|  |_______K/N_______|  |__batch__|
            # (C1, C2, ..., Ci-1, Ci, Ci+1, ..., Cj-1, Cj, ..., Cn)
            #  |______ldc______|  |_______K/N_______|  |__batch__|

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
            
            lda = max(1, stride(A, Ai))
            ldb = max(1, stride(B, Bi))
            ldc = max(1, stride(C, Ci))

            strideA = sa3 == 1 ? 0 : stride(A, Aj)
            strideB = sb3 == 1 ? 0 : stride(B, Bj)
            strideC = stride(C, Cj)
            batchCount = sc3
            
            ptrA = pointer(A)
            ptrB = pointer(B)
            ptrC = pointer(C)

            GC.@preserve A B C begin
                unsafe_gemm_strided_batched!(
                    transA, transB, m, n, ka,
                    alpha, ptrA, lda, strideA,
                    ptrB, ldb, strideB, beta,
                    ptrC, ldc, strideC, batchCount)
            end
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

            Ci = length(m) + 1
            Cj = Ci + length(n)
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
