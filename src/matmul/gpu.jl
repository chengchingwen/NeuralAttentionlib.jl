using CUDA
import CUDA.CUBLAS

for (fname, elty) in
    ((:cublasDgemmStridedBatched,:Float64),
     (:cublasSgemmStridedBatched,:Float32),
     (:cublasHgemmStridedBatched,:Float16),
     (:cublasZgemmStridedBatched,:ComplexF64),
     (:cublasCgemmStridedBatched,:ComplexF32))
    @eval begin

        @inline function gemm_strided_batched!(
            transA::Char, transB::Char,
            alpha::($elty), A::CuArray{$elty, 3},
            B::CuArray{$elty, 3}, beta::($elty),
            C::CuArray{$elty, 3})
            return CUBLAS.gemm_strided_batched!(transA, transB, alpha, A, B, beta, C)
        end

        @inline function gemm_strided_batched_impl!(
            transA::Char, transB::Char,
            m::Int, n::Int, k::Int,
            alpha::($elty), A::CuArray{$elty}, lda::Int, strideA::Int,
            B::CuArray{$elty}, ldb::Int, strideB::Int, beta::($elty),
            C::CuArray{$elty}, ldc::Int, strideC::Int, batchCount::Int)

            CUBLAS.$fname(CUBLAS.handle(),
                          transA, transB, m, n, k,
                          alpha, A, lda, strideA,
                          B, ldb, strideB, beta,
                          C, ldc, strideC, batchCount)

            return C
        end

    end
end
