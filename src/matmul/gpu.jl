import CUDA.CUBLAS

for (fname, elty) in
    ((:cublasDgemmStridedBatched,:Float64),
     (:cublasSgemmStridedBatched,:Float32),
     (:cublasHgemmStridedBatched,:Float16),
     (:cublasZgemmStridedBatched,:ComplexF64),
     (:cublasCgemmStridedBatched,:ComplexF32))
    @eval begin

        @inline function unsafe_gemm_strided_batched!(
            transA::Char, transB::Char,
            m::Int, n::Int, k::Int,
            alpha::($elty), ptrA::CuPtr{$elty}, lda::Int, strideA::Int,
            ptrB::CuPtr{$elty}, ldb::Int, strideB::Int, beta::($elty),
            ptrC::CuPtr{$elty}, ldc::Int, strideC::Int, batchCount::Int)

            CUBLAS.$fname(CUBLAS.handle(),
                          transA, transB, m, n, k,
                          alpha, ptrA, lda, strideA,
                          ptrB, ldb, strideB, beta,
                          ptrC, ldc, strideC, batchCount)
            return nothing
        end

    end
end
