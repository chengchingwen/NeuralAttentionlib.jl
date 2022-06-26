@init @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
    using .CUDA
    import .CUDA.CUBLAS

    for (fname, elty) in
        ((:cublasDgemmStridedBatched,:Float64),
         (:cublasSgemmStridedBatched,:Float32),
         (:cublasHgemmStridedBatched,:Float16),
         (:cublasZgemmStridedBatched,:ComplexF64),
         (:cublasCgemmStridedBatched,:ComplexF32))
        @eval begin

            @inline function gemm_strided_batched!(
                transA::Char, transB::Char,
                alpha::($elty), A::StridedCuArray{$elty, 3},
                B::StridedCuArray{$elty, 3}, beta::($elty),
                C::StridedCuArray{$elty, 3})
                return CUBLAS.gemm_strided_batched!(transA, transB, alpha, A, B, beta, C)
            end

            @inline function gemm_strided_batched_impl!(
                transA::Char, transB::Char,
                m::Int, n::Int, k::Int,
                alpha::($elty), A::StridedCuArray{$elty}, lda::Int, strideA::Int,
                B::StridedCuArray{$elty}, ldb::Int, strideB::Int, beta::($elty),
                C::StridedCuArray{$elty}, ldc::Int, strideC::Int, batchCount::Int)

                CUBLAS.$fname(CUBLAS.handle(),
                              transA, transB, m, n, k,
                              alpha, A, lda, strideA,
                              B, ldb, strideB, beta,
                              C, ldc, strideC, batchCount)

                return C
            end

        end
    end

end
