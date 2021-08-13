@testset "matmul" begin
    using NeuralAttentionlib.Matmul

    function matmul_test(x, y, s)
        cx = x isa CollapsedDimArray ? collapseddim(x) : x
        cy = y isa CollapsedDimArray ? collapseddim(y) : y
        return matmul(x, y, s) ≈ batched_mul(cx, cy) .* s
    end
    uwcs(x) = size(unwrap_collapse(x))

    @testset "gemm_strided Float64" begin
        a = randn(7, 6, 5, 4, 3, 2)
        b = randn(42, 20, 6)
        bt = randn(20, 42, 6)
        c = randn(42, 60, 2)
        ct = randn(60, 42, 2)
        s = rand() + 1

        ca1 = CollapsedDimArray(a, 3, 5)
        ca2 = CollapsedDimArray(a, 3, 6)

        @test matmul_test(ca1, bt, s)
        @test matmul_test(ca1, batched_transpose(b), s)
        @test matmul_test(bt, ca1, s)
        @test matmul_test(batched_transpose(b), ca1, s)
        @test matmul_test(batched_transpose(ca1), b, s)
        @test matmul_test(batched_transpose(ca1), batched_transpose(bt), s)
        @test matmul_test(b, batched_transpose(ca1), s)
        @test matmul_test(batched_transpose(bt), batched_transpose(ca1), s)
        @test matmul_test(ca1, batched_adjoint(b), s)
        @test matmul_test(batched_adjoint(b), ca1, s)
        @test matmul_test(batched_adjoint(ca1), b, s)
        @test matmul_test(batched_adjoint(ca1), batched_adjoint(bt), s)
        @test matmul_test(b, batched_adjoint(ca1), s)
        @test matmul_test(batched_adjoint(bt), batched_adjoint(ca1), s)

        @test uwcs(matmul(ca1, bt)) == (7, 6, 42, 6)
        @test uwcs(matmul(bt, ca1)) == (20, 5, 4, 3, 2)
        @test uwcs(matmul(batched_transpose(ca1), b)) == (5, 4, 20, 6)
        @test uwcs(matmul(batched_transpose(b), ca1)) == (20, 5, 4, 3, 2)

        @test matmul_test(ca2, ct, s)
        @test matmul_test(ca2, batched_transpose(c), s)
        @test matmul_test(ct, ca2, s)
        @test matmul_test(batched_transpose(c), ca2, s)
        @test matmul_test(batched_transpose(ca2), c, s)
        @test matmul_test(batched_transpose(ca2), batched_transpose(ct), s)
        @test matmul_test(c, batched_transpose(ca2), s)
        @test matmul_test(batched_transpose(ct), batched_transpose(ca2), s)

        @test uwcs(matmul(ca2, ct)) == (7, 6, 42, 2)
        @test uwcs(matmul(ct, ca2)) == (60, 5, 4, 3, 2)
        @test uwcs(matmul(batched_transpose(ca2), c)) == (5, 4, 3, 60, 2)
        @test uwcs(matmul(batched_transpose(c), ca2)) == (60, 5, 4, 3, 2)

    end

    @testset "gemm_strided Float32" begin
        a = randn(Float32, 7, 6, 5, 4, 3, 2)
        b = randn(Float32, 42, 20, 6)
        bt = randn(Float32, 20, 42, 6)
        c = randn(Float32, 42, 60, 2)
        ct = randn(Float32, 60, 42, 2)
        s = rand(Float32) + 1

        ca1 = CollapsedDimArray(a, 3, 5)
        ca2 = CollapsedDimArray(a, 3, 6)

        @test matmul_test(ca1, bt, s)
        @test matmul_test(ca1, batched_transpose(b), s)
        @test matmul_test(bt, ca1, s)
        @test matmul_test(batched_transpose(b), ca1, s)
        @test matmul_test(batched_transpose(ca1), b, s)
        @test matmul_test(batched_transpose(ca1), batched_transpose(bt), s)
        @test matmul_test(b, batched_transpose(ca1), s)
        @test matmul_test(batched_transpose(bt), batched_transpose(ca1), s)
        @test matmul_test(ca1, batched_adjoint(b), s)
        @test matmul_test(batched_adjoint(b), ca1, s)
        @test matmul_test(batched_adjoint(ca1), b, s)
        @test matmul_test(batched_adjoint(ca1), batched_adjoint(bt), s)
        @test matmul_test(b, batched_adjoint(ca1), s)
        @test matmul_test(batched_adjoint(bt), batched_adjoint(ca1), s)

        @test matmul_test(ca2, ct, s)
        @test matmul_test(ca2, batched_transpose(c), s)
        @test matmul_test(ct, ca2, s)
        @test matmul_test(batched_transpose(c), ca2, s)
        @test matmul_test(batched_transpose(ca2), c, s)
        @test matmul_test(batched_transpose(ca2), batched_transpose(ct), s)
        @test matmul_test(c, batched_transpose(ca2), s)
        @test matmul_test(batched_transpose(ct), batched_transpose(ca2), s)

    end

    @testset "gemm_strided ComplexF64" begin
        a = randn(ComplexF64, 7, 6, 5, 4, 3, 2)
        b = randn(ComplexF64, 42, 20, 6)
        bt = randn(ComplexF64, 20, 42, 6)
        c = randn(ComplexF64, 42, 60, 2)
        ct = randn(ComplexF64, 60, 42, 2)
        s = rand(ComplexF64) + 1

        ca1 = CollapsedDimArray(a, 3, 5)
        ca2 = CollapsedDimArray(a, 3, 6)

        @test matmul_test(ca1, bt, s)
        @test matmul_test(ca1, batched_transpose(b), s)
        @test matmul_test(bt, ca1, s)
        @test matmul_test(batched_transpose(b), ca1, s)
        @test matmul_test(batched_transpose(ca1), b, s)
        @test matmul_test(batched_transpose(ca1), batched_transpose(bt), s)
        @test matmul_test(b, batched_transpose(ca1), s)
        @test matmul_test(batched_transpose(bt), batched_transpose(ca1), s)
        @test matmul_test(ca1, batched_adjoint(b), s)
        @test matmul_test(batched_adjoint(b), ca1, s)
        @test matmul_test(batched_adjoint(ca1), b, s)
        @test matmul_test(batched_adjoint(ca1), batched_adjoint(bt), s)
        @test matmul_test(b, batched_adjoint(ca1), s)
        @test matmul_test(batched_adjoint(bt), batched_adjoint(ca1), s)

        @test matmul_test(ca2, ct, s)
        @test matmul_test(ca2, batched_transpose(c), s)
        @test matmul_test(ct, ca2, s)
        @test matmul_test(batched_transpose(c), ca2, s)
        @test matmul_test(batched_transpose(ca2), c, s)
        @test matmul_test(batched_transpose(ca2), batched_transpose(ct), s)
        @test matmul_test(c, batched_transpose(ca2), s)
        @test matmul_test(batched_transpose(ct), batched_transpose(ca2), s)

    end

    @testset "gemm_strided ComplexF32" begin
        a = randn(ComplexF32, 7, 6, 5, 4, 3, 2)
        b = randn(ComplexF32, 42, 20, 6)
        bt = randn(ComplexF32, 20, 42, 6)
        c = randn(ComplexF32, 42, 60, 2)
        ct = randn(ComplexF32, 60, 42, 2)
        s = rand(ComplexF32) + 1

        ca1 = CollapsedDimArray(a, 3, 5)
        ca2 = CollapsedDimArray(a, 3, 6)

        @test matmul_test(ca1, bt, s)
        @test matmul_test(ca1, batched_transpose(b), s)
        @test matmul_test(bt, ca1, s)
        @test matmul_test(batched_transpose(b), ca1, s)
        @test matmul_test(batched_transpose(ca1), b, s)
        @test matmul_test(batched_transpose(ca1), batched_transpose(bt), s)
        @test matmul_test(b, batched_transpose(ca1), s)
        @test matmul_test(batched_transpose(bt), batched_transpose(ca1), s)
        @test matmul_test(ca1, batched_adjoint(b), s)
        @test matmul_test(batched_adjoint(b), ca1, s)
        @test matmul_test(batched_adjoint(ca1), b, s)
        @test matmul_test(batched_adjoint(ca1), batched_adjoint(bt), s)
        @test matmul_test(b, batched_adjoint(ca1), s)
        @test matmul_test(batched_adjoint(bt), batched_adjoint(ca1), s)

        @test matmul_test(ca2, ct, s)
        @test matmul_test(ca2, batched_transpose(c), s)
        @test matmul_test(ct, ca2, s)
        @test matmul_test(batched_transpose(c), ca2, s)
        @test matmul_test(batched_transpose(ca2), c, s)
        @test matmul_test(batched_transpose(ca2), batched_transpose(ct), s)
        @test matmul_test(c, batched_transpose(ca2), s)
        @test matmul_test(batched_transpose(ct), batched_transpose(ca2), s)

    end

    @testset "AD" begin
        test_rrule(matmul, randn(7,6,5), randn(6, 2), randn())
        test_rrule(matmul, randn(7,6,5,4), randn(6), randn())
        test_rrule(matmul, CollapsedDimArray(randn(2,2,2,2,2,3), 4, 6), batched_transpose(randn(5,4,3)), randn(); check_inferred=false)
    end

end
