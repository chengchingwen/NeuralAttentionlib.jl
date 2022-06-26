@testset "matmul" begin
    using NeuralAttentionlib.Matmul

    ≂(a, b) = (a ≈ b) && size(a) == size(b)
    function matmul_test(x, y, s)
        cx = x isa CollapsedDimArray ? collapseddim(x) : x
        cy = y isa CollapsedDimArray ? collapseddim(y) : y
        return matmul(x, y, s) ≂ batched_mul(cx, cy) .* s
    end
    uwcs(x) = size(unwrap_collapse(x))

    @testset "API" begin
        a = randn(ComplexF64, 10)
        b = randn(ComplexF64, 1, 10)
        c = randn(ComplexF64, 10, 1)
        d = randn(ComplexF64, 10, 1, 1)
        e = randn(ComplexF64, 1, 10, 1)

        @testset "MatOrVec" begin
            @test matmul(a, b) ≂ a * b
            @test matmul(b, a) ≂ b * a
            @test matmul(c, b) ≂ c * b
            @test matmul(b, c) ≂ b * c

            @test matmul(transpose(a), a) ≂ transpose(a) * a
            @test matmul(adjoint(a), a) ≂ adjoint(a) * a
            @test matmul(a, transpose(a)) ≂ a * transpose(a)
            @test matmul(a, adjoint(a)) ≂ a * adjoint(a)

            @test matmul(transpose(b), b) ≂ transpose(b) * b
            @test matmul(adjoint(b), b) ≂ adjoint(b) * b
            @test matmul(b, transpose(b)) ≂ b * transpose(b)
            @test matmul(b, adjoint(b)) ≂ b * adjoint(b)

            @test matmul(transpose(a), transpose(b)) ≂ transpose(a) * transpose(b)
            @test matmul(adjoint(a), transpose(b)) ≂ adjoint(a) * transpose(b)
            @test matmul(transpose(a), adjoint(b)) ≂ transpose(a) * adjoint(b)
            @test matmul(adjoint(a), adjoint(b)) ≂ adjoint(a) * adjoint(b)

            @test matmul(transpose(c), transpose(b)) ≂ transpose(c) * transpose(b)
            @test matmul(adjoint(c), transpose(b)) ≂ adjoint(c) * transpose(b)
            @test matmul(transpose(c), adjoint(b)) ≂ transpose(c) * adjoint(b)
            @test matmul(adjoint(c), adjoint(b)) ≂ adjoint(c) * adjoint(b)
        end

        @testset "MatOrVec and Tensor" begin
            # no adjortrans
            @test matmul(a, e) ≂ reshape(a * reshape(e, 1, 10), Val(3))
            @test matmul(c, e) ≂ reshape(c * reshape(e, 1, 10), Val(3))
            @test matmul(b, d) ≂ reshape(b * reshape(d, 10, 1), Val(3))
            @test matmul(e, a) ≂ reshape(reshape(e, 1, 10) * a, Val(3))
            @test matmul(e, c) ≂ reshape(reshape(e, 1, 10) * c, Val(3))
            @test matmul(d, b) ≂ reshape(reshape(d, 10, 1) * b, Val(3))

            # matorvec adjortrans
            @test matmul(transpose(a), d) ≂ reshape(transpose(a) * reshape(d, 10, 1), Val(3))
            @test matmul(transpose(c), d) ≂ reshape(transpose(c) * reshape(d, 10, 1), Val(3))
            @test matmul(transpose(b), e) ≂ reshape(transpose(b) * reshape(e, 1, 10), Val(3))
            @test matmul(d, transpose(a)) ≂ reshape(reshape(d, 10, 1) * transpose(a), Val(3))
            @test matmul(d, transpose(c)) ≂ reshape(reshape(d, 10, 1) * transpose(c), Val(3))
            @test matmul(e, transpose(b)) ≂ reshape(reshape(e, 1, 10) * transpose(b), Val(3))
            @test matmul(adjoint(a), d) ≂ reshape(adjoint(a) * reshape(d, 10, 1), Val(3))
            @test matmul(adjoint(c), d) ≂ reshape(adjoint(c) * reshape(d, 10, 1), Val(3))
            @test matmul(adjoint(b), e) ≂ reshape(adjoint(b) * reshape(e, 1, 10), Val(3))
            @test matmul(d, adjoint(a)) ≂ reshape(reshape(d, 10, 1) * adjoint(a), Val(3))
            @test matmul(d, adjoint(c)) ≂ reshape(reshape(d, 10, 1) * adjoint(c), Val(3))
            @test matmul(e, adjoint(b)) ≂ reshape(reshape(e, 1, 10) * adjoint(b), Val(3))

            # tensor adjortrans
            @test matmul(a, batched_transpose(d)) ≂ reshape(a * transpose(reshape(d, 10, 1)), Val(3))
            @test matmul(c, batched_transpose(d)) ≂ reshape(c * transpose(reshape(d, 10, 1)), Val(3))
            @test matmul(b, batched_transpose(e)) ≂ reshape(b * transpose(reshape(e, 1, 10)), Val(3))
            @test matmul(batched_transpose(d), a) ≂ reshape(transpose(reshape(d, 10, 1)) * a, Val(3))
            @test matmul(batched_transpose(d), c) ≂ reshape(transpose(reshape(d, 10, 1)) * c, Val(3))
            @test matmul(batched_transpose(e), b) ≂ reshape(transpose(reshape(e, 1, 10)) * b, Val(3))
            @test matmul(a, batched_adjoint(d)) ≂ reshape(a * adjoint(reshape(d, 10, 1)), Val(3))
            @test matmul(c, batched_adjoint(d)) ≂ reshape(c * adjoint(reshape(d, 10, 1)), Val(3))
            @test matmul(b, batched_adjoint(e)) ≂ reshape(b * adjoint(reshape(e, 1, 10)), Val(3))
            @test matmul(batched_adjoint(d), a) ≂ reshape(adjoint(reshape(d, 10, 1)) * a, Val(3))
            @test matmul(batched_adjoint(d), c) ≂ reshape(adjoint(reshape(d, 10, 1)) * c, Val(3))
            @test matmul(batched_adjoint(e), b) ≂ reshape(adjoint(reshape(e, 1, 10)) * b, Val(3))

            # both adjortrans
            @test matmul(transpose(a), batched_transpose(e)) ≂ reshape(transpose(a) * transpose(reshape(e, 1, 10)), Val(3))
            @test matmul(transpose(c), batched_transpose(e)) ≂ reshape(transpose(c) * transpose(reshape(e, 1, 10)), Val(3))
            @test matmul(transpose(b), batched_transpose(d)) ≂ reshape(transpose(b) * transpose(reshape(d, 10, 1)), Val(3))
            @test matmul(batched_transpose(e), transpose(a)) ≂ reshape(transpose(reshape(e, 1, 10)) * transpose(a), Val(3))
            @test matmul(batched_transpose(e), transpose(c)) ≂ reshape(transpose(reshape(e, 1, 10)) * transpose(c), Val(3))
            @test matmul(batched_transpose(d), transpose(b)) ≂ reshape(transpose(reshape(d, 10, 1)) * transpose(b), Val(3))
            @test matmul(transpose(a), batched_adjoint(e)) ≂ reshape(transpose(a) * adjoint(reshape(e, 1, 10)), Val(3))
            @test matmul(transpose(c), batched_adjoint(e)) ≂ reshape(transpose(c) * adjoint(reshape(e, 1, 10)), Val(3))
            @test matmul(transpose(b), batched_adjoint(d)) ≂ reshape(transpose(b) * adjoint(reshape(d, 10, 1)), Val(3))
            @test matmul(batched_adjoint(e), transpose(a)) ≂ reshape(adjoint(reshape(e, 1, 10)) * transpose(a), Val(3))
            @test matmul(batched_adjoint(e), transpose(c)) ≂ reshape(adjoint(reshape(e, 1, 10)) * transpose(c), Val(3))
            @test matmul(batched_adjoint(d), transpose(b)) ≂ reshape(adjoint(reshape(d, 10, 1)) * transpose(b), Val(3))
            @test matmul(adjoint(a), batched_transpose(e)) ≂ reshape(adjoint(a) * transpose(reshape(e, 1, 10)), Val(3))
            @test matmul(adjoint(c), batched_transpose(e)) ≂ reshape(adjoint(c) * transpose(reshape(e, 1, 10)), Val(3))
            @test matmul(adjoint(b), batched_transpose(d)) ≂ reshape(adjoint(b) * transpose(reshape(d, 10, 1)), Val(3))
            @test matmul(batched_transpose(e), adjoint(a)) ≂ reshape(transpose(reshape(e, 1, 10)) * adjoint(a), Val(3))
            @test matmul(batched_transpose(e), adjoint(c)) ≂ reshape(transpose(reshape(e, 1, 10)) * adjoint(c), Val(3))
            @test matmul(batched_transpose(d), adjoint(b)) ≂ reshape(transpose(reshape(d, 10, 1)) * adjoint(b), Val(3))
            @test matmul(adjoint(a), batched_adjoint(e)) ≂ reshape(adjoint(a) * adjoint(reshape(e, 1, 10)), Val(3))
            @test matmul(adjoint(c), batched_adjoint(e)) ≂ reshape(adjoint(c) * adjoint(reshape(e, 1, 10)), Val(3))
            @test matmul(adjoint(b), batched_adjoint(d)) ≂ reshape(adjoint(b) * adjoint(reshape(d, 10, 1)), Val(3))
            @test matmul(batched_adjoint(e), adjoint(a)) ≂ reshape(adjoint(reshape(e, 1, 10)) * adjoint(a), Val(3))
            @test matmul(batched_adjoint(e), adjoint(c)) ≂ reshape(adjoint(reshape(e, 1, 10)) * adjoint(c), Val(3))
            @test matmul(batched_adjoint(d), adjoint(b)) ≂ reshape(adjoint(reshape(d, 10, 1)) * adjoint(b), Val(3))

        end
    end

    for elt in (Float64, Float32, ComplexF64, ComplexF32)
        @testset "gemm_strided $elt" begin
            a = randn(elt, 7, 6, 5, 4, 3, 2)
            b = randn(elt, 42, 20, 6)
            bt = randn(elt, 20, 42, 6)
            c = randn(elt, 42, 60, 2)
            ct = randn(elt, 60, 42, 2)
            s = rand(elt) + 1

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
    end

    @testset "AD" begin
        test_rrule(matmul, randn(7,6,5), randn(6, 2), randn())
        test_rrule(matmul, randn(7,6,5,4), randn(6), randn())
        test_rrule(matmul, CollapsedDimArray(randn(2,2,2,2,2,3), 4, 6), batched_transpose(randn(5,4,3)), randn(); check_inferred=false)
    end

end
