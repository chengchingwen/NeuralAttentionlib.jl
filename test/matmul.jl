@testset "matmul" begin
    a = randn(7,6,5,4,3,2)
    b = randn(42, 20, 6)
    c = randn(42, 60, 2)

    @test matmul(batched_transpose(CollapsedDimArray(a, 3, 5)), b) ≈ batched_mul(batched_transpose(collapseddim(a, 3, 5)), b)
    

end
