@testset "CollapsedDim" begin
    using NeuralAttentionlib: collapsed_size, noncollapsed_size
    x = randn(7,6,5,4,3,2)

    @test collapsed_size(x, 3, 5) == (42, 20, 6)
    @test collapsed_size(x, 2, 3) == (7, 6, 120)
    @test collapsed_size(x, 4, 8) == (210, 24, 1)
    @test noncollapsed_size(x, 3, 5, 1) == (7, 6)
    @test noncollapsed_size(x, 3, 5, 2) == (5, 4)
    @test noncollapsed_size(x, 3, 5, 3) == (3, 2)
    @test noncollapsed_size(x, 2, 3, 1) == (7,)
    @test noncollapsed_size(x, 2, 3, 2) == (6,)
    @test noncollapsed_size(x, 2, 3, 3) == (5, 4, 3, 2)
    @test noncollapsed_size(x, 4, 8, 1) == (7, 6, 5)
    @test noncollapsed_size(x, 4, 8, 2) == (4, 3, 2, 1)
    @test noncollapsed_size(x, 4, 8, 3) == (1,)

    @test reshape(x, (42, 20, 6)) == collapseddim(x, 3, 5)
    @test reshape(x, (7, 6, 120)) == collapseddim(x, 2, 3)
    @test reshape(x, (210, 24, 1)) == collapseddim(x, 4, 8)

    @test reshape(x, (42, 20, 6)) == CollapsedDimArray(x, 3, 5)
    @test reshape(x, (7, 6, 120)) == CollapsedDimArray(x, 2, 3)
    @test reshape(x, (210, 24, 1)) == CollapsedDimArray(x, 4, 8)

    y = rand(5, 4)

    @test collapsed_size(y, 2, 3) == (5, 4, 1)
    @test collapsed_size(y, 3, 5) == (20, 1, 1)
    @test noncollapsed_size(y, 2, 3, 1) == (5,)
    @test noncollapsed_size(y, 3, 5, 1) == (5, 4)
    @test noncollapsed_size(y, 2, 3, 2) == (4,)
    @test noncollapsed_size(y, 3, 5, 2) == (1, 1)
    @test noncollapsed_size(y, 2, 3, 3) == (1,)
    @test noncollapsed_size(y, 3, 5, 3) == (1,)

    @testset "AD" begin
        x = randn(7,6,5,4,3,2)

        test_rrule(parent, CollapsedDimArray(randn(6)))
        test_rrule(parent, CollapsedDimArray(x, 3, 5))
        test_rrule(parent, CollapsedDimArray(x, 2, 3))
        test_rrule(parent, CollapsedDimArray(x, 4, 8))

        test_rrule(CollapsedDimArray, randn(6), (6, 1, 1),
                   static(2), static(3), static(true))
        test_rrule(CollapsedDimArray, x, (42, 20, 6),
                   static(3), static(5), static(false))
        test_rrule(CollapsedDimArray, x, (7, 6, 120),
                   static(2), static(3), static(false))
        test_rrule(CollapsedDimArray, x, (210, 24, 1),
                   static(4), static(8), static(true))

    end

end
