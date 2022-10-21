if !USE_CUDA
    @testset "CollapsedDim" begin
        using NeuralAttentionlib.Matmul
        using NeuralAttentionlib: collapseddims_fdim1, collapseddims_nonbatch, collapseddims_nonbatch_fdim1
        x = randn(7,6,5,4,3,2)

        @test collapsed_size(x, 2, 2) == (42, 20, 6)
        @test collapsed_size(x, 1, 4) == (7, 6, 120)
        @test collapsed_size(x, 3, 0) == (210, 24, 1)

        @test noncollapsed_size(x, 2, 2, 1) == (7, 6)
        @test noncollapsed_size(x, 2, 2, 2) == (5, 4)
        @test noncollapsed_size(x, 2, 2, 3) == (3, 2)
        @test noncollapsed_size(x, 1, 4, 1) == (7,)
        @test noncollapsed_size(x, 1, 4, 2) == (6,)
        @test noncollapsed_size(x, 1, 4, 3) == (5, 4, 3, 2)
        @test noncollapsed_size(x, 3, 0, 1) == (7, 6, 5)
        @test noncollapsed_size(x, 3, 0, 2) == (4, 3, 2)
        @test noncollapsed_size(x, 3, 0, 3) == (1,)

        @test reshape(x, (42, 20, 6)) == collapseddims(x, 2, 2)
        @test reshape(x, (7, 6, 120)) == collapseddims(x, 1, 4)
        @test reshape(x, (210, 24, 1)) == collapseddims(x, 3, 0)

        @test reshape(x, (42, 20, 6)) == CollapsedDimsArray(x, 2, 2)
        @test reshape(x, (7, 6, 120)) == CollapsedDimsArray(x, 1, 4)
        @test reshape(x, (210, 24, 1)) == CollapsedDimsArray(x, 3, 0)

        csize(x) = size(collapseddims(identity, x))
        csize1(x) = size(collapseddims_fdim1(a->matmul(randn(size(a, 1)+1, size(a, 1)) , a), x))
        @test csize(CollapsedDimsArray(x, 2, 2)) == collapsed_size(x, 2, 2)
        @test csize(CollapsedDimsArray(x, 1, 4)) == collapsed_size(x, 1, 4)
        @test csize(CollapsedDimsArray(x, 3, 0)) == collapsed_size(x, 3, 0)
        @test csize1(CollapsedDimsArray(x, 2, 2)) ==
            (collapsed_size(x, 2, 2, 1) + 1, collapsed_size(x, 2, 2, 2), collapsed_size(x, 2, 2, 3))
        @test csize1(CollapsedDimsArray(x, 1, 4)) ==
            (collapsed_size(x, 1, 4, 1) + 1, collapsed_size(x, 1, 4, 2), collapsed_size(x, 1, 4, 3))
        @test csize1(CollapsedDimsArray(x, 3, 0)) ==
            (collapsed_size(x, 3, 0, 1) + 1, collapsed_size(x, 3, 0, 2), collapsed_size(x, 3, 0, 3))

        fsize(x) = size(unwrap_collapse(collapseddims(identity, x)))
        fsize1(x) = size(unwrap_collapse(collapseddims_fdim1(a->matmul(randn(size(a, 1)+1, size(a, 1)) , a), x)))
        @test fsize(CollapsedDimsArray(x, 2, 2)) == size(x)
        @test fsize(CollapsedDimsArray(x, 1, 4)) == size(x)
        @test fsize(CollapsedDimsArray(x, 3, 0)) == size(x)
        @test fsize1(CollapsedDimsArray(x, 2, 2)) ==
            (collapsed_size(x, 2, 2, 1) + 1, noncollapsed_size(x, 2, 2, 2)..., noncollapsed_size(x, 2, 2, 3)...)
        @test fsize1(CollapsedDimsArray(x, 1, 4)) ==
            (collapsed_size(x, 1, 4, 1) + 1, noncollapsed_size(x, 1, 4, 2)..., noncollapsed_size(x, 1, 4, 3)...)
        @test fsize1(CollapsedDimsArray(x, 3, 0)) ==
            (collapsed_size(x, 3, 0, 1) + 1, noncollapsed_size(x, 3, 0, 2)...)

        csize_nonbatch(x) = size(collapseddims_nonbatch(identity, x))
        csize1_nonbatch(x) = size(collapseddims_nonbatch_fdim1(a->matmul(randn(size(a, 1)+1, size(a, 1)) , a), x))
        @test csize_nonbatch(CollapsedDimsArray(x, 2, 2)) == collapsed_size(x, 2, 2)
        @test csize_nonbatch(CollapsedDimsArray(x, 1, 4)) == collapsed_size(x, 1, 4)
        @test csize_nonbatch(CollapsedDimsArray(x, 3, 0)) == collapsed_size(x, 3, 0)
        @test csize1_nonbatch(CollapsedDimsArray(x, 2, 2)) ==
            (collapsed_size(x, 2, 2, 1) + 1, collapsed_size(x, 2, 2, 2), collapsed_size(x, 2, 2, 3))
        @test csize1_nonbatch(CollapsedDimsArray(x, 1, 4)) ==
            (collapsed_size(x, 1, 4, 1) + 1, collapsed_size(x, 1, 4, 2), collapsed_size(x, 1, 4, 3))
        @test csize1_nonbatch(CollapsedDimsArray(x, 3, 0)) ==
            (collapsed_size(x, 3, 0, 1) + 1, collapsed_size(x, 3, 0, 2), collapsed_size(x, 3, 0, 3))

        fsize_nonbatch(x) = size(unwrap_collapse(collapseddims_nonbatch(identity, x)))
        fsize1_nonbatch(x) = size(unwrap_collapse(collapseddims_nonbatch_fdim1(a->matmul(randn(size(a, 1)+1, size(a, 1)) , a), x)))
        @test fsize_nonbatch(CollapsedDimsArray(x, 2, 2)) == size(x)
        @test fsize_nonbatch(CollapsedDimsArray(x, 1, 4)) == size(x)
        @test fsize_nonbatch(CollapsedDimsArray(x, 3, 0)) == size(x)
        @test fsize1_nonbatch(CollapsedDimsArray(x, 2, 2)) ==
            (collapsed_size(x, 2, 2, 1) + 1, noncollapsed_size(x, 2, 2, 2)..., noncollapsed_size(x, 2, 2, 3)...)
        @test fsize1_nonbatch(CollapsedDimsArray(x, 1, 4)) ==
            (collapsed_size(x, 1, 4, 1) + 1, noncollapsed_size(x, 1, 4, 2)..., noncollapsed_size(x, 1, 4, 3)...)
        @test fsize1_nonbatch(CollapsedDimsArray(x, 3, 0)) ==
            (collapsed_size(x, 3, 0, 1) + 1, noncollapsed_size(x, 3, 0, 2)...)

        y = rand(5, 4)

        @test collapsed_size(y, 1, 0) == (5, 4, 1)
        @test collapsed_size(y, 0, 0) == (20, 1, 1)
        @test noncollapsed_size(y, 1, 0, 1) == (5,)
        @test noncollapsed_size(y, 0, 0, 1) == (5, 4)
        @test noncollapsed_size(y, 1, 0, 2) == (4,)
        @test noncollapsed_size(y, 0, 0, 2) == (1,)
        @test noncollapsed_size(y, 1, 0, 3) == (1,)
        @test noncollapsed_size(y, 0, 0, 3) == (1,)

        @testset "AD" begin
            x = randn(7,6,5,4,3,2)

            test_rrule(parent, CollapsedDimsArray(randn(6)))
            test_rrule(parent, CollapsedDimsArray(x, 2, 2))
            test_rrule(parent, CollapsedDimsArray(x, 1, 4))
            test_rrule(parent, CollapsedDimsArray(x, 3, 0))

            test_rrule(CollapsedDimsArray, randn(6), (6, 1, 1),
                       static(0) ⊢ NoTangent(), static(0) ⊢ NoTangent())
            test_rrule(CollapsedDimsArray, x, (42, 20, 6),
                       static(2) ⊢ NoTangent(), static(2) ⊢ NoTangent())
            test_rrule(CollapsedDimsArray, x, (7, 6, 120),
                       static(1) ⊢ NoTangent(), static(4) ⊢ NoTangent())
            test_rrule(CollapsedDimsArray, x, (210, 24, 1),
                       static(3) ⊢ NoTangent(), static(0) ⊢ NoTangent())

        end

    end
end
