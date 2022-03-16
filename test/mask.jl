@testset "mask" begin
    using LinearAlgebra
    using NeuralAttentionlib.Masks
    using NeuralAttentionlib: getmask

    causal(x) = batched_triu!(copy(x), 0)
    trilu(x, d) = batched_tril!(batched_triu!(copy(x), -d), d)
    bandpart(x, l, u) = batched_tril!(batched_triu!(copy(x), -l), u)

    test_random(x, p) = isapprox(sum(x .* RandomMask(p)) / length(x), 1-p, atol=1e-1)

    @testset "dataless mask" begin
        a = ones(Int, 10, 10)
        b = ones(Int, 10, 10, 2)
        c = ones(Int, 10, 10, 2, 2)

        @test a .* CausalMask() == causal(a)
        @test b .* CausalMask() == causal(b)
        @test c .* CausalMask() == causal(c)

        @test a .* LocalMask(5) == trilu(a, 4)
        @test b .* LocalMask(5) == trilu(b, 4)
        @test c .* LocalMask(5) == trilu(c, 4)

        @test test_random(a, 0.1)
        @test test_random(b, 0.1)
        @test test_random(c, 0.1)
        @test test_random(a, 0.9)
        @test test_random(b, 0.9)
        @test test_random(c, 0.9)

        @test a .* BandPartMask(3, 5) == bandpart(a, 3, 5)
        @test b .* BandPartMask(3, 5) == bandpart(b, 3, 5)
        @test c .* BandPartMask(3, 5) == bandpart(c, 3, 5)

    end

    function grow_length(len1, len2, n)
        @assert size(len1) == size(len2)
        y = zeros(Int, n, n, size(len1)...)
        for i = 1:length(len1)
            idx = Tuple(CartesianIndices(len1)[i])
            l1 = len1[idx...]
            l2 = len2[idx...]
            y[1:l1, 1:l2, idx...] .= 1
        end
        return y
    end

    @testset "array mask" begin
        a = ones(Int, 10, 10)
        b = ones(Int, 10, 10, 2)
        c = ones(Int, 10, 10, 2, 2)

        gmask_a = rand(Bool, 10, 10)
        gmask_b = rand(Bool, 10, 10, 2)
        gmask_c = rand(Bool, 10, 10, 2, 2)

        @test a .* GenericMask(gmask_a) == convert(AbstractArray{Int}, gmask_a)
        @test b .* GenericMask(gmask_b) == convert(AbstractArray{Int}, gmask_b)
        @test c .* GenericMask(gmask_c) == convert(AbstractArray{Int}, gmask_c)

        smask_a = [5]
        smask_b = [3, 5]
        smask_c = rand(1:10, 2, 2)

        @test a .* SymLengthMask(smask_a) == dropdims(grow_length(smask_a, smask_a, 10), dims=3)
        @test b .* SymLengthMask(smask_b) == grow_length(smask_b, smask_b, 10)
        @test c .* SymLengthMask(smask_c) == grow_length(smask_c, smask_c, 10)

        bmaskq_a, bmaskk_a = [4], [5]
        bmaskq_b, bmaskk_b = [3, 5], [4, 6]
        bmaskq_c, bmaskk_c = rand(1:10, 2, 2), rand(1:10, 2, 2)

        @test a .* BiLengthMask(bmaskq_a, bmaskk_a) == dropdims(grow_length(bmaskk_a, bmaskq_a, 10), dims=3)
        @test b .* BiLengthMask(bmaskq_b, bmaskk_b) == grow_length(bmaskk_b, bmaskq_b, 10)
        @test c .* BiLengthMask(bmaskq_c, bmaskk_c) == grow_length(bmaskk_c, bmaskq_c, 10)

    end

    @testset "wrapper mask" begin
        a = ones(Int, 10, 10)
        b = ones(Int, 10, 10, 2)
        c = ones(Int, 10, 10, 2, 2)

        gmask_a = rand(Bool, 10, 10)
        gmask_b = rand(Bool, 10, 10, 2)
        gmask_c = rand(Bool, 10, 10, 2, 2)
        smask_a = [5]
        smask_b = [3, 5]
        smask_c = rand(1:10, 2, 2)
        bmaskq_a, bmaskk_a = [4], [5]
        bmaskq_b, bmaskk_b = [3, 5], [4, 6]
        bmaskq_c, bmaskk_c = rand(1:10, 2, 2), rand(1:10, 2, 2)

        @test a .* !CausalMask() == 1 .- causal(a)
        @test b .* !CausalMask() == 1 .- causal(b)
        @test c .* !CausalMask() == 1 .- causal(c)
        @test a .* !LocalMask(5) == 1 .- trilu(a, 4)
        @test b .* !LocalMask(5) == 1 .- trilu(b, 4)
        @test c .* !LocalMask(5) == 1 .- trilu(c, 4)
        @test a .* !BandPartMask(3, 5) == 1 .- bandpart(a, 3, 5)
        @test b .* !BandPartMask(3, 5) == 1 .- bandpart(b, 3, 5)
        @test c .* !BandPartMask(3, 5) == 1 .- bandpart(c, 3, 5)
        @test a .* !GenericMask(gmask_a) == 1 .- convert(AbstractArray{Int}, gmask_a)
        @test b .* !GenericMask(gmask_b) == 1 .- convert(AbstractArray{Int}, gmask_b)
        @test c .* !GenericMask(gmask_c) == 1 .- convert(AbstractArray{Int}, gmask_c)
        @test a .* !SymLengthMask(smask_a) == 1 .- dropdims(grow_length(smask_a, smask_a, 10), dims=3)
        @test b .* !SymLengthMask(smask_b) == 1 .- grow_length(smask_b, smask_b, 10)
        @test c .* !SymLengthMask(smask_c) == 1 .- grow_length(smask_c, smask_c, 10)
        @test a .* !BiLengthMask(bmaskq_a, bmaskk_a) == 1 .- dropdims(grow_length(bmaskk_a, bmaskq_a, 10), dims=3)
        @test b .* !BiLengthMask(bmaskq_b, bmaskk_b) == 1 .- grow_length(bmaskk_b, bmaskq_b, 10)
        @test c .* !BiLengthMask(bmaskq_c, bmaskk_c) == 1 .- grow_length(bmaskk_c, bmaskq_c, 10)

        @test a .* (!BandPartMask(4, 7) | BiLengthMask(bmaskq_a, bmaskk_a) | LocalMask(2)) == min.((1 .- bandpart(a, 4, 7) .+ dropdims(grow_length(bmaskk_a, bmaskq_a, 10), dims=3) .+ trilu(a, 1)), 1)
        @test b .* (!BandPartMask(4, 7) | BiLengthMask(bmaskq_b, bmaskk_b) | LocalMask(2)) == min.((1 .- bandpart(b, 4, 7) .+ grow_length(bmaskk_b, bmaskq_b, 10) .+ trilu(b, 1)), 1)
        @test c .* (!BandPartMask(4, 7) | BiLengthMask(bmaskq_c, bmaskk_c) | LocalMask(2)) == min.((1 .- bandpart(c, 4, 7) .+ grow_length(bmaskk_c, bmaskq_c, 10) .+ trilu(c, 1)), 1)

        @test a .* (!CausalMask() & BiLengthMask(bmaskq_a, bmaskk_a) & LocalMask(2)) == (1 .- causal(a)) .* dropdims(grow_length(bmaskk_a, bmaskq_a, 10), dims=3) .* trilu(a, 1)
        @test b .* (!CausalMask() & BiLengthMask(bmaskq_b, bmaskk_b) & LocalMask(2)) == (1 .- causal(b)) .* grow_length(bmaskk_b, bmaskq_b, 10) .* trilu(b, 1)
        @test c .* (!CausalMask() & BiLengthMask(bmaskq_c, bmaskk_c) & LocalMask(2)) == (1 .- causal(c)) .* grow_length(bmaskk_c, bmaskq_c, 10) .* trilu(c, 1)

        @test a .* (!CausalMask() | BiLengthMask(bmaskq_a, bmaskk_a) & LocalMask(2) | BandPartMask(4, 7)) == min.((1 .- causal(a)) .+ (dropdims(grow_length(bmaskk_a, bmaskq_a, 10), dims=3) .* trilu(a, 1)) .+ bandpart(a, 4, 7), 1)
        @test b .* (!CausalMask() | BiLengthMask(bmaskq_b, bmaskk_b) & LocalMask(2) | BandPartMask(4, 7)) == min.((1 .- causal(b)) .+ (grow_length(bmaskk_b, bmaskq_b, 10) .* trilu(b, 1)) .+ bandpart(a, 4, 7), 1)
        @test c .* (!CausalMask() | BiLengthMask(bmaskq_c, bmaskk_c) & LocalMask(2) | BandPartMask(4, 7)) == min.((1 .- causal(c)) .+ (grow_length(bmaskk_c, bmaskq_c, 10) .* trilu(c, 1)) .+ bandpart(a, 4, 7), 1)

        @test a .* (!(CausalMask() | BiLengthMask(bmaskq_a, bmaskk_a)) & LocalMask(2) | BandPartMask(4, 7)) ==
            min.((1 .- min.(causal(a) .+ (dropdims(grow_length(bmaskk_a, bmaskq_a, 10), dims=3)), 1)) .* trilu(a, 1) .+ bandpart(a, 4, 7), 1)
        @test b .* (!(CausalMask() | BiLengthMask(bmaskq_b, bmaskk_b)) & LocalMask(2) | BandPartMask(4, 7)) ==
            min.((1 .- min.(causal(b) .+ grow_length(bmaskk_b, bmaskq_b, 10), 1)) .* trilu(b, 1) .+ bandpart(a, 4, 7), 1)
        @test c .* (!(CausalMask() | BiLengthMask(bmaskq_c, bmaskk_c)) & LocalMask(2) | BandPartMask(4, 7)) ==
            min.((1 .- min.(causal(c) .+ grow_length(bmaskk_c, bmaskq_c, 10), 1)) .* trilu(c, 1) .+ bandpart(a, 4, 7), 1)


        @test c .* BatchedMask(BiLengthMask(bmaskq_b, bmaskk_b)) == begin
            m = reshape(grow_length(bmaskk_b, bmaskq_b, 10), 10, 10, 1, 2)
            cat(m, m; dims=3)
        end

        @test c .* BatchedMask(SymLengthMask(smask_b)) == begin
            m = reshape(grow_length(smask_b, smask_b, 10), 10, 10, 1, 2)
            cat(m, m; dims=3)
        end

        d = ones(Int, 10, 10, 2, 2, 2)

        @test d .* BatchedMask(BiLengthMask(bmaskq_c, bmaskk_c)) == begin
            m = reshape(grow_length(bmaskk_c, bmaskq_c, 10), 10, 10, 1, 2, 2)
            cat(m, m; dims=3)
        end
        @test d .* BatchedMask(BiLengthMask(bmaskq_b, bmaskk_b)) == begin
            m = reshape(grow_length(bmaskk_b, bmaskq_b, 10), 10, 10, 1, 1, 2)
            tmp = cat(m, m; dims=3)
            cat(tmp, tmp; dims=4)
        end

        e = ones(Int, 10, 10, 4)
        f = ones(Int, 10, 10, 2, 6)
        @test e .* RepeatMask(BiLengthMask(bmaskq_b, bmaskk_b), 2) == repeat(grow_length(bmaskk_b, bmaskq_b, 10), inner=(1,1,2))
        @test f .* RepeatMask(BiLengthMask(bmaskq_c, bmaskk_c), 3) == repeat(grow_length(bmaskk_c, bmaskq_c, 10), inner=(1,1,1,3))

    end

    @testset "Op" begin
        c = ones(10, 10, 2, 2)

        @test apply_mask(CausalMask(), c) == c .* CausalMask()
        @test apply_mask(NaiveAttenMaskOp(), CausalMask(), c) == c .* CausalMask()
        @test apply_mask(GenericAttenMaskOp(.*, true, 2), CausalMask(), c) == 2 .* c .* !CausalMask()
        @test apply_mask(GenericAttenMaskOp(.*, false, 2), CausalMask(), c) == 2 .* c .* CausalMask()
        @test apply_mask(GenericAttenMaskOp(.+, false, 2), CausalMask(), c) == c .+ 2 .* CausalMask()
        @test apply_mask(GenericAttenMaskOp(+, false, 2), CausalMask(), c) == c .+ 2 .* CausalMask()
        @test apply_mask(GenericAttenMaskOp(-, false, 2), CausalMask(), c) == c .- 2 .* CausalMask()
        @test apply_mask(GenericAttenMaskOp(-, true, 2), CausalMask(), c) == c .- 2 .* !CausalMask()
    end

    @testset "Broadcast" begin
        @test trues(5, 5, 1, 2, 1) .* BatchedMask(SymLengthMask([2])) .* CausalMask() ==
            trues(5, 5, 1, 2, 1) .* (BatchedMask(SymLengthMask([2])) & CausalMask())
        @test trues(5, 5, 1, 2, 1) .* BatchedMask(SymLengthMask([2])) .| CausalMask() ==
            trues(5, 5, 1, 2, 1) .* (BatchedMask(SymLengthMask([2])) | CausalMask())
        @test trues(5, 5, 1, 2, 1) .* (BatchedMask(SymLengthMask([2])) .* CausalMask()) ==
            trues(5, 5, 1, 2, 1) .* (BatchedMask(SymLengthMask([2])) & CausalMask())
        @test trues(5,5,1, 2,1) .* (BatchedMask(SymLengthMask([2])) .| CausalMask() .* LocalMask(1)) ==
            trues(5,5,1, 2,1) .* (BatchedMask(SymLengthMask([2])) | (CausalMask() & LocalMask(1)))
        @test trues(5,5,1, 2,1) .* .!(BatchedMask(SymLengthMask([2])) .| CausalMask() .* LocalMask(1)) ==
            trues(5,5,1, 2,1) .* !(BatchedMask(SymLengthMask([2])) | (CausalMask() & LocalMask(1)))

        @test_throws DimensionMismatch randn(5) .* CausalMask()
        @test_throws DimensionMismatch randn(5, 5, 2) .* SymLengthMask([2])
        @test_throws DimensionMismatch randn(5, 5, 2, 1) .* BiLengthMask([2,3], [2,2])
        @test_throws DimensionMismatch randn(5, 5, 3) .* BiLengthMask([2,3], [2,2])
        @test_throws DimensionMismatch randn(5, 5, 2) .* RepeatMask(BiLengthMask([2,3], [2,2]), 3)
        @test_throws DimensionMismatch randn(5, 5, 3) .* RepeatMask(SymLengthMask([2]), 2)
        @test_throws DimensionMismatch randn(5, 5, 2, 3) .* BatchedMask(BiLengthMask([2,3], [2,2]))
        @test_throws DimensionMismatch randn(5, 4) .* GenericMask(rand(Bool, 3, 4))
        @test_throws DimensionMismatch randn(5, 4, 2) .* (GenericMask(rand(Bool, 3, 4)) | BiLengthMask([2,3], [2,2]))
        @test_throws DimensionMismatch randn(5, 4) .* (GenericMask(rand(Bool, 3, 4)) | SymLengthMask([2]))
    end

    @testset "AD" begin
        m = (LocalMask(1) | CausalMask() & !(BandPartMask(5,5)) | BiLengthMask([2,3], [3, 7]))

        test_rrule(getindex, m ⊢ NoTangent(), 1, 4, 1)
        test_rrule(getindex, NeuralAttentionlib.GetIndexer(m) ⊢ NoTangent(), 5, 4, 2)
        test_rrule(getindex, m ⊢ NoTangent(), (1, 4, 1))
        test_rrule(getindex, NeuralAttentionlib.GetIndexer(m) ⊢ NoTangent(), (5, 4, 2))
        test_rrule(LocalMask, 2)

        test_rrule(getmask, m ⊢ NoTangent(), randn(10, 10, 2), 0.5 ⊢ NoTangent())
        test_rrule(NeuralAttentionlib.apply_mask, NaiveAttenMaskOp(), m ⊢ NoTangent(), randn(10, 10, 2))

        test_rrule(NeuralAttentionlib.apply_broadcast_mask, (*) ⊢ NoTangent(), m ⊢ NoTangent(), randn(10, 10, 2), 3 ⊢ NoTangent())
        test_rrule(NeuralAttentionlib.apply_broadcast_mask, (+) ⊢ NoTangent(), m ⊢ NoTangent(), randn(10, 10, 2), -1e9 ⊢ NoTangent(); atol=5e-2)

        y, back = Flux.pullback(ones(10, 10), RandomMask(0.5)) do x, m
            apply_mask(m, x)
        end
        @test y == back(ones(10, 10))[1]

        y, back = Flux.pullback(ones(10, 10), RandomMask(0.5)) do x, m
            apply_mask(GenericAttenMaskOp(.*, true, 2), m, x)
        end
        @test y == back(ones(10, 10))[1]

    end

end
