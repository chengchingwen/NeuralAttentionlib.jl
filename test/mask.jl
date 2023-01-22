@testset "mask" begin
    using LinearAlgebra
    using NeuralAttentionlib.Masks
    using NeuralAttentionlib: getmask, lengths

    causal(x) = batched_triu!(copy(x), 0) |> device
    trilu(x, d) = batched_tril!(batched_triu!(copy(x), -d), d) |> device
    bandpart(x, l, u) = batched_tril!(batched_triu!(copy(x), -l), u) |> device

    test_random(x, p) = isapprox(sum(x .* RandomMask(p)) / length(x), 1-p, atol=1e-1)

    @testset "dataless mask" begin
        a = dones(Int, 10, 10)
        b = dones(Int, 10, 10, 2)
        c = dones(Int, 10, 10, 2, 2)

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

    function grow_length(_len1, _len2, n)
        len1 = collect(_len1)
        len2 = collect(_len2)
        @assert size(len1) == size(len2)
        y = zeros(Int, n, n, size(len1)...)
        for i = 1:length(len1)
            idx = Tuple(CartesianIndices(len1)[i])
            l1 = len1[idx...]
            l2 = len2[idx...]
            y[1:l1, 1:l2, idx...] .= 1
        end
        return y |> device
    end
    grow_rlength(len1, len2, n) = reverse!(reverse!(grow_length(len1, len2, n); dims=1); dims=2)

    @testset "array mask" begin
        a = dones(Int, 10, 10)
        b = dones(Int, 10, 10, 2)
        c = dones(Int, 10, 10, 2, 2)

        gmask_a = drand(Bool, 10, 10)
        gmask_b = drand(Bool, 10, 10, 2)
        gmask_c = drand(Bool, 10, 10, 2, 2)

        @test a .* GenericAttenMask(gmask_a) == convert(AbstractArray{Int}, gmask_a)
        @test b .* GenericAttenMask(gmask_b) == convert(AbstractArray{Int}, gmask_b)
        @test c .* GenericAttenMask(gmask_c) == convert(AbstractArray{Int}, gmask_c)

        smask_a = [5] |> device
        smask_b = [3, 5] |> device
        smask_c = drand(1:10, 2, 2)

        @test a .* SymLengthMask(smask_a) == dropdims(grow_length(smask_a, smask_a, 10), dims=3)
        @test b .* SymLengthMask(smask_b) == grow_length(smask_b, smask_b, 10)
        @test c .* SymLengthMask(smask_c) == grow_length(smask_c, smask_c, 10)

        bmaskq_a, bmaskk_a = device([4]), device([5])
        bmaskq_b, bmaskk_b = device([3, 5]), device([4, 6])
        bmaskq_c, bmaskk_c = drand(1:10, 2, 2), drand(1:10, 2, 2)

        @test a .* BiLengthMask(bmaskq_a, bmaskk_a) == dropdims(grow_length(bmaskk_a, bmaskq_a, 10), dims=3)
        @test b .* BiLengthMask(bmaskq_b, bmaskk_b) == grow_length(bmaskk_b, bmaskq_b, 10)
        @test c .* BiLengthMask(bmaskq_c, bmaskk_c) == grow_length(bmaskk_c, bmaskq_c, 10)

        rev!(x) = reverse!(reverse!(x; dims=1); dims=2)
        @test a .* RevSymLengthMask(smask_a) == rev!(a .* SymLengthMask(smask_a))
        @test b .* RevSymLengthMask(smask_b) == rev!(b .* SymLengthMask(smask_b))
        @test c .* RevSymLengthMask(smask_c) == rev!(c .* SymLengthMask(smask_c))
        @test a .* RevBiLengthMask(bmaskq_a, bmaskk_a) == rev!(a .* BiLengthMask(bmaskq_a, bmaskk_a))
        @test b .* RevBiLengthMask(bmaskq_b, bmaskk_b) == rev!(b .* BiLengthMask(bmaskq_b, bmaskk_b))
        @test c .* RevBiLengthMask(bmaskq_c, bmaskk_c) == rev!(c .* BiLengthMask(bmaskq_c, bmaskk_c))
    end

    @testset "wrapper mask" begin
        a = dones(Int, 10, 10)
        b = dones(Int, 10, 10, 2)
        c = dones(Int, 10, 10, 2, 2)

        gmask_a = drand(Bool, 10, 10)
        gmask_b = drand(Bool, 10, 10, 2)
        gmask_c = drand(Bool, 10, 10, 2, 2)
        smask_a = [5] |> device
        smask_b = [3, 5] |> device
        smask_c = drand(1:10, 2, 2)
        bmaskq_a, bmaskk_a = device([4]), device([5])
        bmaskq_b, bmaskk_b = device([3, 5]), device([4, 6])
        bmaskq_c, bmaskk_c = drand(1:10, 2, 2), drand(1:10, 2, 2)

        @test a .* !CausalMask() == 1 .- causal(a)
        @test b .* !CausalMask() == 1 .- causal(b)
        @test c .* !CausalMask() == 1 .- causal(c)
        @test a .* !LocalMask(5) == 1 .- trilu(a, 4)
        @test b .* !LocalMask(5) == 1 .- trilu(b, 4)
        @test c .* !LocalMask(5) == 1 .- trilu(c, 4)
        @test a .* !BandPartMask(3, 5) == 1 .- bandpart(a, 3, 5)
        @test b .* !BandPartMask(3, 5) == 1 .- bandpart(b, 3, 5)
        @test c .* !BandPartMask(3, 5) == 1 .- bandpart(c, 3, 5)
        @test a .* (!BiLengthMask(bmaskq_a, bmaskk_a) & BiLengthMask(bmaskq_a, bmaskk_a)) |> iszero
        @test b .* (!BiLengthMask(bmaskq_b, bmaskk_b) & BiLengthMask(bmaskq_b, bmaskk_b)) |> iszero
        @test c .* (!BiLengthMask(bmaskq_c, bmaskk_c) & BiLengthMask(bmaskq_c, bmaskk_c)) |> iszero
        @test a .* !GenericAttenMask(gmask_a) == 1 .- convert(AbstractArray{Int}, gmask_a)
        @test b .* !GenericAttenMask(gmask_b) == 1 .- convert(AbstractArray{Int}, gmask_b)
        @test c .* !GenericAttenMask(gmask_c) == 1 .- convert(AbstractArray{Int}, gmask_c)
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

        @test a .* (!CausalMask() | RevBiLengthMask(bmaskq_a, bmaskk_a) & LocalMask(2) | BandPartMask(4, 7)) == min.((1 .- causal(a)) .+ (dropdims(grow_rlength(bmaskk_a, bmaskq_a, 10), dims=3) .* trilu(a, 1)) .+ bandpart(a, 4, 7), 1)
        @test b .* (!CausalMask() | RevBiLengthMask(bmaskq_b, bmaskk_b) & LocalMask(2) | BandPartMask(4, 7)) == min.((1 .- causal(b)) .+ (grow_rlength(bmaskk_b, bmaskq_b, 10) .* trilu(b, 1)) .+ bandpart(a, 4, 7), 1)
        @test c .* (!CausalMask() | RevBiLengthMask(bmaskq_c, bmaskk_c) & LocalMask(2) | BandPartMask(4, 7)) == min.((1 .- causal(c)) .+ (grow_rlength(bmaskk_c, bmaskq_c, 10) .* trilu(c, 1)) .+ bandpart(a, 4, 7), 1)

        @test a .* (!(CausalMask() | RevBiLengthMask(bmaskq_a, bmaskk_a)) & LocalMask(2) | BandPartMask(4, 7)) ==
            min.((1 .- min.(causal(a) .+ (dropdims(grow_rlength(bmaskk_a, bmaskq_a, 10), dims=3)), 1)) .* trilu(a, 1) .+ bandpart(a, 4, 7), 1)
        @test b .* (!(CausalMask() | RevBiLengthMask(bmaskq_b, bmaskk_b)) & LocalMask(2) | BandPartMask(4, 7)) ==
            min.((1 .- min.(causal(b) .+ grow_rlength(bmaskk_b, bmaskq_b, 10), 1)) .* trilu(b, 1) .+ bandpart(a, 4, 7), 1)
        @test c .* (!(CausalMask() | RevBiLengthMask(bmaskq_c, bmaskk_c)) & LocalMask(2) | BandPartMask(4, 7)) ==
            min.((1 .- min.(causal(c) .+ grow_rlength(bmaskk_c, bmaskq_c, 10), 1)) .* trilu(c, 1) .+ bandpart(a, 4, 7), 1)

        @test c .* BatchedMask(BiLengthMask(bmaskq_b, bmaskk_b)) == begin
            m = reshape(grow_length(bmaskk_b, bmaskq_b, 10), 10, 10, 1, 2)
            cat(m, m; dims=3)
        end

        @test c .* BatchedMask(SymLengthMask(smask_b)) == begin
            m = reshape(grow_length(smask_b, smask_b, 10), 10, 10, 1, 2)
            cat(m, m; dims=3)
        end

        d = dones(Int, 10, 10, 2, 2, 2)

        @test d .* BatchedMask(BiLengthMask(bmaskq_c, bmaskk_c)) == begin
            m = reshape(grow_length(bmaskk_c, bmaskq_c, 10), 10, 10, 1, 2, 2)
            cat(m, m; dims=3)
        end
        @test d .* BatchedMask(BiLengthMask(bmaskq_b, bmaskk_b)) == begin
            m = reshape(grow_length(bmaskk_b, bmaskq_b, 10), 10, 10, 1, 1, 2)
            tmp = cat(m, m; dims=3)
            cat(tmp, tmp; dims=4)
        end

        e = dones(Int, 10, 10, 4)
        f = dones(Int, 10, 10, 2, 6)
        @test e .* RepeatMask(BiLengthMask(bmaskq_b, bmaskk_b), 2) == device(repeat(collect(grow_length(bmaskk_b, bmaskq_b, 10)), inner=(1,1,2)))
        @test f .* RepeatMask(BiLengthMask(bmaskq_c, bmaskk_c), 3) == device(repeat(collect(grow_length(bmaskk_c, bmaskq_c, 10)), inner=(1,1,1,3)))

        nested_mask = (BatchedMask(BatchedMask((LocalMask(1) | CausalMask() & !(BandPartMask(5,5)) & BatchedMask(RevBiLengthMask([2,3], [3, 5])) | RepeatMask(GenericAttenMask(rand(Bool, 10, 10, 1)), 2)))) | CausalMask() & LocalMask(3))
        @test collect(b .* device(nested_mask)) == collect(b) .* nested_mask
        @test collect(c .* device(nested_mask)) == collect(c) .* nested_mask

        nested_mask2 = BatchedMask(!BatchedMask(!BatchedMask((RepeatMask((LocalMask(1) | CausalMask() & !(BandPartMask(5,5)) & BatchedMask(RevBiLengthMask([2,3], [3, 5])) | RepeatMask(GenericAttenMask(rand(Bool, 10, 10, 1)), 2)), 1))))) | RepeatMask(LocalMask(5),2) & CausalMask()
        @test collect(b .* device(nested_mask2)) == collect(b) .* nested_mask2
        @test collect(c .* device(nested_mask2)) == collect(c) .* nested_mask2
    end

    @testset "Sequence" begin
        a0 = hcat([1, 1, 1, 1, 0], [1,1,1,0,0])
        ra0 = hcat([0, 1, 1, 1, 1], [0,0,1,1,1])
        a = device(reshape(a0, (1, 5, 2)))
        ra = device(reshape(ra0, (1, 5, 2)))
        b = device([4,3])
        x = drandn(10, 5, 2)
        a2 = device(repeat(reshape(a0, (1, 5, 1, 2)); inner=(1,1,2,1)))
        ra2 = device(repeat(reshape(ra0, (1, 5, 1, 2)); inner=(1,1,2,1)))
        x2 = drandn(10, 5, 2,2)

        @test lengths(LengthMask(b)) == b
        @test lengths(RevLengthMask(b)) == b
        @test lengths(GenericSequenceMask(a)) == b

        @test x .* a == x .* LengthMask(b)
        @test x .* ra == x .* RevLengthMask(b)
        @test x .* a == x .* GenericSequenceMask(a)
        @test x2 .* a2 == x2 .* BatchedMask(LengthMask(b))
        @test x2 .* ra2 == x2 .* BatchedMask(RevLengthMask(b))
        @test x2 .* a2 == x2 .* BatchedMask(GenericSequenceMask(a))

        len = [5]
        l = 2
        for f in (+, *)
            @test f(l, LengthMask(len)).len[] == f(l, len[])
            @test f(l, RevLengthMask(len)).len[] == f(l, len[])
            @test f(l, SymLengthMask(len)).len[] == f(l, len[])
            @test f(l, BiLengthMask(len, len)).q_len[] == f(l, len[])
            @test f(l, BiLengthMask(len, len)).k_len[] == f(l, len[])
            @test f(l, RevSymLengthMask(len)).len[] == f(l, len[])
            @test f(l, RevBiLengthMask(len, len)).q_len[] == f(l, len[])
            @test f(l, RevBiLengthMask(len, len)).k_len[] == f(l, len[])
        end

        @test (LengthMask(len) - l).len[] == len[] - l
        @test (SymLengthMask(len) - l).len[] == len[] - l
        @test (BiLengthMask(len, len) - l).q_len[] == len[] - l
        @test (BiLengthMask(len, len) - l).k_len[] == len[] - l
        @test (RevLengthMask(len) - l).len[] == len[] - l
        @test (RevSymLengthMask(len) - l).len[] == len[] - l
        @test (RevBiLengthMask(len, len) - l).q_len[] == len[] - l
        @test (RevBiLengthMask(len, len) - l).k_len[] == len[] - l
    end

    @testset "Op" begin
        c = dones(10, 10, 2, 2)

        @test apply_mask(CausalMask(), c) == c .* CausalMask()
        @test apply_mask(NaiveMaskOp(), CausalMask(), c) == c .* CausalMask()
        @test apply_mask(GenericMaskOp(.*, true, 2), CausalMask(), c) == 2 .* c .* !CausalMask()
        @test apply_mask(GenericMaskOp(.*, false, 2), CausalMask(), c) == 2 .* c .* CausalMask()
        @test apply_mask(GenericMaskOp(.+, false, 2), CausalMask(), c) == c .+ 2 .* CausalMask()
        @test apply_mask(GenericMaskOp(+, false, 2), CausalMask(), c) == c .+ 2 .* CausalMask()
        @test apply_mask(GenericMaskOp(-, false, 2), CausalMask(), c) == c .- 2 .* CausalMask()
        @test apply_mask(GenericMaskOp(-, true, 2), CausalMask(), c) == c .- 2 .* !CausalMask()
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

        @test_throws DimensionMismatch drandn(5) .* CausalMask()
        @test_throws DimensionMismatch drandn(5, 5, 2) .* SymLengthMask([2])
        @test_throws DimensionMismatch drandn(5, 5, 2, 1) .* BiLengthMask([2,3], [2,2])
        @test_throws DimensionMismatch drandn(5, 5, 3) .* BiLengthMask([2,3], [2,2])
        @test_throws DimensionMismatch drandn(5, 5, 2) .* RepeatMask(BiLengthMask([2,3], [2,2]), 3)
        @test_throws DimensionMismatch drandn(5, 5, 3) .* RepeatMask(SymLengthMask([2]), 2)
        @test_throws DimensionMismatch drandn(5, 5, 2, 3) .* BatchedMask(BiLengthMask([2,3], [2,2]))
        @test_throws DimensionMismatch drandn(5, 4) .* GenericAttenMask(drand(Bool, 3, 4))
        @test_throws DimensionMismatch drandn(5, 4, 2) .* (GenericAttenMask(drand(Bool, 3, 4)) | BiLengthMask([2,3], [2,2]))
        @test_throws DimensionMismatch drandn(5, 4) .* (GenericAttenMask(drand(Bool, 3, 4)) | SymLengthMask([2]))
    end

    if !USE_CUDA
        @testset "AD" begin
            m = (LocalMask(1) | CausalMask() & !(BandPartMask(5,5)) | BiLengthMask([2,3], [3, 7]))

            test_rrule(getindex, m ⊢ NoTangent(), 1, 4, 1)
            test_rrule(getindex, NeuralAttentionlib.GetIndexer(m) ⊢ NoTangent(), 5, 4, 2)
            test_rrule(getindex, m ⊢ NoTangent(), (1, 4, 1))
            test_rrule(getindex, NeuralAttentionlib.GetIndexer(m) ⊢ NoTangent(), (5, 4, 2))
            test_rrule(LocalMask, 2)

            test_rrule(getmask, m ⊢ NoTangent(), drandn(10, 10, 2), 0.5 ⊢ NoTangent())
            test_rrule(NeuralAttentionlib.apply_mask, NaiveMaskOp(), m ⊢ NoTangent(), drandn(10, 10, 2))

            test_rrule(NeuralAttentionlib.apply_broadcast_mask, (*) ⊢ NoTangent(), m ⊢ NoTangent(), drandn(10, 10, 2), 3 ⊢ NoTangent())
            test_rrule(NeuralAttentionlib.apply_broadcast_mask, (+) ⊢ NoTangent(), m ⊢ NoTangent(), drandn(10, 10, 2), -1e9 ⊢ NoTangent(); atol=5e-2)

            y, back = Flux.pullback(dones(10, 10), RandomMask(0.5)) do x, m
                apply_mask(m, x)
            end
            @test y == back(dones(10, 10))[1]

            y, back = Flux.pullback(dones(10, 10), RandomMask(0.5)) do x, m
                apply_mask(GenericMaskOp(.*, true, 2), m, x)
            end
            @test y == back(dones(10, 10))[1]

        end
    end
end
