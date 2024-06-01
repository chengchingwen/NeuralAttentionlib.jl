@testset "MHA" begin
    using Random
    using NeuralAttentionlib.Masks
    using NeuralAttentionlib: score_returning, dropoutF
    head = 4
    input_dims = 7
    head_dims = 5
    output_dims = 8

    function atten(mha, x, mask=nothing, return_score = false; p = nothing)
        q = mha.iqproj(x)
        k = mha.ikproj(x)
        v = mha.ivproj(x)
        if return_score
            a, score = NeuralAttentionlib.multihead_qkv_attention(score_returning, 4, q, k, v, mask, p)
            return mha.oproj(a), score
        else
            a = NeuralAttentionlib.multihead_qkv_attention(4, q, k, v, mask, p)
            return mha.oproj(a)
        end
    end

    @testset "w/o mask" begin
        x = randn(Float32, input_dims, 3, 2)
        mha = MultiheadAttention(head, input_dims, head_dims, output_dims;
                                 future = true, drop = nothing) # disable dropout

        mha = device(mha)
        x = device(x)

        @test mha(x,x,x) ≈ atten(mha, x)

        gradN = Flux.gradient(mha, x) do mha, x
            sum(sin.(atten(mha, x)))
        end
        gradT = Flux.gradient(mha, x) do mha, x
            sum(sin.(mha(x,x,x)))
        end

        @test isapprox(gradN[1].iqproj.weight, gradT[1].iqproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.weight, gradT[1].ikproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].ivproj.weight, gradT[1].ivproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].oproj.weight, gradT[1].oproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].iqproj.bias, gradT[1].iqproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.bias, gradT[1].ikproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].ivproj.bias, gradT[1].ivproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].oproj.bias, gradT[1].oproj.bias; atol = 1e-4)
        @test gradN[2] ≈ gradT[2]
    end

    @testset "w/ future mask" begin
        x = randn(Float32, input_dims, 3, 2)
        mha = MultiheadAttention(head, input_dims, head_dims, output_dims;
                                 future = false, drop = nothing) # disable dropout

        mha = device(mha)
        x = device(x)

        @test mha(x,x,x) ≈ atten(mha, x, CausalMask())

        gradN = Flux.gradient(mha, x) do mha, x
            sum(sin.(atten(mha, x, CausalMask())))
        end
        gradT = Flux.gradient(mha, x) do mha, x
            sum(sin.(mha(x,x,x)))
        end

        @test isapprox(gradN[1].iqproj.weight, gradT[1].iqproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.weight, gradT[1].ikproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].ivproj.weight, gradT[1].ivproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].oproj.weight, gradT[1].oproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].iqproj.bias, gradT[1].iqproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.bias, gradT[1].ikproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].ivproj.bias, gradT[1].ivproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].oproj.bias, gradT[1].oproj.bias; atol = 1e-4)
        @test gradN[2] ≈ gradT[2]
    end

    @testset "w/ future + length mask" begin
        x = randn(Float32, input_dims, 6, 2)
        mha = MultiheadAttention(head, input_dims, head_dims, output_dims;
                                 future = false, drop = nothing) # disable dropout

        q_len = [3, 5]
        k_len = [4, 4]
        lmask = BatchedMask(BiLengthMask(q_len, k_len))
        mask = CausalMask() & lmask

        mask2 = device(CausalMask() & BatchedMask(LengthMask(q_len)))
        mask3 = device(CausalMask() & BatchedMask(SymLengthMask(q_len)))

        old_q_len = Old_Impl.getmask(map(l->ones(l), q_len))
        old_k_len = Old_Impl.getmask(map(l->ones(l), k_len))
        old_mask = zeros(Float32, 6, 6, 2)
        old_mask[1:4, 1:5, :] .= Old_Impl.getmask(old_k_len, old_q_len)

        mha = device(mha)
        x = device(x)
        mask = device(mask)
        old_mask = device(old_mask)

        @test mha(x,x,x, mask=old_mask) ≈ atten(mha, x, mask)
        @test atten(mha, x, mask2) ≈ atten(mha, x, mask3)

        gradN = Flux.gradient(mha, x, mask) do mha, x, m
            sum(sin.(atten(mha, x, m)))
        end
        gradT = Flux.gradient(mha, x, old_mask) do mha, x, m
            sum(sin.(mha(x,x,x, mask=m)))
        end

        @test isapprox(gradN[1].iqproj.weight, gradT[1].iqproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.weight, gradT[1].ikproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].ivproj.weight, gradT[1].ivproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].oproj.weight, gradT[1].oproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].iqproj.bias, gradT[1].iqproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.bias, gradT[1].ikproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].ivproj.bias, gradT[1].ivproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].oproj.bias, gradT[1].oproj.bias; atol = 1e-4)
        @test gradN[2] ≈ gradT[2]
        @test gradN[3] == gradT[3]
    end

    @testset "w/ future + reverse length mask" begin
        x = randn(Float32, input_dims, 6, 2)
        mha = MultiheadAttention(head, input_dims, head_dims, output_dims;
                                 future = false, drop = nothing) # disable dropout

        q_len = [3, 5]
        k_len = [4, 4]
        lmask = BatchedMask(RevBiLengthMask(q_len, k_len))
        mask = CausalMask() & lmask

        mask2 = device(CausalMask() & BatchedMask(RevLengthMask(q_len)))
        mask3 = device(CausalMask() & BatchedMask(RevSymLengthMask(q_len)))

        old_q_len = reverse!(reverse!(Old_Impl.getmask(map(l->ones(l), q_len)); dims=1); dims=2)
        old_k_len = reverse!(reverse!(Old_Impl.getmask(map(l->ones(l), k_len)); dims=1); dims=2)
        old_mask = zeros(Float32, 6, 6, 2)
        old_mask[3:6, 2:6, :] .= Old_Impl.getmask(old_k_len, old_q_len)

        mha = device(mha)
        x = device(x)
        mask = device(mask)
        old_mask = device(old_mask)

        @test mha(x,x,x, mask=old_mask) ≈ atten(mha, x, mask)
        @test atten(mha, x, mask2) ≈ atten(mha, x, mask3)

        gradN = Flux.gradient(mha, x, mask) do mha, x, m
            sum(sin.(atten(mha, x, m)))
        end
        gradT = Flux.gradient(mha, x, old_mask) do mha, x, m
            sum(sin.(mha(x,x,x, mask=m)))
        end

        @test isapprox(gradN[1].iqproj.weight, gradT[1].iqproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.weight, gradT[1].ikproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].ivproj.weight, gradT[1].ivproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].oproj.weight, gradT[1].oproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].iqproj.bias, gradT[1].iqproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.bias, gradT[1].ikproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].ivproj.bias, gradT[1].ivproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].oproj.bias, gradT[1].oproj.bias; atol = 1e-4)
        @test gradN[2] ≈ gradT[2]
        @test gradN[3] == gradT[3]
    end

    @testset "w/ batched mask" begin
        x = randn(Float32, input_dims, 6, 2)
        mha = MultiheadAttention(head, input_dims, head_dims, output_dims;
                                 future = false, drop = nothing) # disable dropout

        q_len = [3, 5]
        k_len = [4, 4]
        lmask = RevBiLengthMask(q_len, k_len)
        mask = CausalMask() & lmask

        mask2 = device(CausalMask() & RevLengthMask(q_len))
        mask3 = device(CausalMask() & RevSymLengthMask(q_len))

        old_q_len = reverse!(reverse!(Old_Impl.getmask(map(l->ones(l), q_len)); dims=1); dims=2)
        old_k_len = reverse!(reverse!(Old_Impl.getmask(map(l->ones(l), k_len)); dims=1); dims=2)
        old_mask = zeros(Float32, 6, 6, 2)
        old_mask[3:6, 2:6, :] .= Old_Impl.getmask(old_k_len, old_q_len)

        mha = device(mha)
        x = device(x)
        mask = device(mask)
        old_mask = device(old_mask)

        @test mha(x,x,x, mask=old_mask) ≈ atten(mha, x, BatchedMask(mask))
        @test atten(mha, x, BatchedMask(mask2)) ≈ atten(mha, x, BatchedMask(mask3))

        gradN = Flux.gradient(mha, x, mask) do mha, x, m
            sum(sin.(atten(mha, x, BatchedMask(m))))
        end
        gradT = Flux.gradient(mha, x, old_mask) do mha, x, m
            sum(sin.(mha(x,x,x, mask=m)))
        end

        @test isapprox(gradN[1].iqproj.weight, gradT[1].iqproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.weight, gradT[1].ikproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].ivproj.weight, gradT[1].ivproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].oproj.weight, gradT[1].oproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].iqproj.bias, gradT[1].iqproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.bias, gradT[1].ikproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].ivproj.bias, gradT[1].ivproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].oproj.bias, gradT[1].oproj.bias; atol = 1e-4)
        @test gradN[2] ≈ gradT[2]
        @test gradN[3] == gradT[3]
    end

    @testset "w/ score" begin
        x = randn(Float32, input_dims, 6, 2)
        mha = MultiheadAttention(head, input_dims, head_dims, output_dims;
                                 future = false, drop = nothing) # disable dropout

        q_len = [3, 5]
        k_len = [4, 4]
        lmask = RevBiLengthMask(q_len, k_len)
        mask = CausalMask() & lmask

        mask2 = device(CausalMask() & RevLengthMask(q_len))
        mask3 = device(CausalMask() & RevSymLengthMask(q_len))

        old_q_len = reverse!(reverse!(Old_Impl.getmask(map(l->ones(l), q_len)); dims=1); dims=2)
        old_k_len = reverse!(reverse!(Old_Impl.getmask(map(l->ones(l), k_len)); dims=1); dims=2)
        old_mask = zeros(Float32, 6, 6, 2)
        old_mask[3:6, 2:6, :] .= Old_Impl.getmask(old_k_len, old_q_len)

        mha = device(mha)
        x = device(x)
        mask = device(mask)
        old_mask = device(old_mask)

        @test mha(x,x,x, mask=old_mask, return_score=true)[1] ≈ atten(mha, x, BatchedMask(mask), true)[1]
        @test reshape(mha(x,x,x, mask=old_mask, return_score=true)[2],6,6,head,2) ≈ atten(mha, x, BatchedMask(mask), true)[2]
        @test atten(mha, x, BatchedMask(mask2), true)[1] ≈ atten(mha, x, BatchedMask(mask3), true)[1]
        @test atten(mha, x, BatchedMask(mask2), true)[2] ≈ atten(mha, x, BatchedMask(mask3), true)[2]

        gradN = Flux.gradient(mha, x, mask) do mha, x, m
            a, s = atten(mha, x, BatchedMask(m), true)
            sum(sin.(a)) + sum(cos.(s))
        end
        gradT = Flux.gradient(mha, x, old_mask) do mha, x, m
            a, s = mha(x,x,x, mask=m, return_score=true)
            sum(sin.(a)) + sum(cos.(s))
        end

        @test isapprox(gradN[1].iqproj.weight, gradT[1].iqproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.weight, gradT[1].ikproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].ivproj.weight, gradT[1].ivproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].oproj.weight, gradT[1].oproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].iqproj.bias, gradT[1].iqproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.bias, gradT[1].ikproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].ivproj.bias, gradT[1].ivproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].oproj.bias, gradT[1].oproj.bias; atol = 1e-4)
        @test gradN[2] ≈ gradT[2]
        @test gradN[3] == gradT[3]

        gradN = Flux.gradient(mha, x, mask) do mha, x, m
            a, s = atten(mha, x, BatchedMask(m), true)
            sum(cos.(s))
        end
        gradT = Flux.gradient(mha, x, old_mask) do mha, x, m
            a, s = mha(x,x,x, mask=m, return_score=true)
            sum(cos.(s))
        end

        @test isapprox(gradN[1].iqproj.weight, gradT[1].iqproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.weight, gradT[1].ikproj.weight; atol = 1e-4)
        @test gradN[1].ivproj == gradT[1].ivproj
        @test gradN[1].oproj == gradT[1].oproj
        @test isapprox(gradN[1].iqproj.bias, gradT[1].iqproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.bias, gradT[1].ikproj.bias; atol = 1e-4)
        @test gradN[2] ≈ gradT[2]
        @test gradN[3] == gradT[3]

        x0 = device(randn(Float32, input_dims, 6))
        @test mha(x0,x0,x0; return_score=true)[1] ≈ atten(mha, x0, CausalMask(), true)[1]
        @test reshape(mha(x0,x0,x0; return_score=true)[2],6,6,head,1) ≈ atten(mha, x0, CausalMask(), true)[2]

        gradN = Flux.gradient(mha, x0, CausalMask()) do mha, x, m
            a, s = atten(mha, x, BatchedMask(m), true)
            sum(sin.(a)) + sum(cos.(s))
        end
        gradT = Flux.gradient(mha, x0, nothing) do mha, x, m
            a, s = mha(x,x,x, mask=m, return_score=true)
            sum(sin.(a)) + sum(cos.(s))
        end

        @test isapprox(gradN[1].iqproj.weight, gradT[1].iqproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.weight, gradT[1].ikproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].ivproj.weight, gradT[1].ivproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].oproj.weight, gradT[1].oproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].iqproj.bias, gradT[1].iqproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.bias, gradT[1].ikproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].ivproj.bias, gradT[1].ivproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].oproj.bias, gradT[1].oproj.bias; atol = 1e-4)
        @test gradN[2] ≈ gradT[2]
        @test gradN[3] == gradT[3]
    end

    @testset "dropout w/o mask" begin
        rng1 = Xoshiro()
        rng2 = copy(rng1)
        dp1 = dropoutF(; rng = rng1, p = 0.5)
        dp2 = dropoutF(; rng = rng2, p = 0.5)
        x = randn(Float32, input_dims, 3, 2)
        mha = MultiheadAttention(head, input_dims, head_dims, output_dims;
                                 future = true, drop = dp2)

        mha = device(mha)
        x = device(x)

        @test mha(x,x,x) ≈ atten(mha, x; p = dp1)

        gradN = Flux.gradient(mha, x) do mha, x
            sum(sin.(atten(mha, x; p = dp1)))
        end
        gradT = Flux.gradient(mha, x) do mha, x
            sum(sin.(mha(x,x,x)))
        end

        @test isapprox(gradN[1].iqproj.weight, gradT[1].iqproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.weight, gradT[1].ikproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].ivproj.weight, gradT[1].ivproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].oproj.weight, gradT[1].oproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].iqproj.bias, gradT[1].iqproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.bias, gradT[1].ikproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].ivproj.bias, gradT[1].ivproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].oproj.bias, gradT[1].oproj.bias; atol = 1e-4)
        @test gradN[2] ≈ gradT[2]
    end

    @testset "dropout w/ future mask" begin
        rng1 = Xoshiro()
        rng2 = copy(rng1)
        dp1 = dropoutF(; rng = rng1, p = 0.5)
        dp2 = dropoutF(; rng = rng2, p = 0.5)
        x = randn(Float32, input_dims, 3, 2)
        mha = MultiheadAttention(head, input_dims, head_dims, output_dims;
                                 future = false, drop = dp2)

        mha = device(mha)
        x = device(x)

        @test mha(x,x,x) ≈ atten(mha, x, CausalMask(); p = dp1)

        gradN = Flux.gradient(mha, x) do mha, x
            sum(sin.(atten(mha, x, CausalMask(); p = dp1)))
        end
        gradT = Flux.gradient(mha, x) do mha, x
            sum(sin.(mha(x,x,x)))
        end

        @test isapprox(gradN[1].iqproj.weight, gradT[1].iqproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.weight, gradT[1].ikproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].ivproj.weight, gradT[1].ivproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].oproj.weight, gradT[1].oproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].iqproj.bias, gradT[1].iqproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.bias, gradT[1].ikproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].ivproj.bias, gradT[1].ivproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].oproj.bias, gradT[1].oproj.bias; atol = 1e-4)
        @test gradN[2] ≈ gradT[2]
    end

    @testset "dropout w/ future + length mask" begin
        rng1 = Xoshiro()
        rng2 = copy(rng1)
        dp1 = dropoutF(; rng = rng1, p = 0.5)
        dp2 = dropoutF(; rng = rng2, p = 0.5)
        x = randn(Float32, input_dims, 6, 2)
        mha = MultiheadAttention(head, input_dims, head_dims, output_dims;
                                 future = false, drop = dp2)

        q_len = [3, 5]
        k_len = [4, 4]
        lmask = BatchedMask(BiLengthMask(q_len, k_len))
        mask = CausalMask() & lmask

        mask2 = device(CausalMask() & BatchedMask(LengthMask(q_len)))
        mask3 = device(CausalMask() & BatchedMask(SymLengthMask(q_len)))

        old_q_len = Old_Impl.getmask(map(l->ones(l), q_len))
        old_k_len = Old_Impl.getmask(map(l->ones(l), k_len))
        old_mask = zeros(Float32, 6, 6, 2)
        old_mask[1:4, 1:5, :] .= Old_Impl.getmask(old_k_len, old_q_len)

        mha = device(mha)
        x = device(x)
        mask = device(mask)
        old_mask = device(old_mask)

        @test mha(x,x,x, mask=old_mask) ≈ atten(mha, x, mask; p = dp1)
        @test atten(mha, x, mask2; p = dropoutF(; rng = Xoshiro(0), p = 0.5)) ≈ atten(mha, x, mask3; p = dropoutF(; rng = Xoshiro(0), p = 0.5))

        gradN = Flux.gradient(mha, x, mask) do mha, x, m
            sum(sin.(atten(mha, x, m; p = dp1)))
        end
        gradT = Flux.gradient(mha, x, old_mask) do mha, x, m
            sum(sin.(mha(x,x,x, mask=m)))
        end

        @test isapprox(gradN[1].iqproj.weight, gradT[1].iqproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.weight, gradT[1].ikproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].ivproj.weight, gradT[1].ivproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].oproj.weight, gradT[1].oproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].iqproj.bias, gradT[1].iqproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.bias, gradT[1].ikproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].ivproj.bias, gradT[1].ivproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].oproj.bias, gradT[1].oproj.bias; atol = 1e-4)
        @test gradN[2] ≈ gradT[2]
        @test gradN[3] == gradT[3]
    end

    @testset "dropout w/ future + reverse length mask" begin
        rng1 = Xoshiro()
        rng2 = copy(rng1)
        dp1 = dropoutF(; rng = rng1, p = 0.5)
        dp2 = dropoutF(; rng = rng2, p = 0.5)
        x = randn(Float32, input_dims, 6, 2)
        mha = MultiheadAttention(head, input_dims, head_dims, output_dims;
                                 future = false, drop = dp2)

        q_len = [3, 5]
        k_len = [4, 4]
        lmask = BatchedMask(RevBiLengthMask(q_len, k_len))
        mask = CausalMask() & lmask

        mask2 = device(CausalMask() & BatchedMask(RevLengthMask(q_len)))
        mask3 = device(CausalMask() & BatchedMask(RevSymLengthMask(q_len)))

        old_q_len = reverse!(reverse!(Old_Impl.getmask(map(l->ones(l), q_len)); dims=1); dims=2)
        old_k_len = reverse!(reverse!(Old_Impl.getmask(map(l->ones(l), k_len)); dims=1); dims=2)
        old_mask = zeros(Float32, 6, 6, 2)
        old_mask[3:6, 2:6, :] .= Old_Impl.getmask(old_k_len, old_q_len)

        mha = device(mha)
        x = device(x)
        mask = device(mask)
        old_mask = device(old_mask)

        @test mha(x,x,x, mask=old_mask) ≈ atten(mha, x, mask; p = dp1)
        @test atten(mha, x, mask2; p = dropoutF(; rng = Xoshiro(0), p = 0.3)) ≈ atten(mha, x, mask3; p = dropoutF(; rng = Xoshiro(0), p = 0.3))

        gradN = Flux.gradient(mha, x, mask) do mha, x, m
            sum(sin.(atten(mha, x, m; p = dp1)))
        end
        gradT = Flux.gradient(mha, x, old_mask) do mha, x, m
            sum(sin.(mha(x,x,x, mask=m)))
        end

        @test isapprox(gradN[1].iqproj.weight, gradT[1].iqproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.weight, gradT[1].ikproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].ivproj.weight, gradT[1].ivproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].oproj.weight, gradT[1].oproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].iqproj.bias, gradT[1].iqproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.bias, gradT[1].ikproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].ivproj.bias, gradT[1].ivproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].oproj.bias, gradT[1].oproj.bias; atol = 1e-4)
        @test gradN[2] ≈ gradT[2]
        @test gradN[3] == gradT[3]
    end

    @testset "dropout w/ batched mask" begin
        rng1 = Xoshiro()
        rng2 = copy(rng1)
        dp1 = dropoutF(; rng = rng1, p = 0.5)
        dp2 = dropoutF(; rng = rng2, p = 0.5)
        x = randn(Float32, input_dims, 6, 2)
        mha = MultiheadAttention(head, input_dims, head_dims, output_dims;
                                 future = false, drop = dp2)

        q_len = [3, 5]
        k_len = [4, 4]
        lmask = RevBiLengthMask(q_len, k_len)
        mask = CausalMask() & lmask

        mask2 = device(CausalMask() & RevLengthMask(q_len))
        mask3 = device(CausalMask() & RevSymLengthMask(q_len))

        old_q_len = reverse!(reverse!(Old_Impl.getmask(map(l->ones(l), q_len)); dims=1); dims=2)
        old_k_len = reverse!(reverse!(Old_Impl.getmask(map(l->ones(l), k_len)); dims=1); dims=2)
        old_mask = zeros(Float32, 6, 6, 2)
        old_mask[3:6, 2:6, :] .= Old_Impl.getmask(old_k_len, old_q_len)

        mha = device(mha)
        x = device(x)
        mask = device(mask)
        old_mask = device(old_mask)

        @test mha(x,x,x, mask=old_mask) ≈ atten(mha, x, BatchedMask(mask); p = dp1)
        @test atten(mha, x, BatchedMask(mask2); p = dropoutF(; rng = Xoshiro(0), p = 0.3)) ≈ atten(mha, x, BatchedMask(mask3); p = dropoutF(; rng = Xoshiro(0), p = 0.3))

        gradN = Flux.gradient(mha, x, mask) do mha, x, m
            sum(sin.(atten(mha, x, BatchedMask(m); p = dp1)))
        end
        gradT = Flux.gradient(mha, x, old_mask) do mha, x, m
            sum(sin.(mha(x,x,x, mask=m)))
        end

        @test isapprox(gradN[1].iqproj.weight, gradT[1].iqproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.weight, gradT[1].ikproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].ivproj.weight, gradT[1].ivproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].oproj.weight, gradT[1].oproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].iqproj.bias, gradT[1].iqproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.bias, gradT[1].ikproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].ivproj.bias, gradT[1].ivproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].oproj.bias, gradT[1].oproj.bias; atol = 1e-4)
        @test gradN[2] ≈ gradT[2]
        @test gradN[3] == gradT[3]
    end

    @testset "dropout w/ score" begin
        rng1 = Xoshiro()
        rng2 = copy(rng1)
        dp1 = dropoutF(; rng = rng1, p = 0.5)
        dp2 = dropoutF(; rng = rng2, p = 0.5)
        x = randn(Float32, input_dims, 6, 2)
        mha = MultiheadAttention(head, input_dims, head_dims, output_dims;
                                 future = false, drop = dp2)

        q_len = [3, 5]
        k_len = [4, 4]
        lmask = RevBiLengthMask(q_len, k_len)
        mask = CausalMask() & lmask

        mask2 = device(CausalMask() & RevLengthMask(q_len))
        mask3 = device(CausalMask() & RevSymLengthMask(q_len))

        old_q_len = reverse!(reverse!(Old_Impl.getmask(map(l->ones(l), q_len)); dims=1); dims=2)
        old_k_len = reverse!(reverse!(Old_Impl.getmask(map(l->ones(l), k_len)); dims=1); dims=2)
        old_mask = zeros(Float32, 6, 6, 2)
        old_mask[3:6, 2:6, :] .= Old_Impl.getmask(old_k_len, old_q_len)

        mha = device(mha)
        x = device(x)
        mask = device(mask)
        old_mask = device(old_mask)

        @test mha(x,x,x, mask=old_mask, return_score=true)[1] ≈ atten(mha, x, BatchedMask(mask), true; p = dp1)[1]
        @test reshape(mha(x,x,x, mask=old_mask, return_score=true)[2],6,6,head,2) ≈ atten(mha, x, BatchedMask(mask), true; p = dp1)[2]
        @test atten(mha, x, BatchedMask(mask2), true; p = dropoutF(; rng = Xoshiro(0), p = 0.5))[1] ≈ atten(mha, x, BatchedMask(mask3), true; p = dropoutF(; rng = Xoshiro(0), p = 0.5))[1]
        @test atten(mha, x, BatchedMask(mask2), true; p = dropoutF(; rng = Xoshiro(0), p = 0.5))[2] ≈ atten(mha, x, BatchedMask(mask3), true; p = dropoutF(; rng = Xoshiro(0), p = 0.5))[2]

        gradN = Flux.gradient(mha, x, mask) do mha, x, m
            a, s = atten(mha, x, BatchedMask(m), true; p = dp1)
            sum(sin.(a)) + sum(cos.(s))
        end
        gradT = Flux.gradient(mha, x, old_mask) do mha, x, m
            a, s = mha(x,x,x, mask=m, return_score=true)
            sum(sin.(a)) + sum(cos.(s))
        end

        @test isapprox(gradN[1].iqproj.weight, gradT[1].iqproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.weight, gradT[1].ikproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].ivproj.weight, gradT[1].ivproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].oproj.weight, gradT[1].oproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].iqproj.bias, gradT[1].iqproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.bias, gradT[1].ikproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].ivproj.bias, gradT[1].ivproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].oproj.bias, gradT[1].oproj.bias; atol = 1e-4)
        @test gradN[2] ≈ gradT[2]
        @test gradN[3] == gradT[3]

        gradN = Flux.gradient(mha, x, mask) do mha, x, m
            a, s = atten(mha, x, BatchedMask(m), true; p = dp1)
            sum(cos.(s))
        end
        gradT = Flux.gradient(mha, x, old_mask) do mha, x, m
            a, s = mha(x,x,x, mask=m, return_score=true)
            sum(cos.(s))
        end

        @test isapprox(gradN[1].iqproj.weight, gradT[1].iqproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.weight, gradT[1].ikproj.weight; atol = 1e-4)
        @test gradN[1].ivproj == gradT[1].ivproj
        @test gradN[1].oproj == gradT[1].oproj
        @test isapprox(gradN[1].iqproj.bias, gradT[1].iqproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.bias, gradT[1].ikproj.bias; atol = 1e-4)
        @test gradN[2] ≈ gradT[2]
        @test gradN[3] == gradT[3]

        x0 = device(randn(Float32, input_dims, 6))
        @test mha(x0,x0,x0; return_score=true)[1] ≈ atten(mha, x0, CausalMask(), true; p = dp1)[1]
        @test reshape(mha(x0,x0,x0; return_score=true)[2],6,6,head,1) ≈ atten(mha, x0, CausalMask(), true; p = dp1)[2]

        gradN = Flux.gradient(mha, x0, CausalMask()) do mha, x, m
            a, s = atten(mha, x, BatchedMask(m), true; p = dp1)
            sum(sin.(a)) + sum(cos.(s))
        end
        gradT = Flux.gradient(mha, x0, nothing) do mha, x, m
            a, s = mha(x,x,x, mask=m, return_score=true)
            sum(sin.(a)) + sum(cos.(s))
        end

        @test isapprox(gradN[1].iqproj.weight, gradT[1].iqproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.weight, gradT[1].ikproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].ivproj.weight, gradT[1].ivproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].oproj.weight, gradT[1].oproj.weight; atol = 1e-4)
        @test isapprox(gradN[1].iqproj.bias, gradT[1].iqproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].ikproj.bias, gradT[1].ikproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].ivproj.bias, gradT[1].ivproj.bias; atol = 1e-4)
        @test isapprox(gradN[1].oproj.bias, gradT[1].oproj.bias; atol = 1e-4)
        @test gradN[2] ≈ gradT[2]
        @test gradN[3] == gradT[3]
    end

end
