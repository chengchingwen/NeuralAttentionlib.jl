@testset "MHA" begin
    using NeuralAttentionlib.Masks
    using NeuralAttentionlib: score_returning
    head = 4
    input_dims = 7
    head_dims = 5
    output_dims = 8

    function atten(mha, x, mask=nothing, return_score = false, p = nothing)
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
        x = randn(input_dims, 3, 2)
        mha = MultiheadAttention(head, input_dims, head_dims, output_dims;
                                 future = true, pdrop = 0) # disable dropout

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
        x = randn(input_dims, 3, 2)
        mha = MultiheadAttention(head, input_dims, head_dims, output_dims;
                                 future = false, pdrop = 0) # disable dropout

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
        x = randn(input_dims, 6, 2)
        mha = MultiheadAttention(head, input_dims, head_dims, output_dims;
                                 future = false, pdrop = 0) # disable dropout

        q_len = [3, 5]
        k_len = [4, 4]
        lmask = BatchedMask(BiLengthMask(q_len, k_len))
        mask = CausalMask() & lmask

        mask2 = device(CausalMask() & BatchedMask(LengthMask(q_len)))
        mask3 = device(CausalMask() & BatchedMask(SymLengthMask(q_len)))

        old_q_len = Old_Impl.getmask(map(l->ones(l), q_len))
        old_k_len = Old_Impl.getmask(map(l->ones(l), k_len))
        old_mask = zeros(6, 6, 2)
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
        x = randn(input_dims, 6, 2)
        mha = MultiheadAttention(head, input_dims, head_dims, output_dims;
                                 future = false, pdrop = 0) # disable dropout

        q_len = [3, 5]
        k_len = [4, 4]
        lmask = BatchedMask(RevBiLengthMask(q_len, k_len))
        mask = CausalMask() & lmask

        mask2 = device(CausalMask() & BatchedMask(RevLengthMask(q_len)))
        mask3 = device(CausalMask() & BatchedMask(RevSymLengthMask(q_len)))

        old_q_len = reverse!(reverse!(Old_Impl.getmask(map(l->ones(l), q_len)); dims=1); dims=2)
        old_k_len = reverse!(reverse!(Old_Impl.getmask(map(l->ones(l), k_len)); dims=1); dims=2)
        old_mask = zeros(6, 6, 2)
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
        x = randn(input_dims, 6, 2)
        mha = MultiheadAttention(head, input_dims, head_dims, output_dims;
                                 future = false, pdrop = 0) # disable dropout

        q_len = [3, 5]
        k_len = [4, 4]
        lmask = RevBiLengthMask(q_len, k_len)
        mask = CausalMask() & lmask

        mask2 = device(CausalMask() & RevLengthMask(q_len))
        mask3 = device(CausalMask() & RevSymLengthMask(q_len))

        old_q_len = reverse!(reverse!(Old_Impl.getmask(map(l->ones(l), q_len)); dims=1); dims=2)
        old_k_len = reverse!(reverse!(Old_Impl.getmask(map(l->ones(l), k_len)); dims=1); dims=2)
        old_mask = zeros(6, 6, 2)
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
        x = randn(input_dims, 6, 2)
        mha = MultiheadAttention(head, input_dims, head_dims, output_dims;
                                 future = false, pdrop = 0) # disable dropout

        q_len = [3, 5]
        k_len = [4, 4]
        lmask = RevBiLengthMask(q_len, k_len)
        mask = CausalMask() & lmask

        mask2 = device(CausalMask() & RevLengthMask(q_len))
        mask3 = device(CausalMask() & RevSymLengthMask(q_len))

        old_q_len = reverse!(reverse!(Old_Impl.getmask(map(l->ones(l), q_len)); dims=1); dims=2)
        old_k_len = reverse!(reverse!(Old_Impl.getmask(map(l->ones(l), k_len)); dims=1); dims=2)
        old_mask = zeros(6, 6, 2)
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
    end

end
