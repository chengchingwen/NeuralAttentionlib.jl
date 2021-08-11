@testset "MHA" begin
    using NeuralAttentionlib: CausalMask, BiLengthMask, BatchedMask
    head = 4
    input_dims = 7
    head_dims = 5
    output_dims = 8

    function atten(mha, x, mask=nothing)
        q = mha.iqproj(x)
        k = mha.ikproj(x)
        v = mha.ivproj(x)
        a = NeuralAttentionlib.multihead_qkv_attention(4, q, k, v, mask)
        return mha.oproj(a)
    end

    @testset "w/o mask" begin
        x = randn(input_dims, 3, 2)
        mha = MultiheadAttention(head, input_dims, head_dims, output_dims;
                                 future = true, pdrop = 0) # disable dropout

        @test mha(x,x,x) ≈ atten(mha, x)

        gradN = Flux.gradient(mha, x) do mha, x
            sum(sin.(atten(mha, x)))
        end
        gradT = Flux.gradient(mha, x) do mha, x
            sum(sin.(mha(x,x,x)))
        end

        @test isapprox(gradN[1].iqproj.weight, gradT[1].iqproj.weight; atol = 1e-12)
        @test isapprox(gradN[1].ikproj.weight, gradT[1].ikproj.weight; atol = 1e-12)
        @test isapprox(gradN[1].ivproj.weight, gradT[1].ivproj.weight; atol = 1e-12)
        @test isapprox(gradN[1].oproj.weight, gradT[1].oproj.weight; atol = 1e-12)
        @test isapprox(gradN[1].iqproj.bias, gradT[1].iqproj.bias; atol = 1e-12)
        @test isapprox(gradN[1].ikproj.bias, gradT[1].ikproj.bias; atol = 1e-12)
        @test isapprox(gradN[1].ivproj.bias, gradT[1].ivproj.bias; atol = 1e-12)
        @test isapprox(gradN[1].oproj.bias, gradT[1].oproj.bias; atol = 1e-12)
        @test gradN[2] ≈ gradT[2]
    end

    @testset "w/ future mask" begin
        x = randn(input_dims, 3, 2)
        mha = MultiheadAttention(head, input_dims, head_dims, output_dims;
                                 future = false, pdrop = 0) # disable dropout

        @test mha(x,x,x) ≈ atten(mha, x, CausalMask())

        gradN = Flux.gradient(mha, x) do mha, x
            sum(sin.(atten(mha, x, CausalMask())))
        end
        gradT = Flux.gradient(mha, x) do mha, x
            sum(sin.(mha(x,x,x)))
        end

        @test isapprox(gradN[1].iqproj.weight, gradT[1].iqproj.weight; atol = 1e-12)
        @test isapprox(gradN[1].ikproj.weight, gradT[1].ikproj.weight; atol = 1e-12)
        @test isapprox(gradN[1].ivproj.weight, gradT[1].ivproj.weight; atol = 1e-12)
        @test isapprox(gradN[1].oproj.weight, gradT[1].oproj.weight; atol = 1e-12)
        @test isapprox(gradN[1].iqproj.bias, gradT[1].iqproj.bias; atol = 1e-12)
        @test isapprox(gradN[1].ikproj.bias, gradT[1].ikproj.bias; atol = 1e-12)
        @test isapprox(gradN[1].ivproj.bias, gradT[1].ivproj.bias; atol = 1e-12)
        @test isapprox(gradN[1].oproj.bias, gradT[1].oproj.bias; atol = 1e-12)
        @test gradN[2] ≈ gradT[2]
    end

    @testset "w/ future + length mask" begin
        x = randn(input_dims, 6, 2)
        mha = MultiheadAttention(head, input_dims, head_dims, output_dims;
                                 future = false, pdrop = 0) # disable dropout

        q_len = [3, 5]
        k_len = [4, 4]
        lmask = BatchedMask(BiLengthMask(q_len, k_len), -1)
        mask = CausalMask() & lmask

        old_q_len = Old_Impl.getmask(map(l->ones(l), q_len))
        old_k_len = Old_Impl.getmask(map(l->ones(l), k_len))
        old_mask = zeros(6, 6, 2)
        old_mask[1:4, 1:5, :] .= Old_Impl.getmask(old_k_len, old_q_len)

        @test mha(x,x,x, mask=old_mask) ≈ atten(mha, x, mask)

        gradN = Flux.gradient(mha, x, mask) do mha, x, m
            sum(sin.(atten(mha, x, m)))
        end
        gradT = Flux.gradient(mha, x, old_mask) do mha, x, m
            sum(sin.(mha(x,x,x, mask=m)))
        end

        @test isapprox(gradN[1].iqproj.weight, gradT[1].iqproj.weight; atol = 1e-12)
        @test isapprox(gradN[1].ikproj.weight, gradT[1].ikproj.weight; atol = 1e-12)
        @test isapprox(gradN[1].ivproj.weight, gradT[1].ivproj.weight; atol = 1e-12)
        @test isapprox(gradN[1].oproj.weight, gradT[1].oproj.weight; atol = 1e-12)
        @test isapprox(gradN[1].iqproj.bias, gradT[1].iqproj.bias; atol = 1e-12)
        @test isapprox(gradN[1].ikproj.bias, gradT[1].ikproj.bias; atol = 1e-12)
        @test isapprox(gradN[1].ivproj.bias, gradT[1].ivproj.bias; atol = 1e-12)
        @test isapprox(gradN[1].oproj.bias, gradT[1].oproj.bias; atol = 1e-12)
        @test gradN[2] ≈ gradT[2]
        @test gradN[3] == gradT[3]
    end

end
