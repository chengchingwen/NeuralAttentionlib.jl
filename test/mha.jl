@testset "MHA" begin
    head = 4
    input_dims = 7
    head_dims = 5
    output_dims = 8

    x = randn(7, 3, 2)
    mha = MultiheadAttention(head, input_dims, head_dims, output_dims;
                             future = true, pdrop = 0) # disable dropout

    function atten(mha, x)
        q = mha.iqproj(x)
        k = mha.ikproj(x)
        v = mha.ivproj(x)
        a = NeuralAttentionlib.multihead_qkv_attention(4, q, k, v)
        return mha.oproj(a)
    end        
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
