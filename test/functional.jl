@testset "functional" begin
    using Statistics
    using Flux
    using Flux.Zygote
    using ZipFile
    using ChainRulesCore
    using Pickle: npyload
    using NeuralAttentionlib.Matmul
    using NeuralAttentionlib.Functional
    using NeuralAttentionlib: as_collapsed,
      get_scalar_relative_position_embeddings,
      t5_bucketed_position_id, t5_causal_bucketed_position_id,
      layer_norm, rms_layer_norm, get_sincos_position_embeddings

    @testset "score" begin
        if !USE_CUDA
            @testset "AD" begin
                test_rrule(dot_product_score, randn(5, 3, 2), randn(5, 4, 2); check_inferred = false)
                test_rrule(dot_product_score, randn(5, 3, 2, 2), randn(5, 4, 2, 2))
                test_rrule(scaled_dot_product_score, randn(5, 3, 2), randn(5, 4, 2); check_inferred = false)
                test_rrule(scaled_dot_product_score, randn(5, 3, 2, 2), randn(5, 4, 2, 2))
                test_rrule(scaled_dot_product_score, randn(5, 3, 2), randn(5, 4, 2), 0.5; check_inferred = false)
                test_rrule(scaled_dot_product_score, randn(5, 3, 2, 2), randn(5, 4, 2, 2), 0.5)

                f2(x) = 2 .* x
                f3(x) = 3 .* x
                function ChainRulesCore.rrule(::typeof(f2), x)
                    y = f2(x)
                    pullback(Ȳ) = (NoTangent(), 2 .* unthunk(Ȳ))
                    return y, pullback
                end
                function ChainRulesCore.rrule(::typeof(f3), x)
                    y = f3(x)
                    pullback(Ȳ) = (NoTangent(), 3 .* unthunk(Ȳ))
                    return y, pullback
                end
                test_rrule(dot_product_score, f2, randn(5, 3, 2), randn(5, 4, 2); check_inferred = false)
                test_rrule(dot_product_score, f2, randn(5, 3, 2, 2), randn(5, 4, 2, 2))
                test_rrule(dot_product_score, f2, f3, randn(5, 3, 2), randn(5, 4, 2); check_inferred = false)
                test_rrule(dot_product_score, f2, f3, randn(5, 3, 2, 2), randn(5, 4, 2, 2))
                test_rrule(scaled_dot_product_score, f2, randn(5, 3, 2), randn(5, 4, 2); check_inferred = false)
                test_rrule(scaled_dot_product_score, f2, randn(5, 3, 2, 2), randn(5, 4, 2, 2))
                test_rrule(scaled_dot_product_score, f2, f3, randn(5, 3, 2), randn(5, 4, 2); check_inferred = false)
                test_rrule(scaled_dot_product_score, f2, f3, randn(5, 3, 2, 2), randn(5, 4, 2, 2))
                test_rrule(scaled_dot_product_score, 0.5, f2, randn(5, 3, 2), randn(5, 4, 2); check_inferred = false)
                test_rrule(scaled_dot_product_score, 0.5, f2, randn(5, 3, 2, 2), randn(5, 4, 2, 2))
                test_rrule(scaled_dot_product_score, 0.5, f2, f3, randn(5, 3, 2), randn(5, 4, 2); check_inferred = false)
                test_rrule(scaled_dot_product_score, 0.5, f2, f3, randn(5, 3, 2, 2), randn(5, 4, 2, 2))

                test_rrule(biased_score, randn(4, 3), dot_product_score, randn(5, 3, 2), randn(5, 4, 2);
                           check_inferred = false)
                test_rrule(biased_score, randn(4, 3, 2), dot_product_score, randn(5, 3, 2), randn(5, 4, 2);
                           check_inferred = false)
                test_rrule(biased_score, randn(4, 3), dot_product_score, randn(5, 3, 2, 2), randn(5, 4, 2, 2))
                test_rrule(biased_score, randn(4, 3, 2), dot_product_score, randn(5, 3, 2, 2), randn(5, 4, 2, 2))
                test_rrule(
                    normalized_score, NNlib.softmax,
                    dot_product_score,
                    CollapsedDimsArray(randn(5, 21, 3, 2), 1, 2),
                    CollapsedDimsArray(randn(5, 21, 3, 2), 1, 2),
                    ; check_inferred = false)
                test_rrule(
                    normalized_score, x-> (x .^ 2) ./ sum(x; dims=1) ,
                    dot_product_score,
                    CollapsedDimsArray(randn(5, 21, 3, 2), 1, 2),
                    CollapsedDimsArray(randn(5, 21, 3, 2), 1, 2),
                    ; check_inferred = false)
            end
        end
    end

    @testset "scalar_relative_position_embedding" begin
        zipfd = ZipFile.Reader(joinpath(@__DIR__, "t5ids.zip"))
        for zipfile in zipfd.files
            match_file = match(r"t5ids/t5(_causal)?_rel_id_(\d+)x(\d+)_(\d+)_(\d+).pkl", zipfile.name)
            isnothing(match_file) && continue
            _causal, _q, _k, _b, _m = match_file.captures
            (q, k, b, m) = parse.(Int, (_q, _k, _b, _m))
            data = npyload(zipfile)
            if isnothing(_causal)
                @test Broadcast.materialize(t5_bucketed_position_id(b, m, trues(q, k)))' .- 1 == data
            else
                @test Broadcast.materialize(t5_causal_bucketed_position_id(b, m, trues(q, k)))' .- 1 == data
            end
        end

        if !USE_CUDA
            @testset "AD" begin
                test_rrule(
                    scalar_relative_position_embedding, t5_bucketed_position_id(8, 20), randn(3, 8),
                    dot_product_score,
                    CollapsedDimsArray(randn(5, 21, 3, 2), 1, 2),
                    CollapsedDimsArray(randn(5, 21, 3, 2), 1, 2),
                )
                test_rrule(
                    get_scalar_relative_position_embeddings, t5_bucketed_position_id(8, 20), randn(3, 8),
                    CollapsedDimsArray(randn(21, 21, 3, 2), 1, 2),
                )
                test_rrule(
                    scalar_relative_position_embedding, t5_bucketed_position_id(8, 20), randn(3, 8),
                    dot_product_score,
                    CollapsedDimsArray(randn(5, 6, 3, 2), 1, 2),
                    CollapsedDimsArray(randn(5, 6, 3, 2), 1, 2),
                )
                test_rrule(
                    get_scalar_relative_position_embeddings, t5_bucketed_position_id(8, 20), randn(3, 8),
                    CollapsedDimsArray(randn(6, 6, 3, 2), 1, 2),
                )
                test_rrule(
                    scalar_relative_position_embedding, t5_causal_bucketed_position_id(8, 20), randn(3, 8),
                    dot_product_score,
                    CollapsedDimsArray(randn(5, 21, 3, 2), 1, 2),
                    CollapsedDimsArray(randn(5, 21, 3, 2), 1, 2),
                )
                test_rrule(
                    get_scalar_relative_position_embeddings, t5_causal_bucketed_position_id(8, 20), randn(3, 8),
                    CollapsedDimsArray(randn(21, 21, 3, 2), 1, 2),
                )
                test_rrule(
                    scalar_relative_position_embedding, t5_causal_bucketed_position_id(8, 20), randn(3, 8),
                    dot_product_score,
                    CollapsedDimsArray(randn(5, 6, 3, 2), 1, 2),
                    CollapsedDimsArray(randn(5, 6, 3, 2), 1, 2),
                )
                test_rrule(
                    get_scalar_relative_position_embeddings, t5_causal_bucketed_position_id(8, 20), randn(3, 8),
                    CollapsedDimsArray(randn(6, 6, 3, 2), 1, 2),
                )
            end
        end
    end

    @testset "sincos_position_embeddings" begin
        function PE(size, pos, i::Int)
            if rem(i, 2) == 1
                sin((pos-1)/1e4^((i-1)/size))
            else
                cos((pos-1)/1e4^((i-2)/size))
            end
        end
        function sincos_pe(size, max_len)
            embedding = Matrix{Float32}(undef, size, max_len)
            for l = 1:max_len
                map!(i->PE(size, l, i), selectdim(embedding, 2, l), 1:size)
            end
            return embedding
        end
        l2norm(x) = x ./ sqrt.(sum(x .^ 2; dims=1))
        @test get_sincos_position_embeddings(512, false, 1024) ≈ sincos_pe(512, 1024)
        @test get_sincos_position_embeddings(513, false, 1024) ≈ sincos_pe(513, 1024)
        @test get_sincos_position_embeddings(512, true, 1024) ≈ l2norm(sincos_pe(512, 1024))
        @test get_sincos_position_embeddings(513, true, 1024) ≈ l2norm(sincos_pe(513, 1024))
        x1 = drandn(512, 10, 2)
        x2 = drandn(513, 10, 2)
        @test get_sincos_position_embeddings(512, false, x1) ≈ device(sincos_pe(512, 10))
        @test get_sincos_position_embeddings(513, false, x2) ≈ device(sincos_pe(513, 10))
        @test get_sincos_position_embeddings(512, true, x1) ≈ device(l2norm(sincos_pe(512, 10)))
        @test get_sincos_position_embeddings(513, true, x2) ≈ device(l2norm(sincos_pe(513, 10)))
        i1 = drand(1:1024, 10, 2)
        i2 = drand(1:1024, 10, 2)
        @test get_sincos_position_embeddings(512, false, i1) ≈ device(sincos_pe(512, 1024))[:, i1]
        @test get_sincos_position_embeddings(513, false, i2) ≈ device(sincos_pe(513, 1024))[:, i2]
    end

    @testset "rotary position embedding" begin
        function naive_rotary_pe(x)
            sincos = get_sincos_position_embeddings(size(x, 1), false, x)
            sinθ = repeat(sincos[1:2:end, :], inner = (2, 1))
            cosθ = repeat(sincos[2:2:end, :], inner = (2, 1))
            x1 = reshape(x[1:2:end, :, :, :], 1, size(x, 1) >> 1, Base.tail(size(x))...)
            x2 = reshape(x[2:2:end, :, :, :], 1, size(x, 1) >> 1, Base.tail(size(x))...)
            y = reshape(vcat(-x2, x1), size(x))
            return x .* cosθ .+ y .* sinθ
        end
        function naive_rotary_pe_w_dim(dim, x)
            x1 = x[begin:dim, :, :, :]
            x2 = x[dim+1:end, :, :, :]
            y = naive_rotary_pe(x1)
            return vcat(y, x2)
        end
        x = drandn(512, 5, 3, 2)
        @test with_rotary_position_embedding(x) ≈ naive_rotary_pe(x)
        @test with_rotary_position_embedding(256, x) ≈ naive_rotary_pe_w_dim(256, x)
        @test with_rotary_position_embedding(256)(x) ≈ naive_rotary_pe_w_dim(256, x)
        if !USE_CUDA
            @testset "AD" begin
                x = randn(512, 5, 3, 2)
                @test Zygote.gradient(x->sum(sin.(with_rotary_position_embedding(x))), x)[1] ≈
                    Zygote.gradient(x->sum(sin.(naive_rotary_pe(x))), x)[1]
                @test Zygote.gradient(x->sum(sin.(with_rotary_position_embedding(256, x))), x)[1] ≈
                    Zygote.gradient(x->sum(sin.(naive_rotary_pe_w_dim(256, x))), x)[1]
                @test Zygote.gradient(x->sum(sin.(with_rotary_position_embedding(256)(x))), x)[1] ≈
                    Zygote.gradient(x->sum(sin.(naive_rotary_pe_w_dim(256, x))), x)[1]
                x = randn(512, 5, 2)
                @test Zygote.gradient(
                    x->sum(sin.(scaled_dot_product_score(with_rotary_position_embedding(256), x, x))), x)[1] ≈
                Zygote.gradient(
                    x->sum(sin.(scaled_dot_product_score(Base.Fix1(naive_rotary_pe_w_dim, 256), x, x))), x)[1]
                test_rrule(
                    scaled_dot_product_score, with_rotary_position_embedding, randn(512, 5, 2), randn(512, 5, 2);
                    check_inferred = false)
            end
        end
    end

    @testset "layer_norm" begin
        LN = device(LayerNorm(20))
        LN0 = device(LayerNorm(20; ϵ=0))
        LN.diag.scale .= drandn(20)
        LN.diag.bias .= drandn(20)
        LN0.diag.scale .= drandn(20)
        LN0.diag.bias .= drandn(20)
        x = drandn(20, 7)
        rms_layer_norm_naive(g, x) = g .* x ./ sqrt.(mean(x .^ 2; dims=1))

        @test isapprox(LN(x), layer_norm(LN.ϵ, LN.diag.scale, LN.diag.bias, x); atol=1e-3)
        @test rms_layer_norm_naive(LN.diag.scale, x) ≈ rms_layer_norm(LN.ϵ, LN.diag.scale, x)
        @test LN0(x) ≈ layer_norm(0, LN0.diag.scale, LN0.diag.bias, x)
        @test rms_layer_norm_naive(LN0.diag.scale, x) ≈ rms_layer_norm(0, LN0.diag.scale, x)

        x32 = drand(Float32, 100, 2) .+ 1f4
        @test isapprox(Flux.normalise(x32; dims=1, ϵ = 1f-10), layer_norm(1f-10, nothing, nothing, x32); atol = 2e-1)
        @test isapprox(
            Flux.gradient(x->sum(sin.(Flux.normalise(x; dims=1, ϵ = 1f-10))), x32)[1],
            Flux.gradient(x->sum(sin.(layer_norm(1f-10, nothing, nothing, x))), x32)[1];
            atol = 5e-1
        )

        if !USE_CUDA
            @testset "AD" begin
                g = randn(20)
                b = randn(20)
                x = randn(20, 7)
                test_rrule(layer_norm, 1e-5, g, b, x; atol = 1e-5)
                test_rrule(rms_layer_norm, 1e-5, g, x; atol = 1e-5, check_inferred = false)
                test_rrule(layer_norm, g, b, x; atol = 1e-5)
                test_rrule(rms_layer_norm, g, x; atol = 1e-5, check_inferred = false)
                test_rrule(layer_norm, 0, g, b, x)
                test_rrule(rms_layer_norm, 0, g, x; check_inferred = false)

                test_rrule(layer_norm, 1e-5, 0.5, 0.3, x; atol = 1e-5)
                test_rrule(rms_layer_norm, 1e-5, 0.5, x; atol = 1e-5, check_inferred = false)
                test_rrule(layer_norm, 0.5, 0.3, x; atol = 1e-5)
                test_rrule(rms_layer_norm, 0.5, x; atol = 1e-5, check_inferred = false)
                test_rrule(layer_norm, 0, 0.5, 0.3, x)
                test_rrule(rms_layer_norm, 0, 0.5, x; check_inferred = false)

                test_rrule(layer_norm, 1e-5, 0.5, b, x; atol = 1e-5)
                test_rrule(layer_norm, 0.5, b, x; atol = 1e-5)
                test_rrule(layer_norm, 0, 0.5, b, x)
                test_rrule(layer_norm, 1e-5, g, 0.3, x; atol = 1e-5)
                test_rrule(layer_norm, g, 0.3, x; atol = 1e-5)
                test_rrule(layer_norm, 0, g, 0.3, x)

            end
        end
    end

    @testset "attention" begin
        @testset "multihead_qkv_attention" begin
            if !USE_CUDA
                @testset "AD" begin
                    for i = 1:3
                        a = randn(20, 3, 2)
                        b = randn(20, 5, 2)
                        c = randn(20, 5, 2)
                        test_rrule(multihead_qkv_attention, 4, a, b, c; atol = 1e-5)
                    end
                end
            end
        end

        @testset "grouped_query_attention" begin
            function naive_grouped_query_attention(head, group, q, k, v)
                fdim, len, batch = size(q)
                hdim = div(fdim, head)
                hq = reshape(q, hdim, head, len, batch)
                nq = div(head, group)
                qs = ntuple(nq) do i
                    reshape(hq[:, i:nq:end, :, :], group * hdim, len, batch)
                end
                os = ntuple(nq) do i
                    reshape(multihead_qkv_attention(group, qs[i], k, v), hdim, group, len, batch)
                end
                o = reshape(cat(os...; dims=1), :, len, batch)
                return o
            end

            a = drandn(30, 3, 2)
            b = drandn(10, 5, 2)
            c = drandn(10, 5, 2)

            grad = Zygote.gradient((x,y,z)->sum(sin.(grouped_query_attention(6, 2, x, y, z))), a, b, c)
            ngrad = Zygote.gradient((x,y,z)->sum(sin.(naive_grouped_query_attention(6, 2, x, y, z))), a, b, c)
            @test grad[1] ≈ ngrad[1]
            @test grad[2] ≈ ngrad[2]
            @test grad[3] ≈ ngrad[3]

            if !USE_CUDA
                @testset "AD" begin
                    for i = 1:3
                        a = randn(30, 3, 2)
                        b = randn(10, 5, 2)
                        c = randn(10, 5, 2)
                        test_rrule(grouped_query_attention, 6, 2, a, b, c; atol = 1e-5)
                    end
                end
            end
        end
    end

end
