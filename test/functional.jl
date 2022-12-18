@testset "functional" begin
    using Statistics
    using Flux
    using ZipFile
    using ChainRulesCore
    using Pickle: npyload
    using NeuralAttentionlib.Matmul
    using NeuralAttentionlib: as_collapsed, scaled_dot_product_score, dot_product_score, normalized_score,
      biased_score, scalar_relative_position_embedding, get_scalar_relative_position_embeddings,
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

end
