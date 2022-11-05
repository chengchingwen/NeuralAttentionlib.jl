@testset "functional" begin
    using Statistics
    using Flux
    using ZipFile
    using Pickle: npyload
    using NeuralAttentionlib.Matmul
    using NeuralAttentionlib: as_collapsed, dot_product_score, normalized_score,
      scalar_relative_position_embedding, get_scalar_relative_position_embeddings,
      t5_bucketed_position_id, t5_causal_bucketed_position_id,
      layer_norm, rms_layer_norm

    @testset "score" begin
        if !USE_CUDA
            @testset "AD" begin
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
