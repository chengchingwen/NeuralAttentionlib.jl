@testset "functional" begin
    using ZipFile
    using Pickle: npyload
    using NeuralAttentionlib.Matmul
    using NeuralAttentionlib: as_collapsed, dot_product_score, normalized_score,
      scalar_relative_position_embedding, get_scalar_relative_position_embeddings,
      t5_bucketed_position_id, t5_causal_bucketed_position_id

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

end
