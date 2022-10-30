import Base.Broadcast: broadcasted

scalar_relative_position_embedding(relative_position_id_func) =
    scalar_relative_position_embedding $ relative_position_id_func
scalar_relative_position_embedding(relative_position_id_func, embedding_table::AbstractArray) =
    scalar_relative_position_embedding $ relative_position_id_func $ embedding_table
function scalar_relative_position_embedding(relative_position_id_func, embedding_table, score, args...)
    score_val = score(args...)
    relative_position_embedding =
        get_scalar_relative_position_embeddings(relative_position_id_func, embedding_table, score_val)
    return collapseddims_nonbatch(.+, score_val, relative_position_embedding)
end

function get_scalar_relative_position_embeddings(relative_position_id_func, embedding_table::AbstractArray, score)
    bdims = noncollapsed_size(score, 3)
    @assert Base.front(Base.size(embedding_table)) == Base.front(bdims) "There are $(join(Base.front(bdims), 'x')) attention heads, but get $(join(Base.front(size(embedding_table)), 'x')) scalar position embeddings."
    relative_position_id = relative_position_id_func(score)
    embeddings = similar(score, eltype(score), size(score, 1), size(score, 2), Base.front(bdims)..., 1)
    ids = similar(unwrap_collapse(score), Int32, size(relative_position_id)..., 1)
    ids .= relative_position_id
    perm = (ntuple(Base.Fix1(+, 2), static(length(bdims)) - static(1))..., 1, 2, ndims(embeddings))
    dst = PermutedDimsArray{eltype(embeddings), ndims(embeddings), perm, invperm(perm), typeof(embeddings)}(embeddings)
    NNlib.gather!(dst, embedding_table, ids)
    # embeddings = permutedims(gather(embedding_table, ids), invperm(perm))
    return embeddings
end

struct T5_bucketed_position_id_func <: Function
    n_buckets::Int
    max_distance::Int
end
(f::T5_bucketed_position_id_func)(score) = t5_bucketed_position_id(f.n_buckets, f.max_distance, score)

struct T5_causal_bucketed_position_id_func <: Function
    n_buckets::Int
    max_distance::Int
end
(f::T5_causal_bucketed_position_id_func)(score) = t5_causal_bucketed_position_id(f.n_buckets, f.max_distance, score)

t5_bucketed_position_id(n_buckets::Int, max_distance::Int) =
    T5_bucketed_position_id_func(n_buckets, max_distance)
t5_causal_bucketed_position_id(n_buckets::Int, max_distance::Int) =
    T5_causal_bucketed_position_id_func(n_buckets, max_distance)

t5_bucketed_position_id(n_buckets, max_distance, score) =
    _t5_bucketed_position_id(Val(false), n_buckets, max_distance, score)
t5_causal_bucketed_position_id(n_buckets, max_distance, score) =
    _t5_bucketed_position_id(Val(true), n_buckets, max_distance, score)

function _t5_bucketed_position_id(::Val{causal}, n_buckets, max_distance, score) where causal
    key_length = size(score, 1)
    query_length = size(score, 2)
    key_position = Base.OneTo(key_length)
    query_position = Base.OneTo(query_length)
    if causal
        relative_position = # - min.(1:query_length' .- 1:key_length, 0)
            broadcasted(max, broadcasted(-, query_position', key_position), 0)
    else
        n_buckets >>= 1
        relative_position = # 1:key_length .- 1:query_length'
            broadcasted(-, key_position, query_position')
        relative_buckets = # ifelse.(relative_position .> 0, n_buckets, 0)
            broadcasted(ifelse, broadcasted(>, relative_position, 0), n_buckets, 0)
        relative_position = broadcasted(abs, relative_position)
    end
    max_exact = n_buckets >> 1
    is_small = broadcasted(<, relative_position, max_exact)

    _Int_floor(x) = isinf(x) && x < 0 ? typemin(Int) : floor(Int, x)
    # min.(max_exact .+ floor.((log.(relative_position ./ max_exact) ./ log(max_distance / max_exact)) .* (n_buckets - max_exact)), n_buckets - 1)
    if_large = broadcasted(
        min, broadcasted(
            +, broadcasted(
                _Int_floor, broadcasted(
                    *, broadcasted(
                        /, broadcasted(log, broadcasted(/, relative_position, max_exact)),
                        log(max_distance / max_exact)),
                    n_buckets - max_exact)),
            max_exact),
        n_buckets - 1
    )
    if causal
        relative_buckets = broadcasted(ifelse, is_small, relative_position, if_large)
    else
        relative_buckets = broadcasted(+, relative_buckets, broadcasted(ifelse, is_small, relative_position, if_large))
    end
    return broadcasted(+, relative_buckets, 1)
end
