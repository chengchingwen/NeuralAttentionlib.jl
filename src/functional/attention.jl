@inline function generic_qkv_attention(mixingf, scoref, q, k, v, args...)
    return mixing(mixingf, as_collapsed(v), scoref, as_collapsed(q), as_collapsed(k), args...)
end

@inline function generic_multihead_qkv_attention(
    mixingf, scoref,
    head::Integer,
    q::AbstractArray, k::AbstractArray, v::AbstractArray,
    args...
)
    return generic_multihead_qkv_attention(
        mixingf, scoref,
        head, as_collapsed(q), as_collapsed(k), as_collapsed(v), args...
    )
end

@inline function generic_multihead_qkv_attention(
    mixingf, scoref,
    head::Integer,
    q::CollapsedDimsArray, k::CollapsedDimsArray, v::CollapsedDimsArray,
    args...
)
    hq = _split_and_move_head(head, q)
    hk = _split_and_move_head(head, k)
    hv = _split_and_move_head(head, v)
    t = generic_qkv_attention(mixingf, scoref, hq, hk, hv, args...)
    return _move_and_merge_head(t)
end

naive_attention_score() = normalized_score(NNlib.softmax) $ scaled_dot_product_score
naive_attention_score(mask) = normalized_score(NNlib.softmax) $ masked_score(GenericMaskOp(), mask) $ scaled_dot_product_score
naive_attention_score(mask, p) = dropout_score(p) $ naive_attention_score(mask)

naive_qkv_attention(q::AbstractArray, k::AbstractArray, v::AbstractArray, args...) =
    generic_qkv_attention(weighted_sum_mixing, naive_attention_score(args...), q, k, v)

naive_qkv_attention(::typeof(score_returning), q::AbstractArray, k::AbstractArray, v::AbstractArray, args...) =
    generic_qkv_attention(score_returning(weighted_sum_mixing), naive_attention_score(args...), q, k, v)

multihead_qkv_attention(head::Integer, q::AbstractArray, k::AbstractArray, v::AbstractArray, args...) =
    generic_multihead_qkv_attention(weighted_sum_mixing, naive_attention_score(args...), head, q, k, v)

multihead_qkv_attention(::typeof(score_returning), head::Integer, q::AbstractArray, k::AbstractArray, v::AbstractArray, args...) =
    generic_multihead_qkv_attention(score_returning(weighted_sum_mixing), naive_attention_score(args...), head, q, k, v)


@inline function generic_grouped_query_attention(
    mixingf, scoref,
    head::Integer, group::Integer,
    q::AbstractArray, k::AbstractArray, v::AbstractArray,
    args...
)
    return generic_grouped_query_attention(
        mixingf, scoref,
        head, group, as_collapsed(q), as_collapsed(k), as_collapsed(v), args...
    )
end

@inline function generic_grouped_query_attention(
    mixingf, scoref,
    head::Integer, group::Integer,
    q::CollapsedDimsArray, k::CollapsedDimsArray, v::CollapsedDimsArray,
    args...
)
    rp, rem = divrem(head, group)
    @assert iszero(rem)
    hq = _split_and_move_head(head, q)
    hk = repeat_head_group(rp, _split_and_move_head(group, k))
    hv = repeat_head_group(rp, _split_and_move_head(group, v))
    t = generic_qkv_attention(mixingf, scoref, hq, hk, hv, args...)
    return _move_and_merge_head(t)
end

grouped_query_attention(head::Integer, group::Integer, q::AbstractArray, k::AbstractArray, v::AbstractArray, args...) =
    generic_grouped_query_attention(weighted_sum_mixing, naive_attention_score(args...), head, group, q, k, v)

grouped_query_attention(::typeof(score_returning), head::Integer, group::Integer, q::AbstractArray, k::AbstractArray, v::AbstractArray, args...) =
    generic_grouped_query_attention(score_returning(weighted_sum_mixing), naive_attention_score(args...), head, group, q, k, v)
