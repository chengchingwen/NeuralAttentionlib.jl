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
    hq = (move_head_dim_out ∘ split_head)(head, q)
    hk = (move_head_dim_out ∘ split_head)(head, k)
    hv = (move_head_dim_out ∘ split_head)(head, v)
    t = generic_qkv_attention(mixingf, scoref, hq, hk, hv, args...)
    if t isa NamedTuple
        a = t.hidden_state
        y = (merge_head ∘ move_head_dim_in)(a)
        return merge(t, (hidden_state = y,))
    else
        a = t
        return (merge_head ∘ move_head_dim_in)(a)
    end
end

function naive_qkv_attention(q::AbstractArray, k::AbstractArray, v::AbstractArray, mask=nothing)
    return generic_qkv_attention(
        weighted_sum_mixing,
        normalized_score(NNlib.softmax) $ masked_score(GenericAttenMaskOp(), mask) $ scaled_dot_product_score,
        q, k, v
    )
end

function multihead_qkv_attention(head::Integer, q::AbstractArray, k::AbstractArray, v::AbstractArray, mask=nothing)
    return generic_multihead_qkv_attention(
        weighted_sum_mixing,
        normalized_score(NNlib.softmax) $ masked_score(GenericAttenMaskOp(), mask) $ scaled_dot_product_score,
        head, q, k, v
    )
end

function naive_qkv_attention(::typeof(score_returning), q::AbstractArray, k::AbstractArray, v::AbstractArray, mask=nothing)
    return generic_qkv_attention(
        score_returning(weighted_sum_mixing),
        normalized_score(NNlib.softmax) $ masked_score(GenericAttenMaskOp(), mask) $ scaled_dot_product_score,
        q, k, v
    )
end

function multihead_qkv_attention(::typeof(score_returning), head::Integer, q::AbstractArray, k::AbstractArray, v::AbstractArray, mask=nothing)
    return generic_multihead_qkv_attention(
        score_returning(weighted_sum_mixing),
        normalized_score(NNlib.softmax) $ masked_score(GenericAttenMaskOp(), mask) $ scaled_dot_product_score,
        head, q, k, v
    )
end
