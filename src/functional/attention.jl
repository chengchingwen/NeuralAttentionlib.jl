@inline function generic_qkv_attention(mixingf, scoref, q, k, v, args...)
    return unwrap_collapse(mixing(mixingf, v, scoref, q, k, args...))
end

@inline function generic_multihead_qkv_attention(mixingf, scoref, head, q, k, v, args...)
    hq = CollapsedDimArray((move_head_dim_out ∘ split_head)(head, q), static(2), static(ndims(q)))
    hk = CollapsedDimArray((move_head_dim_out ∘ split_head)(head, k), static(2), static(ndims(k)))
    hv = CollapsedDimArray((move_head_dim_out ∘ split_head)(head, v), static(2), static(ndims(v)))
    a = generic_qkv_attention(mixingf, scoref, hq, hk, hv, args...)
    return (merge_head ∘ move_head_dim_in)(a)
end

@inline naive_qkv_attention(q, k, v, mask=nothing) = unwrap_collapse(weighted_sum_mixing(
    normalized_score(NNlib.softmax, masked_score, GenericAttenMaskOp(), mask, scaled_dot_product_score, q, k),
    v))

@inline function multihead_qkv_attention(head, q, k, v, mask=nothing)
    hq = CollapsedDimArray((move_head_dim_out ∘ split_head)(head, q), static(2), static(ndims(q)))
    hk = CollapsedDimArray((move_head_dim_out ∘ split_head)(head, k), static(2), static(ndims(k)))
    hv = CollapsedDimArray((move_head_dim_out ∘ split_head)(head, v), static(2), static(ndims(v)))
    a = naive_qkv_attention(hq, hk, hv, mask)
    return (merge_head ∘ move_head_dim_in)(a)
end


generic_attention(mixingf, scoref, q::AbstractArray, k::AbstractArray, v::AbstractArray, args...) = generic_qkv_attention(mixingf, scoref, q, k, v)
generic_attention(mixingf, scoref, head::Integer, q::AbstractArray, k::AbstractArray, v::AbstractArray, args...) = generic_multihead_qkv_attention(mixingf, scoref, head, q, k, v)
attention(head::Integer, q::AbstractArray, k::AbstractArray, v::AbstractArray, mask=nothing) = multihead_qkv_attention(head, q, k, v, mask)
attention(q::AbstractArray, k::AbstractArray, v::AbstractArray, mask=nothing) = naive_qkv_attention(q, k, v, mask)
