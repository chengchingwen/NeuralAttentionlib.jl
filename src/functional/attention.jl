function attention(weighted_sum, attention_score, )
    attention_score(mask, )
    return weighted_sum()
end

navie_qkv_attention(q, k, v, mask=nothing) = dot_product_weighted_sum(
    normalized_score(NNlib.softmax, masked_score, scaled_dot_product_score, mask, q, k),
    v)


function multihead_qkv_attention(head, q, k, v, mask=nothing)
    hq = (move_head_dim_out ∘ split_head)(head, q)
    hk = (move_head_dim_out ∘ split_head)(head, k)
    hv = (move_head_dim_out ∘ split_head)(head, v)
    a = navie_qkv_attention(hq, hk, hv, mask)
    return (merge_head ∘ move_head_dim_in)(a)
end

# function attention(q, k, v, mask)
#     s = score(q, k)
#     s = normalize(s)
#     s = apply_mask(s, mask)
#     return weightsum(s, v)
# end
