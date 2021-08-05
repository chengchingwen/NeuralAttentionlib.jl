@inline dot_product_score(q, k) = scaled_dot_product_score(q, k, true)

@inline function scaled_dot_product_score(q, k, s = sqrt(inv(size(k, 1))))
    return matmul(batched_transpose(k), q, s)
end

masked_score(mask::Union{AbstractAttenMaskOp, AbstractAttenMask, Nothing}) = masked_score $ mask
@inline masked_score(::Nothing, score, args...) = score(args...)
@inline masked_score(maskop::AbstractAttenMaskOp, score, args...) = apply_mask(maskop, score(args...))
@inline masked_score(mask::AbstractAttenMask, score, args...) = masked_score(MaskOp(mask), mask, score, args...)
@inline function masked_score(maskop::AbstractAttenMaskOp, mask::AbstractAttenMask, score, args...)
    return apply_mask(maskop, mask, score(args...))
end

normalized_score(norm) = normalized_score $ norm
@inline function normalized_score(norm, score, args...)
    return norm(score(args...))
end

@inline attention_score(f, args...) = f(args...)
