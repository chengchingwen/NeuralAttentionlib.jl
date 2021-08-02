@inline dot_product_score(q, k) = scaled_dot_product_score(q, k, one(eltype(q)))

@inline function scaled_dot_product_score(q, k, s = sqrt(inv(size(k, 1))))
    return matmul(batched_transpose(k), q, s)
end

attention_score(f, args...) = f(args...)

masked_score(score) = Base.Fix1(masked_score, score)
@inline masked_score(score, mask, args...) = masked_score(score, MaskOp(mask), mask, args...)
@inline masked_score(score, ::Nothing, args...) = score(args...)
@inline function masked_score(score, maskop::AbstractAttenMaskOp, mask, args...)
    return apply_mask(maskop, mask, score(args...))
end

normalized_score(norm) = Base.Fix1(normalized_score, norm)
@inline function normalized_score(norm, score, args...)
    return norm(score(args...))
end
