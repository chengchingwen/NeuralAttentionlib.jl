dot_product_score(q::AbstractMatrix, k::AbstractMatrix) = k' * q
dot_product_score(q::AbstractArray{T, 3}, k::AbstractArray{T, 3}) where T = NNlib.batched_mul(NNlib.batched_transpose(k), q)

function scaled_dot_product_score(q, k, s=size(k, 1))
    score = dot_product_score(q, k)
    return score .* convert(eltype(score), inv(s))
end

attention_score(f, args...) = f(args...)

masked_score(score) = Base.Fix1(masked_score, score)
masked_score(score, mask, args...) = masked_score(score, MaskOp(mask), mask, args...)
function masked_score(score, maskop::AbstractAttenMaskOp, mask, args...)
    return apply_mask(maskop, mask, score(args...))
end

normalized_score(norm) = Base.Fix1(normalized_score, norm)
function normalized_score(norm, score, args...)
    return norm(score(args...))
end
