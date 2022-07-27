@inline dot_product_score(q, k) = scaled_dot_product_score(q, k, true)

@inline function scaled_dot_product_score(q, k::AbstractVecOrMat, s = sqrt(inv(size(k, 1))))
    return matmul(k', q, s)
end
@inline function scaled_dot_product_score(q, k, s = sqrt(inv(size(k, 1))))
    return matmul(batched_adjoint(k), q, s)
end

masked_score(mask::Union{AbstractMaskOp, AbstractAttenMask, Nothing}) = masked_score $ mask
masked_score(maskop::AbstractMaskOp, mask::Union{AbstractAttenMask, Nothing}) = masked_score $ maskop $ mask
@inline masked_score(::Nothing, score, args...) = score(args...)
@inline masked_score(::AbstractMaskOp, ::Nothing, score, args...) = score(args...)
@inline masked_score(mask::AbstractAttenMask, score, args...) = masked_score(NaiveMaskOp(), mask, score, args...)
@inline masked_score(maskop::AbstractMaskOp, mask::AbstractAttenMask, score, args...) =
    collapseddims_nonbatch(apply_mask $ maskop $ mask, score(args...))

normalized_score(norm) = normalized_score $ norm
@inline normalized_score(norm, score, args...) = collapseddims(norm, score(args...))

dropout(x, p) = x .* getmask(RandomMask(p), x, inv(1 - p))
dropout_score(p) = dropout_score $ p
@inline dropout_score(p, score, args...) = collapseddims(Base.Fix2(dropout, p), score(args...))

@inline attention_score(f, args...) = f(args...)
