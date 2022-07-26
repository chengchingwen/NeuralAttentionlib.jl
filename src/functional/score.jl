@inline dot_product_score(q, k) = scaled_dot_product_score(q, k, true)

@inline function scaled_dot_product_score(q, k::AbstractVecOrMat, s = sqrt(inv(size(k, 1))))
    return matmul(k', q, s)
end
@inline function scaled_dot_product_score(q, k, s = sqrt(inv(size(k, 1))))
    return matmul(batched_adjoint(k), q, s)
end

masked_score(mask::Union{AbstractAttenMaskOp, AbstractAttenMask, Nothing}) = masked_score $ mask
masked_score(maskop::AbstractAttenMaskOp, mask::Union{AbstractAttenMask, Nothing}) = masked_score $ maskop $ mask
@inline masked_score(::Nothing, score, args...) = score(args...)
@inline masked_score(::AbstractAttenMaskOp, ::Nothing, score, args...) = score(args...)
@inline masked_score(mask::AbstractAttenMask, score, args...) = masked_score(MaskOp(mask), mask, score, args...)
@inline function masked_score(maskop::AbstractAttenMaskOp, mask::AbstractAttenMask, score, args...)
    return collapseddims_nonbatch(score(args...)) do sc
        apply_mask(maskop, mask, sc)
    end
end

normalized_score(norm) = normalized_score $ norm
@inline function normalized_score(norm, score, args...)
    return collapseddims(norm, score(args...))
end

@inline attention_score(f, args...) = f(args...)
