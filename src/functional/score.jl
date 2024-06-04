@inline dot_product_score(q, k) = scaled_dot_product_score(q, k, true)

@inline dot_product_score(f, q, k) = dot_product_score(f, f, q, k)
@inline dot_product_score(qf, kf, q, k) = dot_product_score(qf(q), kf(k))

@inline function scaled_dot_product_score(q, k, s::Number = sqrt(inv(size(k, 1))))
    return matmul(collapsed_adjoint(k), q, s)
end

@inline scaled_dot_product_score(s::Number, q, k) = scaled_dot_product_score(q, k, s)
@inline scaled_dot_product_score(f, q, k) = scaled_dot_product_score(f, f, q, k)
@inline scaled_dot_product_score(qf, kf, q, k) = scaled_dot_product_score(qf(q), kf(k))
@inline scaled_dot_product_score(s::Number, f, q, k) = scaled_dot_product_score(s, f, f, q, k)
@inline scaled_dot_product_score(s::Number, qf, kf, q, k) = scaled_dot_product_score(qf(q), kf(k),  s)

masked_score(mask::Union{AbstractMaskOp, AbstractMask, Nothing}) = masked_score $ mask
masked_score(maskop::AbstractMaskOp, mask::Union{AbstractMask, Nothing}) = masked_score $ maskop $ mask
@inline masked_score(::Nothing, score, args...) = score(args...)
@inline masked_score(::AbstractMaskOp, ::Nothing, score, args...) = score(args...)
@inline masked_score(mask::AbstractMask, score, args...) = masked_score(NaiveMaskOp(), mask, score, args...)
@inline masked_score(maskop::AbstractMaskOp, mask::AbstractMask, score, args...) =
    collapseddims_nonbatch(apply_mask $ maskop $ AttenMask(mask), score(args...))

normalized_score(norm) = normalized_score $ norm
@inline normalized_score(norm, score, args...) = collapseddims(norm, score(args...))

include("dropout.jl")
_dropout_func(p::Real) = dropoutF(; p)
_dropout_func(p::Function) = p
dropout_score(p) = dropout_score $ p
@inline dropout_score(p::Real, score, args...) = dropout_score(_dropout_func(p), score, args...)
@inline dropout_score(p, score, args...) = collapseddims(_dropout_func(p), score(args...))
@inline dropout_score(::Nothing, score, args...) = score(args...)

bias_add(b) = bias_add $ b
function bias_add(b, s)
    @assert size(s, 1) == size(b, 1) && size(s, 2) == size(b, 2) && ndims(s) >= ndims(b)
    return s .+ b
end
biased_score(b) = biased_score $ b
@inline biased_score(b, score, args...) = collapseddims_nonbatch(bias_add(b), score(args...))

@inline attention_score(f, args...) = f(args...)
@inline attention_score(pf::PrefixedFunction, args...) = attention_score(pf.f, pf.arg..., args...)
