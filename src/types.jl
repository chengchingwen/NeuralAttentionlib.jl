abstract type AbstractAttenOp end
abstract type AbstractAttenScoreOp end
abstract type AbstractMixingOp end

struct DotProductScoreOp <: AbstractAttenScoreOp end
(::DotProductScoreOp)(q, k) = dot_product_score(q, k)

struct ScaledDotProductScoreOp{F} <: AbstractAttenScoreOp
    scale::F
end
ScaledDotProductScoreOp() = ScaledDotProductScoreOp(nothing)
(::ScaledDotProductScoreOp{Nothing})(q, k) = scaled_dot_product_score(q, k)
(op::ScaledDotProductScoreOp)(q, k) = scaled_dot_product_score(q, k, op.scale)

struct NormalizedScoreOp{S, N} <: AbstractAttenScoreOp
    score::S
    norm::N
end
(op::NormalizedScoreOp)(args...) = normalized_score(op.norm, op.score, args...)

struct MaskedScoreOp{M, S} <: AbstractAttenScoreOp
    maskop::M
    score::S
end
MaskedScoreOp(score) = MaskedScoreOp(NaiveMaskOp(), score)
(op::MaskedScoreOp)(mask, args...) = masked_score(op.maskop, mask, op.score, args...)

struct WeightedSumMixingOp <: AbstractMixingOp end
(op::WeightedSumMixingOp)(args...) = weighted_sum_mixing(args...)

struct NaiveQKVAttenOp{F} <: AbstractAttenOp
    p::F
end
NaiveQKVAttenOp() = NaiveQKVAttenOp(nothing)
(op::NaiveQKVAttenOp)(q, k, v, mask = nothing, p = op.p) = naive_qkv_attention(q, k, v, mask, p)

"""
    struct MultiheadQKVAttenOp{F} <: AbstractAttenOp
        head::Int                   # number of head
        p::F  # dropout probability
    end

Structure for holding parameter of `multihead_qkv_attention`.

    (op::MultiheadQKVAttenOp)(q, k, v, mask = nothing)

Perform multihead attention.

"""
struct MultiheadQKVAttenOp{F} <: AbstractAttenOp
    head::Int
    p::F
end
MultiheadQKVAttenOp(head) = MultiheadQKVAttenOp(head, nothing)
(op::MultiheadQKVAttenOp)(q, k, v, mask = nothing) = multihead_qkv_attention(op.head, q, k, v, isnothing(mask) ? nothing : BatchedMask(mask), op.p)

"""
    struct MultiheadQKVAttenOpWithScore{F} <: AbstractAttenOp
        head::Int
        p::F
    end

Same as [`MultiheadQKVAttenOp`](@ref) but also return the attention score
"""
struct MultiheadQKVAttenOpWithScore{F} <: AbstractAttenOp
    head::Int
    p::F
end
MultiheadQKVAttenOpWithScore(head) = MultiheadQKVAttenOpWithScore(head, nothing)
(op::MultiheadQKVAttenOpWithScore)(q, k, v, mask = nothing) = multihead_qkv_attention(score_returning, op.head, q, k, v, isnothing(mask) ? nothing : BatchedMask(mask), op.p)

"""
    struct CausalMultiheadQKVAttenOp{F} <: AbstractAttenOp
        head::Int                   # number of head
        p::F  # dropout probability
    end

Structure for holding parameter of `multihead_qkv_attention`.

    (op::CausalMultiheadQKVAttenOp)(q, k, v, mask = nothing)

Perform multihead attention where `mask` would be combined with a [`CausalMask`](@ref)

"""
struct CausalMultiheadQKVAttenOp{F} <: AbstractAttenOp
    head::Int
    p::F
end
CausalMultiheadQKVAttenOp(head) = CausalMultiheadQKVAttenOp(head, nothing)
(op::CausalMultiheadQKVAttenOp)(q, k, v, mask = nothing) = multihead_qkv_attention(op.head, q, k, v, BatchedMask(CausalMask() & mask), op.p)

"""
    struct CausalMultiheadQKVAttenOpWithScore{F} <: AbstractAttenOp
        head::Int
        p::F
    end

Same as [`CausalMultiheadQKVAttenOp`](@ref) but also return the attention score
"""
struct CausalMultiheadQKVAttenOpWithScore{F} <: AbstractAttenOp
    head::Int
    p::F
end
CausalMultiheadQKVAttenOpWithScore(head) = CausalMultiheadQKVAttenOpWithScore(head, nothing)
(op::CausalMultiheadQKVAttenOpWithScore)(q, k, v, mask = nothing) = multihead_qkv_attention(score_returning, op.head, q, k, v, BatchedMask(CausalMask() & mask), op.p)
