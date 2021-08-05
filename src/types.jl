abstract type AbstractAttenOp end
abstract type AbstractAttenScoreOp end
abstract type AbstractMixingOp end

struct DotProductScore <: AbstractAttenScoreOp end
(::DotProductScore)(args...) = dot_product_score(args...)

struct ScaledDotProductScore <: AbstractAttenScoreOp end
(::ScaledDotProductScore)(args...) = scaled_dot_product_score(args...)

struct NormalizedScoreOp{S, N} <: AbstractAttenScoreOp
    score::S
    norm::N
end
(op::NormalizedScoreOp)(args...) = normalized_score(op.norm, op.score, args...)

struct MaskedScoreOp{M, S} <: AbstractAttenScoreOp
    maskop::M
    score::S
end
MaskedScoreOp(score) = MaskedScoreOp(nothing, score)
(op::MaskedScoreOp)(mask, args...) = masked_score(op.score, op.maskop, mask, args...)

struct WeightedSumMixing <: AbstractMixingOp end
(op::WeightedSumMixing)(args...) = weighted_sum_mixing(args...)

struct NaiveQKVAttenOp <: AbstractAttenOp end
(::NaiveQKVAttenOp)(args...) = naive_qkv_attention(args...)

struct MultiheadQKVAttenOp <: AbstractAttenOp
    head::Int
end
(op::MultiheadQKVAttenOp)(args...) = multihead_qkv_attention(op.head, args...)
