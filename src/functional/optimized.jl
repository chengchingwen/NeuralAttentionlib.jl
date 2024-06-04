using Random

const _BASED_SCORE_FUNC = Union{typeof(scaled_dot_product_score), typeof(dot_product_score)}
const _MASKED_SCORE_FUNC = Union{typeof(masked_score), PrefixedFunction{typeof(masked_score)}}
const _NORMALIZED_SCORE_FUNC = Union{typeof(normalized_score), PrefixedFunction{typeof(normalized_score)}}

dropout_score(f::dropoutF, score::Union{_MASKED_SCORE_FUNC, _BASED_SCORE_FUNC, _NORMALIZED_SCORE_FUNC}, args...) = collapseddims(dropoutF!(f), score(args...))

function ChainRulesCore.rrule(
    config::RuleConfig, ::typeof(dropout_score), f::dropoutF,
    score::Union{_MASKED_SCORE_FUNC, _BASED_SCORE_FUNC}, args...
)
    score_tape = rrule(config, score, args...)
    isnothing(score_tape) && (score_tape = rrule_via_ad(config, score, args...))
    s, score_pullback = score_tape
    proj = ProjectTo(s)
    x = collapseddims(s)
    rng = isnothing(f.rng) ? Random.default_rng() : f.rng
    y, dropout_dx! = _dropout_rrule!(rng, x, x, f.p, f.dims)
    output_size = size(y)
    function dropout_score_pullback(Ybar)
        Ȳ = reshape(unthunk(Ybar), output_size)
        ∂x = dropout_dx!(similar(x), Ȳ)
        ∂s = proj(∂x)
        ∂score, ∂args... = score_pullback(∂s)
        return (NoTangent(), NoTangent(), ∂score, ∂args...)
    end
    return proj(y), dropout_score_pullback
end

normalized_score(norm::typeof(softmax), score::Union{_MASKED_SCORE_FUNC, _BASED_SCORE_FUNC}, args...) =
    collapseddims(softmax!, score(args...))

function ChainRulesCore.rrule(
    config::RuleConfig, ::typeof(normalized_score), ::typeof(softmax),
    score::Union{_MASKED_SCORE_FUNC, _BASED_SCORE_FUNC}, args...
)
    score_tape = rrule(config, score, args...)
    isnothing(score_tape) && (score_tape = rrule_via_ad(config, score, args...))
    s, score_pullback = score_tape
    proj = ProjectTo(s)
    x = collapseddims(s)
    y = softmax!(x)
    output_size = size(y)
    function softmax_normalized_score_pullback(Ybar)
        Ȳ = reshape(unthunk(Ybar), output_size)
        ∂x = NNlib.∇softmax_data(Ȳ, y)
        ∂s = proj(∂x)
        ∂score, ∂args... = score_pullback(∂s)
        return (NoTangent(), NoTangent(), ∂score, ∂args...)
    end
    return proj(y), softmax_normalized_score_pullback
end

masked_score(maskop::AbstractMaskOp, mask::AbstractMask, score::_BASED_SCORE_FUNC, args...) = collapseddims_nonbatch(apply_mask! $ maskop $ AttenMask(mask), score(args...))

function ChainRulesCore.rrule(
    config::RuleConfig, ::typeof(masked_score),
    maskop::GenericMaskOp{<:Base.BroadcastFunction{typeof(+)}}, mask::AbstractMask, score::_BASED_SCORE_FUNC, args...
)
    score_tape = rrule(config, score, args...)
    isnothing(score_tape) && (score_tape = rrule_via_ad(config, score, args...))
    score_val, score_pullback = score_tape
    proj = ProjectTo(score_val)
    x = collapseddims_nonbatch(score_val)
    output_size = size(x)
    apply_mask!(maskop, mask, x)
    broadcast_add_masked_score_pullback(Ȳ) = (NoTangent(), NoTangent(), NoTangent(), score_pullback(Ȳ)...)
    return proj(x), broadcast_add_masked_score_pullback
end
