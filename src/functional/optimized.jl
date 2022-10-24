const _BASED_SCORE_FUNC = Union{typeof(scaled_dot_product_score), typeof(dot_product_score)}

normalized_score(norm::typeof(softmax), score::_BASED_SCORE_FUNC, args...) = collapseddims(softmax!, score(args...))
function ChainRulesCore.rrule(
    config::RuleConfig, ::typeof(normalized_score), ::typeof(softmax),
    score::_BASED_SCORE_FUNC, args...
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
