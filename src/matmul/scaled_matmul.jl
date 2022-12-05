scaled_matmul(a, b, s = true) = unwrap_collapse(matmul(a, b, s))

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(scaled_matmul), a, b)
    y, scaled_pullback = rrule(config, scaled_matmul, a, b, true)
    pullback(Ȳ) = Base.front(scaled_pullback(Ȳ))
    return y, pullback
end
function ChainRulesCore.rrule(config::RuleConfig, ::typeof(scaled_matmul), a, b, s)
    mm, matmul_pullback = rrule(config, matmul, a, b, s)
    y, unwrap_pullback = rrule(config, unwrap_collapse, mm)
    function pullback(Ȳ)
        _, ∂mm = unwrap_pullback(Ȳ)
        _, ∂a, ∂b, _ = matmul_pullback(∂mm)
        return (NoTangent(), ∂a, ∂b, NoTangent())
    end
    return y, pullback
end
