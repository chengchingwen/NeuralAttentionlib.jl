@inline weighted_sum_mixing(s, v) = matmul(v, s)

@inline mixing(f, v, g, args...) = unwrap_collapse(f(attention_score(g, args...), v))

struct ScoreReturning{F}; f::F; end

score_returning(f) = ScoreReturning(f)

@inline function mixing(sr::ScoreReturning, v, g, args...)
    s = attention_score(g, args...)
    y = unwrap_collapse(sr.f(s, v))
    return (hidden_state = y, attention_score = unwrap_collapse(s))
end
