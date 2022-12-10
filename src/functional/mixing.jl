@inline weighted_sum_mixing(s, v) = scaled_matmul(v, s)

@inline mixing(f, v, g, args...) = f(attention_score(g, args...), v)

struct ScoreReturning{F} <: Function; f::F; end
(sr::ScoreReturning)(s, v) = (hidden_state = sr.f(s, v), attention_score = unwrap_collapse(s))

score_returning(f) = ScoreReturning(f)
