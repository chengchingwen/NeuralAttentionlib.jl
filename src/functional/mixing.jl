@inline weighted_sum_mixing(s, v) = matmul(v, s)

@inline mixing(f, v, g, args...) = f(attention_score(g, args...), v)
