@inline weighted_sum_mixing(s, v) = matmul(v, s)

@inline mixing(f, args...) = f(args...)
