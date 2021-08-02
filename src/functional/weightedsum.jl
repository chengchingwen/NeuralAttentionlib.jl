@inline dot_product_weighted_sum(s, v) = matmul(v, s)

@inline weighted_sum(f, args...) = f(args...)

