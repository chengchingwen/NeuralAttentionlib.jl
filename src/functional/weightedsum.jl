dot_product_weighted_sum(s::AbstractMatrix, v::AbstractMatrix) = v * s
dot_product_weighted_sum(s::AbstractArray{T, 3}, v::AbstractArray{T, 3}) where T = NNlib.batched_mul(v, s)

weighted_sum(f, args...) = f(args...)

