module Masks

using ..NeuralAttentionlib: @imexport

@imexport import ..NeuralAttentionlib:
    CausalMask, LocalMask, RandomMask, BandPartMask

"""
    CausalMask() <: AbstractDatalessMask

Attention mask that block the future values.

Similar to apply `LinearAlgebra.triu!` on the score matrix
"""
CausalMask

"""
    LocalMask(width::Int) <: AbstractDatalessMask

Attention mask that only allow local (diagonal like) values to pass.

`width` should be ≥ 0 and `A .* LocalMask(1)` is similar to `Diagonal(A)`
"""
LocalMask

"""
    RandomMask(p::Float64) <: AbstractDatalessMask

Attention mask that block value randomly.

`p` specify the percentage of value to block. e.g. `A .* RandomMask(0)` is equvalent to `identity(A)` and
 `A .* RandomMask(0)` is equvalent to `zero(A)`.
"""
RandomMask

"""
    BandPartMask(l::Int, u::Int) <: AbstractDatalessMask

Attention mask that only allow [band_part](https://www.tensorflow.org/api_docs/python/tf/linalg/band_part) values to pass.
"""
BandPartMask


end
