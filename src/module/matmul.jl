module Matmul

using ..NeuralAttentionlib: @imexport

@imexport import ..NeuralAttentionlib: matmul,
    collapsed_size, noncollapsed_size,
    CollapsedDimArray, collapseddim, unwrap_collapse

"""
    collapseddim(x::AbstractArray, xi, xj)

Reshape `x` into 3 dim array, equivalent to `reshape(x, collapsed_size(x, xi, xj))`

See also: [`collapsed_size`](@ref)
"""
collapseddim(x::AbstractArray, xi, xj)

"""
    collapseddim(ca::CollapsedDimArray)

remove the wrapper and really reshape it.

See also: [`CollapsedDimArray`](@ref), [`unwrap_collapse`](@ref)
"""
collapseddim(ca::CollapsedDimArray)

"""
    unwrap_collapse(ca::CollapsedDimArray)

Return the underlying array of `CollapsedDimArray`, otherwise just return the input.
"""
unwrap_collapse

"""
    matmul(a::AbstractArray, b::AbstractArray, s::Number = 1)

Equivalent to `s .* (a * b)` if `a` and `b` are `Vector` or `Matrix`. For array with higher dimension,
 it will convert `a` and `b` to [`CollapsedDimArray`](@ref) and perform batched matrix multiplication, and then
 return the result as `CollapsedDimArray`. This is useful for preserving the dimensionality. If the batch dimension
 of `a` and `b` have different shape, it pick the shape of `b` for batch dimension. Work with `NNlib.batch_transpose`
 and `NNlib.batch_adjoint`.

# Example

```julia
# b-dim shape: (6,)
julia> a = CollapsedDimArray(randn(3,4,2,3,6), 3, 5); size(a)
(12, 6, 6)

# b-dim shape: (3,1,2)
julia> b = CollapsedDimArray(randn(6,2,3,1,2), 2, 3); size(b)
(6, 2, 6)

julia> c = matmul(a, b); size(c), typeof(c)
((12, 2, 6), CollapsedDimArray{Float64, Array{Float64, 6}, Static.StaticInt{3}, Static.StaticInt{4}, Static.False})

# b-dim shape: (3,1,2)
julia> d = unwrap_collapse(c); size(d), typeof(d)
((3, 4, 2, 3, 1, 2), Array{Float64, 6})

# equivanlent to `batched_mul` but preserve shape
julia> NNlib.batched_mul(collapseddim(a), collapseddim(b)) == collapseddim(matmul(a, b))
true

```

See also: [`CollapsedDimArray`](@ref), [`unwrap_collapse`](@ref), [`collapseddim`](@ref)
"""
matmul


end
