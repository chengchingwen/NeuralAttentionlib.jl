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


end
