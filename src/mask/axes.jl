struct MaskAxes{N, T<:Tuple}
    axes::T
end

MaskAxes{N}() where N = MaskAxes{N}(())
MaskAxes{N}(t) where N = MaskAxes{N, typeof(t)}(t)

Base.ndims(::MaskAxes{N}) where N = N

# function combine_maskaxes(a::MaskAxes, b::MaskAxes)
#     a.axes == b.axes == () && return MaskAxes{max(ndims(a), ndims(b))}()    
# end

import Base.Broadcast: broadcast_shape

_bc_maskshape(shape::Tuple, a::MaskAxes) = _bc_maskshape(a, shape)
_bc_maskshape(a::MaskAxes{N, Tuple{}}, shape::Tuple) where N = length(shape) ≥ ndims(a) ? shape :
    throw(DimensionMismatch("arrays could not be broadcast to a common size"))

function _step_maskshape(n::Int, a::MaskAxes, shape::Tuple)
    if n > 0
        return (shape[1], _step_maskshape(n-1, a, Base.tail(shape))...)
    else
        return _bc_shape_eq(shape, a.axes)
    end
end

_length(x::Integer) = x
_length(x) = length(x)

_bc_shape_eq(a::Tuple{}, b::Tuple{}) = ()
function _bc_shape_eq(a::Tuple, b::Tuple)
    la = _length(a[1])
    lb = _length(b[1])
    la == lb || throw(DimensionMismatch("arrays could not be broadcast to a common size; got a dimension with lengths $la and $lb"))
    return (a[1], _bc_shape_eq(Base.tail(a), Base.tail(b))...)
end

function _bc_maskshape(a::MaskAxes, shape::Tuple)
    n = ndims(a)
    checkdim = n - length(a.axes)
    sn = length(shape)
    sn == n || # required num of dim match
        (sn == checkdim && all(isone∘_length, a.axes)) || # only 1 mask and 1 input
        throw(DimensionMismatch("arrays could not be broadcast to a common size; mask require $n dims input"))
    return _step_maskshape(checkdim, a, shape)
end

# can be use for batched
# function _bc_maskshape(a::MaskAxes, shape::Tuple)
#     n = ndims(a)
#     sn = length(shape)
#     sn ≥ n || throw(DimensionMismatch("arrays could not be broadcast to a common size"))
#     return _step_maskshape(sn - length(a.axes), a, shape)
# end

broadcast_shape(a::MaskAxes) = a
broadcast_shape(shape::Tuple, a::MaskAxes, shapes::Union{Tuple, MaskAxes}...) =
    broadcast_shape(_bc_maskshape(a, shape), shapes...)
broadcast_shape(a::MaskAxes, shape::Tuple, shapes::Union{Tuple, MaskAxes}...) =
    broadcast_shape(_bc_maskshape(a, shape), shapes...)

