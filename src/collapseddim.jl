@inline function noncollapsed_size(x, xi, xj, n)
    @assert n > 0
    if n == 1
        return ntuple(i->size(x, i), xi-1)
    elseif n == 2
        offseti = xi-1
        return ntuple(i->size(x, i+offseti), xj-xi)
    elseif n == 3
        offsetj = xj-1
        return ntuple(i->size(x, i+offsetj), max(ndims(x)-xj, 0)+1)
    else
        return (1,)
    end    
end

function collapsed_size(x, xi, xj, n)
    if n > 3
        return 1
    else
        ncs = noncollapsed_size(x, xi, xj, n)
        return stable_tuple_prod(ncs)
    end    
end

"""
    collapsed_size(x, xi, xj)

Collapse the dimensionality of `x` into 3 according to `xi` and `xj`.

    (X1, X2, ..., Xi-1, Xi, Xi+1, ..., Xj-1, Xj, ..., Xn)
     |_____dim1______|  |_______dim2______|  |___dim3__|

This is equivalent to `size(reshape(x, prod(size(x)[1:(xi-1)]), prod(size(x)[xi:(xj-1)]), prod(size(x)[xj:end])))`.

#Example

```julia
julia> x = randn(7,6,5,4,3,2);

julia> collapsed_size(x, 3,5)
(42, 20, 6)

```
"""
function collapsed_size(x, xi, xj)
    m = collapsed_size(x, xi, xj, 1)
    n = collapsed_size(x, xi, xj, 2)
    b = collapsed_size(x, xi, xj, 3)
    return (m,n,b)
end

struct CollapsedDimArray{T, A<:AbstractArray{T}, S1<:StaticInt, S2<:StaticInt, S3<:StaticBool} <: AbstractArray{T, 3}
    parent::A
    dims::Dims{3}
    si::S1
    sj::S2
    onebatch::S3

    function CollapsedDimArray(p::AbstractArray, dims::Dims{3}, si::StaticInt, sj::StaticInt, onebatch::StaticBool)
        @assert isone(dims[3]) == Bool(onebatch)
        return new{eltype(p), typeof(p), typeof(si), typeof(sj), typeof(onebatch)}(p, dims, si, sj, onebatch)
    end
end

Base.unsafe_convert(::Type{Ptr{T}}, ca::CollapsedDimArray{T}) where {T} = Base.unsafe_convert(Ptr{T}, parent(ca))
Base.parent(ca::CollapsedDimArray) = ca.parent
Base.similar(ca::CollapsedDimArray, eltype::Type, dims::Dims) = similar(parent(ca), eltype, dims)
Base.eltype(::CollapsedDimArray{T}) where T = T
Base.length(ca::CollapsedDimArray) = length(parent(ca))
Base.size(ca::CollapsedDimArray) = ca.dims

Base.strides(ca::CollapsedDimArray) = strides(Base.ReshapedArray(parent(ca), ca.dims, ()))

CollapsedDimArray(ca::CollapsedDimArray) = ca
CollapsedDimArray(parent) = CollapsedDimArray(parent, static(2), static(3))
CollapsedDimArray(parent::AbstractVecOrMat) = CollapsedDimArray(parent, static(2), static(3), static(true))
function CollapsedDimArray(p, si, sj)
    s1 = static(si)
    s2 = static(sj)
    dims = collapsed_size(p, s1, s2)
    onebatch = static(isone(dims[3]))
    return CollapsedDimArray(p, dims, s1, s2, onebatch)
end

function CollapsedDimArray(p::AbstractArray, si::StaticInt, sj::StaticInt, onebatch::StaticBool)
    dims = collapsed_size(p, si, sj)
    return CollapsedDimArray(p, dims, si, sj, onebatch)
end

Broadcast.broadcastable(ca::CollapsedDimArray) = collapseddim(ca)

function Base.getindex(ca::CollapsedDimArray, i...)
    return Base.getindex(Base.ReshapedArray(parent(ca), ca.dims, ()), i...)
end

function Base.setindex!(ca::CollapsedDimArray, args...)
    return Base.setindex!(Base.ReshapedArray(parent(ca), ca.dims, ()), args...)
end

const CollapsedAdjOrTrans{T} = NNlib.BatchedAdjOrTrans{T, <:CollapsedDimArray{T}}
const Collapsed{T} = Union{CollapsedAdjOrTrans{T}, <:CollapsedDimArray{T}}

function collapseddim(x::AbstractArray, xi, xj)
    return reshape(x, collapsed_size(x, xi, xj))
end

collapseddim(x::AbstractArray{T, 3}) where T = x
collapseddim(ca::CollapsedDimArray) = reshape(parent(ca), ca.dims)
collapseddim(ca::CollapsedAdjOrTrans) = ca isa NNlib.BatchedTranspose ? batched_transpose(collapseddim(parent(ca))) :
    batched_adjoint(collapseddim(parent(ca)))

unwrap_collapse(x) = x
unwrap_collapse(ca::CollapsedDimArray) = parent(ca)

@inline isonebatch(x::AbstractArray{T, 3}) where T = isone(size(x, 3))
@inline isonebatch(ca::CollapsedDimArray) = as_bool(ca.onebatch)
