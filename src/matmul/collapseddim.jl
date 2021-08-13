"""
    noncollapsed_size(x, xi, xj, n)

Collapse the dimensionality of `x` into 3 according to `xi` and `xj`.

    (X1, X2, ..., Xi-1, Xi, Xi+1, ..., Xj-1, Xj, ..., Xn)
     |_____dim1______|  |_______dim2______|  |___dim3__|

But take the size before collapse. e.g. `noncollapsed_size(x, xi, xj, 2)` will be `(Xi, Xi+1, ..., Xj-1)`.

#Example

```julia
julia> x = randn(7,6,5,4,3,2);

julia> noncollapsed_size(x, 3, 5, 1)
(7, 6)

julia> noncollapsed_size(x, 3, 5, 2)
(5, 4)

julia> noncollapsed_size(x, 3, 5, 3)
(3, 2)

```

See also: [`collapsed_size`](@ref)
"""
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
    collapsed_size(x, xi, xj)::Dim{3}

Collapse the dimensionality of `x` into 3 according to `xi` and `xj`.

    (X1, X2, ..., Xi-1, Xi, Xi+1, ..., Xj-1, Xj, ..., Xn)
     |_____dim1______|  |_______dim2______|  |___dim3__|

This is equivalent to `size(reshape(x, prod(size(x)[1:(xi-1)]), prod(size(x)[xi:(xj-1)]), prod(size(x)[xj:end])))`.

#Example

```julia
julia> x = randn(7,6,5,4,3,2);

julia> collapsed_size(x, 3, 5)
(42, 20, 6)

```

See also: [`noncollapsed_size`](@ref)
"""
function collapsed_size(x, xi, xj)
    m = collapsed_size(x, xi, xj, 1)
    n = collapsed_size(x, xi, xj, 2)
    b = collapsed_size(x, xi, xj, 3)
    return (m,n,b)
end


"""
    CollapsedDimArray{T}(array, si::Integer=2, sj::Integer=3) <: AbstractArray{T, 3}

Similar to lazy reshape array with [`collapsed_size`](@ref)
"""
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
    return Base.getindex(collapseddim(ca), i...)
end

function Base.setindex!(ca::CollapsedDimArray, args...)
    return Base.setindex!(collapseddim(ca), args...)
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

noncollapsed_size(ca::CollapsedDimArray, n) = noncollapsed_size(parent(ca), ca.si, ca.sj, n)

collapseddim_nonbatch(x) = x
collapseddim_nonbatch(ca::CollapsedDimArray) = reshape(parent(ca), (size(ca, 1), size(ca, 2), noncollapsed_size(ca, 3)...))

@inline isonebatch(x::AbstractArray{T, 3}) where T = isone(size(x, 3))
@inline isonebatch(ca::CollapsedDimArray) = as_bool(ca.onebatch)

# preserve the dim of first arg
@inline function _collapsed_call(f, ca::CollapsedDimArray, args...; kwargs...)
    real_size = size(parent(ca))
    y = f(collapseddim(ca), args...; kwargs...)
    return CollapsedDimArray(reshape(y, real_size), ca.dims, ca.si, ca.sj, ca.onebatch)
end
@inline _collapsed_call(f, args...; kwargs...) = f(args...; kwargs...)

@inline function _collapsed_nonbatch_call(f, ca::CollapsedDimArray, args...; kwargs...)
    real_size = size(parent(ca))
    y = f(collapseddim_nonbatch(ca), args...; kwargs...)
    return CollapsedDimArray(reshape(y, real_size), ca.dims, ca.si, ca.sj, ca.onebatch)
end
@inline _collapsed_nonbatch_call(f, args...; kwargs...) = f(args...; kwargs...)

import Adapt: adapt_structure, adapt
adapt_structure(to, x::CollapsedDimArray) = CollapsedDimArray(adapt(to, parent(x)), x.dims, x.si, x.sj, x.onebatch)

# GPU kernel compat
#Adapt.adapt(to::CUDA.Adaptor, x::CollapsedDimArray) = Adapt.adapt(to, collapseddim(x))

@inline function Base.view(ca::CollapsedDimArray, I::Vararg{Any,N}) where {N}
    return view(collapseddim(ca), I...)
end

Base.print_array(io::IO, ca::CollapsedDimArray) = Base.print_array(io, collapseddim(ca))
