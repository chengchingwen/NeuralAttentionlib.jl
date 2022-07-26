struct CollapsedDimArray{T, A<:AbstractArray{T}, S1<:StaticInt, S2<:StaticInt} <: AbstractArray{T, 3}
    parent::A
    dims::Dims{3}
    ni::S1
    nj::S2
end
function CollapsedDimArray(parent::AbstractArray, ni::StaticInt, nj::StaticInt)
    @assert ni + nj <= ndims(parent) "$ni + $nj <= $(ndims(parent))"
    return CollapsedDimArray(parent, collapsed_size(parent, ni, nj), ni, nj)
end
CollapsedDimArray(parent::AbstractArray, ni::Integer, nj::Integer) = CollapsedDimArray(parent, static(ni), static(nj))

CollapsedDimArray(ca::CollapsedDimArray) = ca
CollapsedDimArray(parent::AbstractVector) = CollapsedDimArray(parent, static(0), static(0))
CollapsedDimArray(parent) = CollapsedDimArray(parent, static(1), static(ndims(parent)) - static(2))

_getter(s, offset) = Base.Fix1(getindex, s)∘Base.Fix1(+, offset)

@inline function noncollapsed_size(s::Tuple, ni, nj, n)
    if n == 1
        N = static(length(s)) - static(ni) - static(nj)
        return N == 0 ? (1,) : ntuple(_getter(s, 0), N)
    elseif n == 2
        N = static(ni)
        offset = static(length(s)) - static(ni) - static(nj)
        return N == 0 ? (1,) : ntuple(_getter(s, offset), N)
    elseif n == 3
        N = static(nj)
        offset = static(length(s)) - static(nj)
        return N == 0 ? (1,) : ntuple(_getter(s, offset), N)
    else
        return (1,)
    end
end
noncollapsed_size(x, ni, nj, n) = noncollapsed_size(size(x), ni, nj, n)
noncollapsed_size(x, ni, nj) = (s = size(x); (
    noncollapsed_size(s, ni, nj, static(1)),
    noncollapsed_size(s, ni, nj, static(2)),
    noncollapsed_size(s, ni, nj, static(3))
))

collapsed_size(x, ni, nj, n) = n > 3 ? 1 : prod(noncollapsed_size(x, ni, nj, n))
collapsed_size(x, ni, nj) = prod.(noncollapsed_size(x, ni, nj))

Base.unsafe_convert(::Type{Ptr{T}}, ca::CollapsedDimArray{T}) where {T} = Base.unsafe_convert(Ptr{T}, parent(ca))
Base.pointer(ca::CollapsedDimArray) = pointer(parent(ca))
Base.parent(ca::CollapsedDimArray) = ca.parent

Base.similar(ca::CollapsedDimArray, eltype::Type, dims::Dims) = similar(parent(ca), eltype, dims)
Base.similar(ca::CollapsedDimArray, eltype::Type) = CollapsedDimArray(similar(parent(ca), eltype), ca.dims, ca.ni, ca.nj)

Base.eltype(::CollapsedDimArray{T}) where T = T
Base.length(ca::CollapsedDimArray) = length(parent(ca))
Base.size(ca::CollapsedDimArray) = ca.dims

Base.strides(ca::CollapsedDimArray) = strides(Base.ReshapedArray(parent(ca), ca.dims, ()))

Base.reshape(ca::CollapsedDimArray, dims::Dims) = reshape(parent(ca), dims)
Base.reshape(ca::CollapsedDimArray, dims::Tuple{Vararg{Union{Colon, Int}}}) = reshape(parent(ca), dims)

Base.isapprox(a::CollapsedDimArray, b::CollapsedDimArray; kwargs...) = isapprox(collapseddim(a), collapseddim(b); kwargs...)
Base.isapprox(a::CollapsedDimArray, b::AbstractArray; kwargs...) = isapprox(collapseddim(a), b; kwargs...)
Base.isapprox(a::AbstractArray, b::CollapsedDimArray; kwargs...) = isapprox(a, collapseddim(b); kwargs...)

Base.collect(ca::CollapsedDimArray) = reshape(collect(parent(ca)), ca.dims)

Base.getindex(ca::CollapsedDimArray, i::Integer) = parent(ca)[i]
Base.getindex(ca::CollapsedDimArray, i::Integer...) = getindex(ca, Base._sub2ind(axes(ca), i...))

Base.setindex!(ca::CollapsedDimArray, v, i::Integer) = setindex!(parent(ca), v, i)
Base.setindex!(ca::CollapsedDimArray, v, i::Integer...) = setindex!(ca, v, Base._sub2ind(axes(ca), i...))

Base.view(ca::CollapsedDimArray, I::Vararg{Any, N}) where N = view(collapseddim(ca), I...)

const CollapsedAdjOrTrans{T} = NNlib.BatchedAdjOrTrans{T, <:CollapsedDimArray{T}}
const Collapsed{T} = Union{CollapsedAdjOrTrans{T}, <:CollapsedDimArray{T}}

_unwrap(x::NNlib.BatchedTranspose) = parent(x), batched_transpose
_unwrap(x::NNlib.BatchedAdjoint) = parent(x), batched_adjoint

noncollapsed_size(ca::CollapsedDimArray, n) = noncollapsed_size(parent(ca), ca.ni, ca.nj, n)
noncollapsed_size(ca::CollapsedDimArray) = noncollapsed_size(parent(ca), ca.ni, ca.nj)
collapsed_size(ca::CollapsedDimArray, n) = size(ca, n)
collapsed_size(ca::CollapsedDimArray) = size(ca)

collapseddim(x::AbstractArray, ni, nj) = reshape(x, collapsed_size(x, ni, nj))

collapseddim(ca::CollapsedDimArray) = reshape(parent(ca), ca.dims)
function collapseddim(ca::CollapsedAdjOrTrans)
    x, rewrap = _unwrap(ca)
    return rewrap(collapseddim(x))
end

collapseddim_nonbatch(ca::CollapsedDimArray) = reshape(parent(ca), (size(ca, 1), size(ca, 2), noncollapsed_size(ca, 3)...))

function collapseddim(f, ca::CollapsedDimArray, args...; kwargs...)
    parent_size = size(parent(ca))
    y = f(collapseddim(ca), args...; kwargs...)
    _y = reshape(y, (:, Base.tail(parent_size)...))
    if size(_y, 1) == parent_size[1]
        return CollapsedDimArray(_y, ca.dims, ca.ni, ca.nj)
    else
        return CollapsedDimArray(_y, ca.ni, ca.nj)
    end
end

function collapseddim_nonbatch(f, ca::CollapsedDimArray, args...; kwargs...)
    parent_size = size(parent(ca))
    y = f(collapseddim_nonbatch(ca), args...; kwargs...)
    _y = reshape(y, (:, Base.tail(parent_size)...))
    if size(_y, 1) == parent_size[1]
        return CollapsedDimArray(_y, ca.dims, ca.ni, ca.nj)
    else
        return CollapsedDimArray(_y, ca.ni, ca.nj)
    end
end

unwrap_collapse(x::AbstractArray) = x
unwrap_collapse(ca::CollapsedDimArray) = parent(ca)

# GPU
adapt_structure(to, x::CollapsedDimArray) = CollapsedDimArray(Adapt.adapt_structure(to, parent(x)), x.dims, x.ni, x.nj)

Base.print_array(io::IO, ca::CollapsedDimArray) = Base.print_array(io, collapseddim(ca))

Broadcast.BroadcastStyle(::Type{<:CollapsedDimArray{T, S}}) where {T, S} = Broadcast.BroadcastStyle(S)
