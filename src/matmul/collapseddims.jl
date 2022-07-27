struct CollapsedDimsArray{T, A<:AbstractArray{T}, S1<:StaticInt, S2<:StaticInt} <: AbstractArray{T, 3}
    parent::A
    dims::Dims{3}
    ni::S1
    nj::S2
end
function CollapsedDimsArray(parent::AbstractArray, ni::StaticInt, nj::StaticInt)
    @assert ni + nj <= ndims(parent) "$ni + $nj <= $(ndims(parent))"
    return CollapsedDimsArray(parent, collapsed_size(parent, ni, nj), ni, nj)
end
CollapsedDimsArray(parent::AbstractArray, ni::Integer, nj::Integer) = CollapsedDimsArray(parent, static(ni), static(nj))

CollapsedDimsArray(ca::CollapsedDimsArray) = ca
CollapsedDimsArray(parent::AbstractVector) = CollapsedDimsArray(parent, static(0), static(0))
CollapsedDimsArray(parent) = CollapsedDimsArray(parent, static(1), static(ndims(parent)) - static(2))

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

Base.unsafe_convert(::Type{Ptr{T}}, ca::CollapsedDimsArray{T}) where {T} = Base.unsafe_convert(Ptr{T}, parent(ca))
Base.pointer(ca::CollapsedDimsArray) = pointer(parent(ca))
Base.parent(ca::CollapsedDimsArray) = ca.parent

Base.similar(ca::CollapsedDimsArray, eltype::Type, dims::Dims) = similar(parent(ca), eltype, dims)
Base.similar(ca::CollapsedDimsArray, eltype::Type) = CollapsedDimsArray(similar(parent(ca), eltype), ca.dims, ca.ni, ca.nj)

Base.eltype(::CollapsedDimsArray{T}) where T = T
Base.length(ca::CollapsedDimsArray) = length(parent(ca))
Base.size(ca::CollapsedDimsArray) = ca.dims

Base.strides(ca::CollapsedDimsArray) = strides(Base.ReshapedArray(parent(ca), ca.dims, ()))

Base.reshape(ca::CollapsedDimsArray, dims::Dims) = reshape(parent(ca), dims)
Base.reshape(ca::CollapsedDimsArray, dims::Tuple{Vararg{Union{Colon, Int}}}) = reshape(parent(ca), dims)

Base.isapprox(a::CollapsedDimsArray, b::CollapsedDimsArray; kwargs...) = isapprox(collapseddims(a), collapseddims(b); kwargs...)
Base.isapprox(a::CollapsedDimsArray, b::AbstractArray; kwargs...) = isapprox(collapseddims(a), b; kwargs...)
Base.isapprox(a::AbstractArray, b::CollapsedDimsArray; kwargs...) = isapprox(a, collapseddims(b); kwargs...)

Base.collect(ca::CollapsedDimsArray) = reshape(collect(parent(ca)), ca.dims)

Base.getindex(ca::CollapsedDimsArray, i::Integer) = parent(ca)[i]
Base.getindex(ca::CollapsedDimsArray, i::Integer...) = getindex(ca, Base._sub2ind(axes(ca), i...))

Base.setindex!(ca::CollapsedDimsArray, v, i::Integer) = setindex!(parent(ca), v, i)
Base.setindex!(ca::CollapsedDimsArray, v, i::Integer...) = setindex!(ca, v, Base._sub2ind(axes(ca), i...))

Base.view(ca::CollapsedDimsArray, I::Vararg{Any, N}) where N = view(collapseddims(ca), I...)

const CollapsedAdjOrTrans{T} = NNlib.BatchedAdjOrTrans{T, <:CollapsedDimsArray{T}}
const Collapsed{T} = Union{CollapsedAdjOrTrans{T}, <:CollapsedDimsArray{T}}

_unwrap(x::NNlib.BatchedTranspose) = parent(x), batched_transpose
_unwrap(x::NNlib.BatchedAdjoint) = parent(x), batched_adjoint

noncollapsed_size(ca::CollapsedDimsArray, n) = noncollapsed_size(parent(ca), ca.ni, ca.nj, n)
noncollapsed_size(ca::CollapsedDimsArray) = noncollapsed_size(parent(ca), ca.ni, ca.nj)
collapsed_size(ca::CollapsedDimsArray, n) = size(ca, n)
collapsed_size(ca::CollapsedDimsArray) = size(ca)

collapseddims(x::AbstractArray, ni, nj) = reshape(x, collapsed_size(x, ni, nj))

collapseddims(ca::CollapsedDimsArray) = reshape(parent(ca), ca.dims)
function collapseddims(ca::CollapsedAdjOrTrans)
    x, rewrap = _unwrap(ca)
    return rewrap(collapseddims(x))
end

collapseddims_nonbatch(ca::CollapsedDimsArray) = reshape(parent(ca), (size(ca, 1), size(ca, 2), noncollapsed_size(ca, 3)...))

function collapseddims(f, ca::CollapsedDimsArray, args...; kwargs...)
    parent_size = size(parent(ca))
    y = f(collapseddims(ca), args...; kwargs...)
    _y = reshape(y, (:, Base.tail(parent_size)...))
    if size(_y, 1) == parent_size[1]
        return CollapsedDimsArray(_y, ca.dims, ca.ni, ca.nj)
    else
        return CollapsedDimsArray(_y, ca.ni, ca.nj)
    end
end

function collapseddims_nonbatch(f, ca::CollapsedDimsArray, args...; kwargs...)
    parent_size = size(parent(ca))
    y = f(collapseddims_nonbatch(ca), args...; kwargs...)
    _y = reshape(y, (:, Base.tail(parent_size)...))
    if size(_y, 1) == parent_size[1]
        return CollapsedDimsArray(_y, ca.dims, ca.ni, ca.nj)
    else
        return CollapsedDimsArray(_y, ca.ni, ca.nj)
    end
end

unwrap_collapse(x::AbstractArray) = x
unwrap_collapse(ca::CollapsedDimsArray) = parent(ca)

# GPU
adapt_structure(to, x::CollapsedDimsArray) = CollapsedDimsArray(Adapt.adapt_structure(to, parent(x)), x.dims, x.ni, x.nj)

Base.print_array(io::IO, ca::CollapsedDimsArray) = Base.print_array(io, collapseddims(ca))

Broadcast.BroadcastStyle(::Type{<:CollapsedDimsArray{T, S}}) where {T, S} = Broadcast.BroadcastStyle(S)
