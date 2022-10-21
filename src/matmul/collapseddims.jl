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
noncollapsed_size(s::Tuple, ni, nj) = (
    noncollapsed_size(s, ni, nj, static(1)),
    noncollapsed_size(s, ni, nj, static(2)),
    noncollapsed_size(s, ni, nj, static(3))
)

noncollapsed_size(x, ni, nj, n) = noncollapsed_size(size(x), ni, nj, n)
noncollapsed_size(x, ni, nj) = noncollapsed_size(size(x), ni, nj)

noncollapsed_size(ca::CollapsedDimsArray, n) = noncollapsed_size(size(parent(ca)), ca.ni, ca.nj, n)
noncollapsed_size(ca::CollapsedDimsArray) = noncollapsed_size(size(parent(ca)), ca.ni, ca.nj)

collapsed_size(x, ni, nj, n) = n > 3 ? 1 : prod(noncollapsed_size(x, ni, nj, n))
collapsed_size(x, ni, nj) = prod.(noncollapsed_size(x, ni, nj))

collapsed_size(ca::CollapsedDimsArray, n) = size(ca, n)
collapsed_size(ca::CollapsedDimsArray) = size(ca)

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

collapseddims(x::AbstractArray, ni, nj) = reshape(x, collapsed_size(x, ni, nj))

collapseddims(ca::CollapsedDimsArray) = reshape(parent(ca), ca.dims)
function collapseddims(ca::CollapsedAdjOrTrans)
    x, rewrap = _unwrap(ca)
    return rewrap(collapseddims(x))
end

collapseddims_nonbatch(ca::CollapsedDimsArray) = reshape(parent(ca), (size(ca, 1), size(ca, 2), noncollapsed_size(ca, 3)...))

collapseddims(f, ca::CollapsedDimsArray, args...; kwargs...) =
    _collapseddims(_collapseddims_f_config(collapseddims)..., f, ca, args...; kwargs...)
collapseddims_nonbatch(f, ca::CollapsedDimsArray, args...; kwargs...) =
    _collapseddims(_collapseddims_f_config(collapseddims_nonbatch)..., f, ca, args...; kwargs...)

collapseddims_fdim1(f, ca::CollapsedDimsArray, args...; kwargs...) =
    _collapseddims(_collapseddims_f_config(collapseddims_fdim1)..., f, ca, args...; kwargs...)
collapseddims_nonbatch_fdim1(f, ca::CollapsedDimsArray, args...; kwargs...) =
    _collapseddims(_collapseddims_f_config(collapseddims_nonbatch_fdim1)..., f, ca, args...; kwargs...)

_collapseddims_f_config(::typeof(collapseddims))                = static(false), static(false)
_collapseddims_f_config(::typeof(collapseddims_nonbatch))       = static(true) , static(false)
_collapseddims_f_config(::typeof(collapseddims_fdim1))          = static(false), static(true)
_collapseddims_f_config(::typeof(collapseddims_nonbatch_fdim1)) = static(true) , static(true)

function _collapseddims(nonbatch, fdim1, f, ca::CollapsedDimsArray, args...; kwargs...)
    parent_size = size(parent(ca))
    x = as_bool(nonbatch) ? collapseddims_nonbatch(ca) : collapseddims(ca)
    _y = f(x, args...; kwargs...)
    if !as_bool(fdim1)
        @assert size(_y, 1) == size(ca, 1) "f cannot change the size of feature dimension; use func with \"_fdim1\" suffix"
        y = reshape(_y, parent_size)
        return CollapsedDimsArray(y, size(ca), ca.ni, ca.nj)
    else
        ni, nj = ca.ni, ca.nj
        n = ca.ni + ca.nj
        offset = static(length(parent_size)) - ni - nj
        tail_size = n == 0 ? () : ntuple(_getter(parent_size, offset), n)
        y = reshape(_y, (:, tail_size...))
        return CollapsedDimsArray(y, ni, nj)
    end
end

unwrap_collapse(x::AbstractArray) = x
unwrap_collapse(ca::CollapsedDimsArray) = parent(ca)

collapsed_transpose(x::CollapsedDimsArray) = batched_transpose(x)
collapsed_transpose(x::AbstractVecOrMat) = transpose(x)
collapsed_transpose(x::AbstractArray) = collapsed_transpose(CollapsedDimsArray(x))

collapsed_adjoint(x::CollapsedDimsArray) = batched_adjoint(x)
collapsed_adjoint(x::AbstractVecOrMat) = adjoint(x)
collapsed_adjoint(x::AbstractArray) = collapsed_adjoint(CollapsedDimsArray(x))

# GPU
adapt_structure(to, x::CollapsedDimsArray) = CollapsedDimsArray(Adapt.adapt_structure(to, parent(x)), x.dims, x.ni, x.nj)

Base.print_array(io::IO, ca::CollapsedDimsArray) = Base.print_array(io, collapseddims(ca))

Broadcast.BroadcastStyle(::Type{<:CollapsedDimsArray{T, S}}) where {T, S} = Broadcast.BroadcastStyle(S)

GPUArraysCore.backend(T::Type{<:CollapsedDimsArray{E, <:CuArray}}) where E = GPUArraysCore.backend(CuArray{E, 3})
