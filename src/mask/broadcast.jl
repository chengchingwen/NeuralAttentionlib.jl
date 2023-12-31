import Base.Broadcast: BroadcastStyle

Broadcast.broadcastable(m::AbstractMask) = m
Broadcast.broadcastable(m::Indexer) = m

Base.has_fast_linear_indexing(::Indexer) = false

BroadcastStyle(::Type{<:AbstractMask}) = Broadcast.DefaultArrayStyle{0}()
BroadcastStyle(::Type{<:Indexer{T, N}}) where {T, N} = Broadcast.DefaultArrayStyle{N}()

Base.size(m::AbstractMask) = ()

const AxesConstraints = Tuple{Vararg{AxesConstraint}}

function Base.Broadcast.preprocess(dest, m::AbstractMask)
    check_constraint(AxesConstraint(m), axes(dest))
    return GetIndexer(m, size(dest))
end
