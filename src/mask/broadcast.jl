import Base.Broadcast: BroadcastStyle

Broadcast.broadcastable(m::AbstractMask) = m
Broadcast.broadcastable(m::Indexer) = m

Base.has_fast_linear_indexing(::AbstractMask) = false
Base.has_fast_linear_indexing(::Indexer) = false

BroadcastStyle(::Type{<:AbstractMask}) = Broadcast.DefaultArrayStyle{0}()
BroadcastStyle(::Type{<:Indexer}) = Broadcast.DefaultArrayStyle{0}()

Base.size(m::AbstractMask) = ()
Base.size(m::Indexer) = ()

const AxesConstraints = Tuple{Vararg{AxesConstraint}}

function Base.Broadcast.preprocess(dest, m::AbstractMask)
    check_constraint(AxesConstraint(m), axes(dest))
    return GetIndexer(m, as_bool(require_dest(m)) ? size(dest) : nothing)
end
