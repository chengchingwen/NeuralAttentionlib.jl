import Base.Broadcast: BroadcastStyle

Broadcast.broadcastable(m::AbstractMask) = m
Broadcast.broadcastable(m::Indexer) = m

Base.has_fast_linear_indexing(::Indexer) = false

BroadcastStyle(::Type{<:AbstractMask}) = Broadcast.DefaultArrayStyle{0}()

Base.size(m::AbstractMask) = ()

const AxesConstraints = Tuple{Vararg{AxesConstraint}}

Base.Broadcast.preprocess(dest, m::AbstractMask) = GetIndexer(m, size(dest))
