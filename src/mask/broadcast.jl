import Base.Broadcast: BroadcastStyle

Base.has_fast_linear_indexing(::AbstractMask) = false

BroadcastStyle(::Type{<:AbstractMask}) = Broadcast.DefaultArrayStyle{0}()

Base.size(m::AbstractMask) = ()

const AxesConstraints = Tuple{Vararg{AxesConstraint}}

function Base.Broadcast.preprocess(dest, m::AbstractMask)
    check_constraint(AxesConstraint(m), axes(dest))
    return GetIndexer(m, as_bool(require_dest(m)) ? size(dest) : nothing)
end
