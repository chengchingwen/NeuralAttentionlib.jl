import Base.Broadcast: BroadcastStyle, Broadcasted, combine_axes, broadcast_shape, instantiate

BroadcastStyle(::Type{<:AbstractAttenMask}) =  MaskStyle()

struct MaskStyle{A} <: Broadcast.BroadcastStyle end

MaskStyle() = MaskStyle(Broadcast.DefaultArrayStyle{0}())
MaskStyle(a) = MaskStyle{typeof(a)}()

BroadcastStyle(a::MaskStyle{M}, b::Broadcast.BroadcastStyle) where M = MaskStyle(Broadcast.result_style(M(), b))

Base.similar(bc::Broadcasted{MaskStyle{M}}, ::Type{Eltype}) where {M, Eltype} = similar(Broadcasted{M}(bc.f, bc.args, bc.axes), Eltype)

@inline function instantiate(bc::Broadcasted{MaskStyle{M}}) where M
    if bc.axes isa Nothing
        axes = combine_axes(bc.args...)
    else
        axes = bc.axes
    end
    return Broadcasted{M}(bc.f, bc.args, axes)
end

Base.axes(m::AbstractAttenMask) = AxesConstraint(m)

const AxesConstraints = Tuple{Vararg{AxesConstraint}}

broadcast_shape(c1::AxesConstraints, c2::AxesConstraints, shapes::Tuple...) = broadcast_shape(merge_constraint(c1, c2), shapes...)
broadcast_shape(c1::AxesConstraints, shape::Tuple, shapes::Tuple...) = broadcast_shape(check_constraint(c1, shape), shapes...)
broadcast_shape(shape::Tuple, c1::AxesConstraints, shapes::Tuple...) = broadcast_shape(check_constraint(c1, shape), shapes...)

function Base.mapreduce(f, op, A::Broadcasted{<:MaskStyle}, As::Base.AbstractArrayOrBroadcasted...;
                        dims = :, init = nothing)
    return mapreduce(f, op, instantiate(A), map(instantiate, As)...; dims, init)
end
