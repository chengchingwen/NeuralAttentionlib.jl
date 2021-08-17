import Base.Broadcast: BroadcastStyle, Broadcasted, combine_axes, check_broadcast_axes

BroadcastStyle(::Type{<:AbstractAttenMask}) =  MaskStyle()

struct MaskStyle{A} <: Broadcast.BroadcastStyle end

MaskStyle() = MaskStyle(Broadcast.DefaultArrayStyle{0}())
MaskStyle(a) = MaskStyle{typeof(a)}()

BroadcastStyle(a::MaskStyle{M}, b::Broadcast.BroadcastStyle) where M = MaskStyle(Broadcast.result_style(M(), b))

Base.similar(bc::Broadcasted{MaskStyle{M}}, ::Type{Eltype}) where {M, Eltype} = similar(Broadcasted{M}(bc.f, bc.args, bc.axes), Eltype)

check_constrain(m::AbstractAttenMask, x) = check_constrain(AxesConstrain(m), x)
check_constrain(y, x) = Broadcast.broadcast_shape(axes(y), x) # fallback

@inline combine_axes(A, B::AbstractAttenMask) = combine_axes(B, A)
@inline combine_axes(A::AbstractAttenMask, B) = check_constrain(A, axes(B))
@inline combine_axes(A, B::Broadcasted{<:MaskStyle}) = combine_axes(B, A)
@inline combine_axes(A::Broadcasted{<:MaskStyle}, B) = check_constrain(A.args, axes(B))

@inline check_broadcast_axes(shp, A::AbstractAttenMask) = check_constrain(A, shp)
@inline check_broadcast_axes(shp, A::Broadcasted{<:MaskStyle}) = isnothing(A.axes) ? combine_axes(A.args, shp) : check_broadcast_axes(shp, axes(A))

AxesConstrain(::AbstractDatalessMask) = (NDimConstrain(2, true),)
