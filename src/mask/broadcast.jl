import Base.Broadcast: BroadcastStyle, Broadcasted, combine_axes, check_broadcast_axes

BroadcastStyle(::Type{<:AbstractAttenMask}) =  MaskStyle()

struct MaskStyle{A} <: Broadcast.BroadcastStyle end

MaskStyle() = MaskStyle(Broadcast.DefaultArrayStyle{0}())
MaskStyle(a) = MaskStyle{typeof(a)}()

BroadcastStyle(a::MaskStyle{M}, b::Broadcast.BroadcastStyle) where M = MaskStyle(Broadcast.result_style(M(), b))

Base.similar(bc::Broadcasted{MaskStyle{M}}, ::Type{Eltype}) where {M, Eltype} = similar(Broadcasted{M}(bc.f, bc.args, bc.axes), Eltype)


"""
    AxesConstrain

Abstract type for define mask's constrain. return a tuple of concrete subtype of AxesConstrain.
"""
abstract type AxesConstrain end

struct All1Constrain <: AxesConstrain
    from::Int
    n::Int
end

struct DimConstrain <: AxesConstrain
    dim::Int
    val::Int
end

struct NDimConstrain <: AxesConstrain
    n::Int
    least::Bool
end
NDimConstrain(n) = NDimConstrain(n, false)

@noinline thrdm(s) = throw(DimensionMismatch("arrays could not be broadcast to a common size; $s"))

function check_constrain(c::All1Constrain, x)
    foreach(c.from:length(x)) do i
        if axislength(x[i]) != 1
            thrdm("mask require $i-th dimension to be 1, but get $(axislength(x[i]))")
        end
    end
    return x
end

function check_constrain(c::DimConstrain, x)
    if c.dim > 0
        dim = c.dim
        xi = length(x) ≥ dim ? axislength(x[dim]) : 1
        xi != c.val && thrdm("mask require $dim-th dimension to be $(c.val), but get $xi")
    else
        dim = length(x) + c.dim + 1
        xi = axislength(x[dim])
        xi != c.val && thrdm("mask require $dim-th dimension to be $(c.val), but get $xi")
    end
    return x
end

function check_constrain(c::NDimConstrain, x)
    cmp = c.least ? (≥) : (==)
    cmp(length(x), c.n) || thrdm("mask require ndims(A) $(cmp) $(c.n)")
    return x
end

check_constrain(cs::Tuple{}, x) = x
check_constrain(cs::Tuple, x) = check_constrain(Base.tail(cs), check_constrain(cs[1], x))

check_constrain(m::AbstractAttenMask, x) = check_constrain(AxesConstrain(m), x)
check_constrain(y, x) = Broadcast.broadcast_shape(axes(y), x) # fallback

@inline combine_axes(A, B::AbstractAttenMask) = combine_axes(B, A)
@inline combine_axes(A::AbstractAttenMask, B) = check_constrain(A, axes(B))
@inline combine_axes(A, B::Broadcasted{<:MaskStyle}) = combine_axes(B, A)
@inline combine_axes(A::Broadcasted{<:MaskStyle}, B) = check_constrain(A.args, axes(B))

@inline check_broadcast_axes(shp, A::AbstractAttenMask) = check_constrain(A, shp)
@inline check_broadcast_axes(shp, A::Broadcasted{<:MaskStyle}) = isnothing(A.axes) ? combine_axes(A.args, shp) : check_broadcast_axes(shp, axes(A))


axislength(x::Integer) = x
axislength(x) = length(x)

AxesConstrain(::AbstractDatalessMask) = (NDimConstrain(2, true),)

Base.iterate(c::AxesConstrain) = (c, nothing)
Base.iterate(c::AxesConstrain, ::Nothing) = nothing