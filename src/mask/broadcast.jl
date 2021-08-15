import Base.Broadcast: BroadcastStyle, Broadcasted, combine_axes

BroadcastStyle(::Type{<:AbstractAttenMask}) =  MaskStyle()

struct MaskStyle{A} <: Broadcast.BroadcastStyle end

MaskStyle() = MaskStyle(Broadcast.DefaultArrayStyle{0}())
MaskStyle(a) = MaskStyle{typeof(a)}()

BroadcastStyle(a::MaskStyle{M}, b::Broadcast.BroadcastStyle) where M = MaskStyle(Broadcast.result_style(M(), b))

Base.similar(bc::Broadcasted{MaskStyle{M}}, ::Type{Eltype}) where {M, Eltype} = similar(Broadcasted{M}(bc.f, bc.args, bc.axes), Eltype)

abstract type AxesConstrain end

struct All1Constrain <: AxesConstrain
    from::Int
    n::Int
end

struct DimConstrain <: AxesConstrain
    dim::Int
    val::Int
end

struct ExactNDimConstrain <: AxesConstrain
    n::Int
end

struct LeastNDimConstrain <: AxesConstrain
    n::Int
end

thrdm(s) = throw(DimensionMismatch("arrays could not be broadcast to a common size; $s"))

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
        xi = length(x) ≥ dim ? axislength(x[c.dim]) : 1
        return xi == c.val ? x :
            thrdm("mask require $dim-th dimension to be $(c.val), but get $xi")
    else
        #TODO: write code
        return x
    end
end

check_constrain(c::LeastNDimConstrain, x) = length(x) ≥ c.n ? x : thrdm("mask require ndims(A) ≥ $(c.n)")

check_constrain(c::ExactNDimConstrain, x) = length(x) == c.n ? x : thrdm("mask require ndims(A) == $(c.n)")

check_constrain(cs::Tuple{}, x) = x
check_constrain(cs::Tuple, x) = check_constrain(Base.tail(cs), check_constrain(cs[1], x))

check_constrain(m::AbstractAttenMask, x) = check_constrain(AxesConstrain(m), x)

@inline combine_axes(A, B::AbstractAttenMask) = combine_axes(B, A)
@inline combine_axes(A::AbstractAttenMask, B) = check_constrain(A, axes(B))

axislength(x::Integer) = x
axislength(x) = length(x)

AxesConstrain(::AbstractDatalessMask) = (LeastNDimConstrain(2),)

Base.iterate(c::AxesConstrain) = (c, nothing)
Base.iterate(c::AxesConstrain, ::Nothing) = nothing
