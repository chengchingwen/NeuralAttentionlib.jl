####################  Dataless Mask  ####################

AxesConstraint(::AbstractAttenMask{DATALESS}) = (NDimConstraint(2, true),)

struct CausalMask <: AbstractAttenMask{DATALESS} end

Base.@propagate_inbounds maskgetindex(::Dims, ::CausalMask, i::Integer, j::Integer, _::Integer...) = j >= i

struct LocalMask <: AbstractAttenMask{DATALESS}
    width::Int
end

Base.@propagate_inbounds maskgetindex(::Dims, m::LocalMask, i::Integer, j::Integer, _::Integer...) = j - m.width < i < j + m.width

struct RandomMask{R} <: AbstractAttenMask{DATALESS}
    p::Float32
    rng::R
end
RandomMask(p) = RandomMask(convert(Float32, p), nothing)

include("prand.jl")
adapt_structure(to::Type{Indexer}, x::RandomMask) = RandomMask(x.p, adapt(to, @something(x.rng, CPLCGm32())))

Base.@propagate_inbounds maskgetindex(::Dims, m::RandomMask{Nothing}, _::Integer...) = rand(Float32) >= m.p
Base.@propagate_inbounds function maskgetindex(destsize::Dims, m::RandomMask, I::Integer...)
    s = +((UInt32.(destsize) .* UInt32.(reverse(I)))...)
    v, rng = prand(Float32, setpos(m.rng, s), s)
    return v >= m.p
end

AxesConstraint(m::RandomMask) = ()

struct BandPartMask <: AbstractAttenMask{DATALESS}
    l::Int
    u::Int
end

Base.@propagate_inbounds maskgetindex(::Dims, m::BandPartMask, i::Integer, j::Integer, _::Integer...) = (m.l < 0 || i <= j + m.l) && (m.u < 0 || i >= j - m.u)
