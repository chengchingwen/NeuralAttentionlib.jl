####################  Dataless Mask  ####################

struct CausalMask <: AbstractDatalessMask end

Base.@propagate_inbounds Base.getindex(::CausalMask, i, j, _...) = j >= i

struct LocalMask <: AbstractDatalessMask
    width::Int
end

Base.@propagate_inbounds Base.getindex(m::LocalMask, i, j, _...) = j - m.width < i < j + m.width

struct RandomMask <: AbstractDatalessMask
    p::Float64
end

Base.@propagate_inbounds Base.getindex(m::RandomMask, _::Integer...) = rand() > m.p

AxesConstraint(m::RandomMask) = ()

struct BandPartMask <: AbstractDatalessMask
    l::Int
    u::Int
end

Base.@propagate_inbounds Base.getindex(m::BandPartMask, i, j, _...) = (m.l < 0 || i <= j + m.l) && (m.u < 0 || i >= j - m.u)
