####################  Dataless Mask  ####################

struct CausalMask <: AbstractDatalessMask end

Base.@propagate_inbounds Base.getindex(::CausalMask, I::Tuple) = I[2] >= I[1]

struct LocalMask <: AbstractDatalessMask
    width::Int
end

Base.@propagate_inbounds Base.getindex(m::LocalMask, I::Tuple) = I[2] - m.width < I[1] < I[2] + m.width

struct RandomMask <: AbstractDatalessMask
    p::Float64
end

Base.@propagate_inbounds Base.getindex(m::RandomMask, _::Tuple) = rand() > m.p

struct BandPartMask <: AbstractDatalessMask
    l::Int
    u::Int
end

Base.@propagate_inbounds Base.getindex(m::BandPartMask, I::Tuple) = (m.l < 0 || I[1] <= I[2] + m.l) && (m.u < 0 || I[1] >= I[2] - m.u)

