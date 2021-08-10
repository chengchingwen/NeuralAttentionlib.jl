####################  Dataless Mask  ####################

struct CausalMask <: AbstractDatalessMask end

getmask_at(::CausalMask, I::Tuple) = I[2] >= I[1]

struct LocalMask <: AbstractDatalessMask
    width::Int
end

getmask_at(m::LocalMask, I::Tuple) = I[2] - m.width < I[1] < I[2] + m.width

struct RandomMask <: AbstractDatalessMask
    p::Float64
end

getmask_at(m::RandomMask, _::Tuple) = rand() > m.p

struct BandPartMask <: AbstractDatalessMask
    l::Int
    u::Int
end

getmask_at(m::BandPartMask, I::Tuple) = (m.l < 0 || I[1] <= I[2] + m.l) && (m.u < 0 || I[1] >= I[2] - m.u)

