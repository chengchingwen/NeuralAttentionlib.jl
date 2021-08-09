include("./indexer.jl")

"""
Wrapper type for mask data, can be viewed as AbstractArray{Bool}
"""
abstract type AbstractAttenMask end

abstract type AbstractDatalessMask <: AbstractAttenMask end
abstract type AbstractArrayMask <: AbstractAttenMask end

Broadcast.broadcastable(m::AbstractAttenMask) = m
Broadcast.BroadcastStyle(::Type{<:AbstractAttenMask}) = Broadcast.DefaultArrayStyle{0}()
Base.size(::AbstractAttenMask) = ()
Base.eltype(::AbstractAttenMask) = Bool
Base.@propagate_inbounds Broadcast.newindex(arg::AbstractAttenMask, I::CartesianIndex) = I

const AbstractMaskIndexer = Indexer{<:AbstractArrayMask}
Broadcast.BroadcastStyle(::Type{<:AbstractMaskIndexer}) = Broadcast.DefaultArrayStyle{0}()
Base.@propagate_inbounds Broadcast.newindex(arg::AbstractMaskIndexer, I::CartesianIndex) = I
Base.size(::AbstractMaskIndexer) = ()
Base.eltype(::AbstractMaskIndexer) = Bool

Base.getindex(m::AbstractAttenMask, I...) = getmask_at(m, I...)

GetIndexer(m::AbstractDatalessMask) = m

getmask_at(m::AbstractAttenMask, i::CartesianIndex) = getmask_at(m, Tuple(i)...)
getmask_at(m::AbstractAttenMask, I::Integer...) = getmask_at(GetIndexer(m), I...)

Base.getindex(m::AbstractMaskIndexer, I...) = getmask_at(m, I...)
getmask_at(m::AbstractMaskIndexer, i::CartesianIndex) = getmask_at(m, Tuple(i)...)

####################  Dataless Mask  ####################

struct CausalMask <: AbstractDatalessMask end

getmask_at(::CausalMask, i, j, _...) = j >= i

struct LocalMask <: AbstractDatalessMask
    width::Int
end

getmask_at(m::LocalMask, i, j, _...) = j - m.width < i < j + m.width

struct RandomMask <: AbstractDatalessMask
    p::Float64
end

getmask_at(m::RandomMask, _::Integer...) = rand() > m.p

struct BandPartMask <: AbstractDatalessMask
    l::Int
    u::Int
end

####################  Array Mask  ####################

getmask_at(m::BandPartMask, i, j, _...) = (m.l < 0 || i <= j + m.l) && (m.u < 0 || i >= j - m.u)

struct SymLengthMask{N, L <: AbstractArray{Int32, N}} <: AbstractArrayMask
    len::L
end

SymLengthMask(len) = SymLengthMask(convert(AbstractArray{Int32}, len))

using Adapt
using CUDA
import Adapt: adapt_structure, adapt
adapt_structure(to, x::SymLengthMask) = SymLengthMask(adapt(to, x.len))

Adapt.adapt(to::CUDA.Adaptor, m::AbstractArrayMask) = Indexer{typeof(m)}(map(Base.Fix1(Adapt.adapt, to), GetIndexer(m).__fields))

Base.@propagate_inbounds function getmask_at(m::Indexer{<:SymLengthMask}, i, j, I...)
    l = m.len[I...]
    return i <= l && j <= l
end

struct BiLengthMask{N, L <: AbstractArray{Int32, N}} <: AbstractArrayMask
    q_len::L
    k_len::L
end

BiLengthMask(q_len, k_len) = BiLengthMask(convert(AbstractArray{Int32}, q_len), convert(AbstractArray{Int32}, k_len))

Base.@propagate_inbounds function getmask_at(m::Indexer{<:BiLengthMask}, i, j, I...)
    ql = m.q_len[I...]
    kl = m.k_len[I...]
    return i <= kl && j <= ql
end

####################  Wrap Mask  ####################

struct FlipMask{M} <: AbstractAttenMask
    mask::M
end

Base.:!(m::AbstractAttenMask) = FlipMask(m)
Base.:!(m::FlipMask) = m.mask

Broadcast.broadcastable(m::FlipMask) = Broadcast.broadcasted(!, Broadcast.broadcastable(m.mask))

Base.show(io::IO, m::FlipMask) = (print(io, '!'); show(io, m.mask); io)

struct CombinedMask{C, Ts<:Tuple} <: AbstractAttenMask
    f::C
    masks::Ts
end

Base.:|(m1::AbstractAttenMask, m2::AbstractAttenMask) = CombinedMask(|, (m1, m2))
Base.:|(m1::CombinedMask{typeof(|)}, m2::AbstractAttenMask) = CombinedMask(|, (m1.masks..., m2))
Base.:|(m1::AbstractAttenMask, m2::CombinedMask{typeof(|)}) = CombinedMask(|, (m1, m2.masks...))
Base.:|(m1::CombinedMask{typeof(|)}, m2::CombinedMask{typeof(|)}) = CombinedMask(|, (m1.masks..., m2.masks...))
Base.:&(m1::AbstractAttenMask, m2::AbstractAttenMask) = CombinedMask(&, (m1, m2))
Base.:&(m1::CombinedMask{typeof(&)}, m2::AbstractAttenMask) = CombinedMask(&, (m1.masks..., m2))
Base.:&(m1::AbstractAttenMask, m2::CombinedMask{typeof(&)}) = CombinedMask(&, (m1, m2.masks...))
Base.:&(m1::CombinedMask{typeof(&)}, m2::CombinedMask{typeof(&)}) = CombinedMask(&, (m1.masks..., m2.masks...))

Broadcast.broadcastable(m::CombinedMask) = Broadcast.broadcasted(m.f, map(Broadcast.broadcastable, m.masks)...)

function Base.show(io::IO, m::CombinedMask)
    show(io, first(m.masks))
    for mask in Base.tail(m.masks)
        need_par = mask isa CombinedMask
        print(io, ' ', m.f, need_par ? " (" : ' ')
        show(io, mask)
        need_par && print(io, ')')
    end
    io
end

struct BatchedMask{M<:AbstractArrayMask}
    mask::M
    batch_dim::Int
end


"""
Trait-like type for holding operation related argument, defined how the mask should be apply to input array
"""
abstract type AbstractMaskOp end

# """
# Each mask is associate with a default maskop
# """
# MaskOp(m) = NaiveAttenMaskOp()
# MaskOp(::Nothing) = nothing

# """
#     apply_mask(maskop, mask, score)

# applying the mask to score according to maskop

# Q: why seperate maskop and mask?
# A: It's possible that we want to apply different maskop, but
# the mask data is the same.
# """
# apply_mask(m, s) = apply_mask(MaskOp(m), m, s)

# struct AttenDropout <: Abstract 
#     mask::RandomMaskOp
# end

# """
# m = CausalMaskOp() | GenericAttenMask(mask) | RandomMaskOp(0.5)
# m::CombinedMask{typeof(|), Tuple{CausalMaskOp, GenericAttenMask, RandomMaskOp}}
# apply_mask(GenericAttenMaskOp(.+, static(true), 1e-9), 
#            m, score)
# """
