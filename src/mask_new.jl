using Static
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

#const MaskIndexer = Indexer{<:AbstractArrayMask}
const MaskIndexer = Indexer{<:AbstractAttenMask}
Broadcast.BroadcastStyle(::Type{<:MaskIndexer}) = Broadcast.DefaultArrayStyle{0}()
Base.@propagate_inbounds Broadcast.newindex(arg::MaskIndexer, I::CartesianIndex) = I
Base.size(::MaskIndexer) = ()
Base.eltype(::MaskIndexer) = Bool

Base.getindex(m::AbstractAttenMask, I...) = getmask_at(m, I...)
Base.getindex(m::MaskIndexer, I...) = getmask_at(m, I...)

GetIndexer(m::AbstractDatalessMask) = m

getmask_at(m::AbstractAttenMask, i::CartesianIndex) = getmask_at(m, Tuple(i))
getmask_at(m::MaskIndexer, i::CartesianIndex) = getmask_at(m, Tuple(i))

getmask_at(m::M, I::Tuple) where {M >: AbstractDatalessMask} = getmask_at(GetIndexer(m), I)


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

####################  Array Mask  ####################

getmask_at(m::BandPartMask, I::Tuple) = (m.l < 0 || I[1] <= I[2] + m.l) && (m.u < 0 || I[1] >= I[2] - m.u)

struct SymLengthMask{N, L <: AbstractArray{Int32, N}} <: AbstractArrayMask
    len::L
end

SymLengthMask(len) = SymLengthMask(convert(AbstractArray{Int32}, len))

using Adapt
using CUDA
import Adapt: adapt_structure, adapt
adapt_structure(to, x::SymLengthMask) = SymLengthMask(adapt(to, x.len))

Adapt.adapt(to::CUDA.Adaptor, m::AbstractArrayMask) = Indexer{typeof(m)}(map(Base.Fix1(Adapt.adapt, to), GetIndexer(m).__fields))

# TODO: add boundcheck
Base.@propagate_inbounds function getmask_at(m::Indexer{<:SymLengthMask}, I::Tuple)
    i = I[1]
    j = I[2]
    l = m.len[Base.tail(Base.tail(I))...]
    return i <= l && j <= l
end

struct BiLengthMask{N, L <: AbstractArray{Int32, N}} <: AbstractArrayMask
    q_len::L
    k_len::L
end

BiLengthMask(q_len, k_len) = BiLengthMask(convert(AbstractArray{Int32}, q_len), convert(AbstractArray{Int32}, k_len))

adapt_structure(to, x::BiLengthMask) = BiLengthMask(adapt(to, x.q_len), adapt(to, x.k_len))

Base.@propagate_inbounds function getmask_at(m::Indexer{<:BiLengthMask}, I::Tuple)
    i = I[1]
    j = I[2]
    J = Base.tail(Base.tail(I))
    ql = m.q_len[J...]
    kl = m.k_len[J...]
    return i <= kl && j <= ql
end

####################  Wrap Mask  ####################

struct FlipMask{M} <: AbstractAttenMask
    mask::M
end

Base.:!(m::AbstractAttenMask) = FlipMask(m)
Base.:!(m::FlipMask) = m.mask

adapt_structure(to, x::FlipMask) = FlipMask(adapt(to, x.mask))
GetIndexer(m::FlipMask) = Indexer{typeof(m)}((mask = GetIndexer(m.mask),))

Adapt.adapt(to::CUDA.Adaptor, m::FlipMask) = Indexer{typeof(m)}((mask = adapt(to, m.mask),))

Base.@propagate_inbounds getmask_at(m::Indexer{<:FlipMask}, I::Tuple) = !getmask_at(m.mask, I)

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

Adapt.adapt(to::CUDA.Adaptor, m::CombinedMask) = Indexer{typeof(m)}((f = adapt(to, m.f),
                                                                     masks = map(Base.Fix1(adapt, to), m.masks)))
adapt_structure(to, x::CombinedMask) = CombinedMask(x.f, adapt(to, x.masks))
GetIndexer(m::CombinedMask) = Indexer{typeof(m)}((m.f, masks = map(GetIndexer, m.masks)))

@inline function _combine_getmask(f, masks, I)
    if length(masks) == 2
        m1 = getmask_at(masks[1], I)
        m2 = getmask_at(masks[2], I)
        return f(m1, m2)
    else
        m1 = getmask_at(masks[1], I)
        m2 = _combine_getmask(f, Base.tail(masks), I)
        return f(m1, m2)
    end
end

Base.@propagate_inbounds function getmask_at(m::Indexer{M}, I::Tuple) where M <: CombinedMask
    return _combine_getmask(m.f, m.masks, I)
end

function Base.show(io::IO, m::CombinedMask)
    print(io, '(')
    show(io, first(m.masks))
    for mask in Base.tail(m.masks)
        print(io, ' ', m.f, ' ')
        show(io, mask)
    end
    print(io, ')')
    io
end

struct BatchedMask{M<:AbstractArrayMask, S<:StaticInt} <: AbstractAttenMask
    mask::M
    batch_dim::S
end

BatchedMask(mask, batch_dim::Integer) = BatchedMask(mask, static(batch_dim))

Adapt.adapt(to::CUDA.Adaptor, m::BatchedMask) = Indexer{typeof(m)}((mask = adapt(to, m.mask), batch_dim = m.batch_dim))

adapt_structure(to, x::BatchedMask) = BatchedMask(adapt(to, x.mask), x.batch_dim)

GetIndexer(m::BatchedMask) = Indexer{typeof(m)}((mask = GetIndexer(m.mask), batch_dim = m.batch_dim))

Base.@propagate_inbounds function getmask_at(m::Indexer{M}, I::Tuple) where M <: BatchedMask
    i = I[1]
    j = I[2]
    J = ntuple(i->I[i-1+m.batch_dim], 1+length(I)-m.batch_dim)
    return getmask_at(m.mask, (i, j, J...))
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
