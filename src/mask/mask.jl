Base.@enum MASKDATA::UInt8 DATALESS ARRAYDATA MIXDATA
Base.@enum MASKTYPE::UInt8 ATTENTION SEQUENCE MIXTYPE
abstract type AbstractMask{D, T} end

abstract type AbstractWrapperMask{D, T} <: AbstractMask{D, T} end
const AbstractAttenMask{D} = AbstractMask{D, ATTENTION}
const AbstractSeqMask{D} = AbstractMask{D, SEQUENCE}

const AbstractDatalessAttenMask = AbstractAttenMask{DATALESS}
const AbstractArrayDataAttenMask = AbstractAttenMask{ARRAYDATA}
const AbstractDatalessSeqMask = AbstractSeqMask{DATALESS}
const AbstractArrayDataSeqMask = AbstractSeqMask{ARRAYDATA}

MASKDATA(::AbstractMask{D, T}) where {D, T} = D
MASKTYPE(::AbstractMask{D, T}) where {D, T} = T
MASKDATA(t1::MASKDATA, t2::MASKDATA) = t1 == t2 ? t1 : MIXDATA
MASKTYPE(t1::MASKTYPE, t2::MASKTYPE) = t1 == t2 ? t1 : MIXTYPE

_combine_masktag(f, t1::T, t2::T) where {T <: Union{MASKDATA, MASKTYPE}} = f(t1, t2)
_combine_masktag(f, t::Tuple{T}) where {T <: Union{MASKDATA, MASKTYPE}} = t[1]
_combine_masktag(f, t::NTuple{2, T}) where {T <: Union{MASKDATA, MASKTYPE}} = _combine_masktag(f, t[1], t[2])
function _combine_masktag(f, t::Tuple{T, T, T, Vararg{T}}) where {T <: Union{MASKDATA, MASKTYPE}}
    return _combine_masktag(f, _combine_masktag(f, t[1], t[2]), Base.tail(Base.tail(t)))
end
_combine_masktag(f, t0::T, ::Tuple{}) where {T <: Union{MASKDATA, MASKTYPE}} = t0
function _combine_masktag(f, t0::T, t::Tuple{T, Vararg{T}}) where {T <: Union{MASKDATA, MASKTYPE}}
    return _combine_masktag(f, _combine_masktag(f, t0, t[1]), Base.tail(t))
end

function _combine_masktag(f, t0::T, m::Tuple{AbstractMask, Vararg{AbstractMask}}) where {T <: Union{MASKDATA, MASKTYPE}}
    _combine_masktag(f, _combine_masktag(f, t0, T(m[1])), Base.tail(m))
end
function _combine_masktag(f::Type{T}, m::Tuple{AbstractMask, Vararg{AbstractMask}}) where {T <: Union{MASKDATA, MASKTYPE}}
    return _combine_masktag(f, T(m[1]), Base.tail(m))
end

combine_maskdatatag(args...) = _combine_masktag(MASKDATA, args...)
combine_masktypetag(args...) = _combine_masktag(MASKTYPE, args...)

"""
    AbstractMask

Abstract type for mask data.
"""
AbstractMask

"""
    AbstractSeqMask <: AbstractMask

Abstract type for mask data specifically for sequence.
"""
AbstractSeqMask

"""
    AbstractAttenMask <: AbstractMask

Abstract type for mask data specifically for attention.
"""
AbstractAttenMask

AttenMask(m::AbstractAttenMask) = m
SeqMask(m::AbstractSeqMask) = m

Base.eltype(::AbstractMask) = Bool
randomness(::AbstractMask) = static(false)

"""
    AbstractMaskOp

Trait-like abstract type for holding operation related argument, defined how the mask should be apply to input array
"""
abstract type AbstractMaskOp end

apply_mask(::Nothing, s) = s
apply_mask(_, ::Nothing, s) = s
apply_mask(m, s) = apply_mask(NaiveMaskOp(), m, s)

struct NaiveMaskOp <: AbstractMaskOp end

"""
    apply_mask(op::NaiveMaskOp, mask::AbstractMask, score)

Directly broadcast multiply mask to attention score, i.e. `score .* mask`.
"""
apply_mask(op::NaiveMaskOp, mask::AbstractMask, score) = score .* mask
apply_mask!(op::NaiveMaskOp, mask::AbstractMask, score) = score .*= mask

struct GenericMaskOp{F, B<:StaticBool, T} <: AbstractMaskOp
    apply::F
    flip::B
    scale::T
end
GenericMaskOp(apply::F, flip::Bool, scale) where F = GenericMaskOp(apply, static(flip), scale)

GenericMaskOp(::typeof(+), flip::StaticBool, scale) = GenericMaskOp(.+, flip, scale)
GenericMaskOp(::typeof(-), flip::StaticBool, scale) = GenericMaskOp(.+, flip, -scale)
GenericMaskOp(::typeof(.-), flip::StaticBool, scale) = GenericMaskOp(.+, flip, -scale)

# softmax norm default value
GenericMaskOp() = GenericMaskOp(.+, static(true), -1e9)

getmask(m::AbstractMask, score, scale = one(eltype(score))) = getmask!(similar(score), m, score, scale)
getmask!(tmp, m::AbstractMask, score, scale = one(eltype(score))) = @. tmp = m * scale

apply_broadcast_mask(f, mask::AbstractMask, score, scale) = @. f(score, mask * scale)
apply_broadcast_mask!(f, mask::AbstractMask, score, scale) = @. score = f(score, mask * scale)

"""
    apply_mask(op::GenericMaskOp, mask::AbstractMask, score)

Equivalent to `op.apply(score, op.scale .* (op.flip ? .! mask : mask))`.

# Example

```julia
julia> x = randn(10, 10);

julia> m = CausalMask()
CausalMask()

julia> apply_mask(GenericMaskOp(.+, true, -1e9), m, x) ==  @. x + (!m * -1e9)
true

```
"""
function apply_mask(op::GenericMaskOp, mask::AbstractMask, score)
    scale = convert(eltype(score), op.scale)
    if isinf(scale)
        scale = scale > 0 ? prevfloat(scale) :  nextfloat(scale)
        @assert !isinf(scale)
    end
    apply = op.apply
    m = as_bool(op.flip) ? !mask : mask

    if apply isa Base.BroadcastFunction
        masked_score = apply_broadcast_mask(apply.f, m, score, scale)
    else
        tmp = getmask(m, score, scale)
        masked_score = apply(tmp, score)
    end
    return masked_score
end

function apply_mask!(op::GenericMaskOp, mask::AbstractMask, score)
    scale = convert(eltype(score), op.scale)
    if isinf(scale)
        scale = scale > 0 ? prevfloat(scale) :  nextfloat(scale)
        @assert !isinf(scale)
    end
    apply = op.apply
    m = as_bool(op.flip) ? !mask : mask

    if apply isa Base.BroadcastFunction
        return apply_broadcast_mask!(apply.f, m, score, scale)
    else
        tmp = getmask(m, score, scale)
        tmp .= apply(tmp, score)
        return tmp
    end
end
