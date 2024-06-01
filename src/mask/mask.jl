Base.@enum MASKDATA::UInt8 DATALESS ARRAYDATA MIXDATA
Base.@enum MASKTYPE::UInt8 ATTENTION SEQUENCE MIXTYPE
const MASKTAG = Union{MASKDATA, MASKTYPE}
abstract type AbstractMask{D, T} end
abstract type AbstractWrapperMask{D, T} <: AbstractMask{D, T} end

const AbstractAttenMask{D} = AbstractMask{D, ATTENTION}
const AbstractSeqMask{D} = AbstractMask{D, SEQUENCE}
const AbstractArrayMask{T} = AbstractMask{ARRAYDATA, T}
const AbstractDatalessMask{T} = AbstractMask{DATALESS, T}

const AbstractDatalessAttenMask = AbstractAttenMask{DATALESS}
const AbstractArrayDataAttenMask = AbstractAttenMask{ARRAYDATA}
const AbstractDatalessSeqMask = AbstractSeqMask{DATALESS}
const AbstractArrayDataSeqMask = AbstractSeqMask{ARRAYDATA}

MASKDATA(t::MASKDATA) = t
MASKTYPE(t::MASKTYPE) = t
MASKDATA(::AbstractMask{D, T}) where {D, T} = D
MASKTYPE(::AbstractMask{D, T}) where {D, T} = T
MASKDATA(t1::MASKDATA, t2::MASKDATA) = t1 == t2 ? t1 : MIXDATA
MASKTYPE(t1::MASKTYPE, t2::MASKTYPE) = t1 == t2 ? t1 : MIXTYPE

_combine_masktag(::Type{T}, t::NTuple{1}) where T <: MASKTAG = T(t[1])
_combine_masktag(::Type{T}, t::Tuple) where T <: MASKTAG = T(T(first(t)), _combine_masktag(T, Base.tail(t)))
combine_maskdatatag(args...) = _combine_masktag(MASKDATA, args)
combine_masktypetag(args...) = _combine_masktag(MASKTYPE, args)

AttenMask(m::AbstractAttenMask) = m
SeqMask(m::AbstractSeqMask) = m

Base.eltype(::AbstractMask) = Bool

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
apply_mask(op::NaiveMaskOp, mask::AbstractMask, score) = apply_mask!(op, mask, copy(score))
apply_mask!(op::NaiveMaskOp, mask::AbstractMask, score) = _fast_broadcast!(*, score, GetIndexer(mask, size(score)))

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
getmask!(tmp, m::AbstractMask, score, scale = one(eltype(score))) = _fast_broadcast2!(identity, tmp, GetIndexer(m, size(score), convert(eltype(score), scale)))

apply_broadcast_mask(f, mask::AbstractMask, score, scale) = apply_broadcast_mask!(f, mask, copy(score), scale)
apply_broadcast_mask!(f, mask::AbstractMask, score, scale) = _fast_broadcast!(f, score, GetIndexer(mask, size(score), convert(eltype(score), scale))) #@. score = f(score, mask * scale)

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
apply_mask(op::GenericMaskOp, mask::AbstractMask, score) = apply_mask!(op, mask, copy(score))
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
        score .= apply(score, tmp)
        return score
    end
end
