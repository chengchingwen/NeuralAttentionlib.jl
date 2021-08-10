"""
Trait-like type for holding operation related argument, defined how the mask should be apply to input array
"""
abstract type AbstractAttenMaskOp end

"""
Wrapper type for mask data, can be viewed as AbstractArray{Bool}
"""
abstract type AbstractAttenMask end

apply_mask(::Nothing, s) = s
apply_mask(_, ::Nothing, s) = s
apply_mask(::Nothing, m, s) = apply_mask(m, s)
apply_mask(m, s) = apply_mask(MaskOp(m), m, s)
MaskOp(m) = NaiveAttenMaskOp()
MaskOp(::Nothing) = nothing

struct NaiveAttenMaskOp <: AbstractAttenMaskOp end

"""
Directly broadcast multiply mask to attention score.
"""
apply_mask(op::NaiveAttenMaskOp, mask::AbstractAttenMask, score) = score .* mask


struct GenericAttenMaskOp{F, B<:StaticBool, T} <: AbstractAttenMaskOp
    apply::F
    flip::B
    scale::T
end

GenericAttenMaskOp(apply, flip::Bool, scale) = GenericAttenMaskOp(apply, static(flip), scale)

# softmax norm default value
GenericAttenMaskOp() = GenericAttenMaskOp(.+, static(true), -1e9)


"""
Equivalent to `op.apply(score, op.scale .* (op.flip ? .! mask : mask))`.

For example: `apply_generic_mask(GenericAttenMaskOp(.+, static(true), -1e9), mask, score) == @. score + (!mask * -1e9)`.
"""
function apply_mask(op::GenericAttenMaskOp, mask::AbstractAttenMask, score)
    scale = convert(eltype(score), op.scale)
    apply = op.apply
    m = Base.broadcasted(*, Bool(op.flip) ? !mask : mask, scale)
    if apply isa Base.BroadcastFunction
        masked_score = Base.Broadcast.materialize(Base.broadcasted(apply.f, score, m))
    else
        masked_score = apply(score, Base.Broadcast.materialize(m))
    end
    return masked_score
end

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

using Adapt
using CUDA
import Adapt: adapt_structure, adapt

Adapt.adapt(to::CUDA.Adaptor, m::AbstractArrayMask) = Indexer{typeof(m)}(map(Base.Fix1(Adapt.adapt, to), GetIndexer(m).__fields))
