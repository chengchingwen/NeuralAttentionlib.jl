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

GenericAttenMaskOp(::typeof(+), flip::Bool, scale) = GenericAttenMaskOp(.+, flip, scale)
GenericAttenMaskOp(::typeof(-), flip::Bool, scale) = GenericAttenMaskOp(.+, flip, -scale)
GenericAttenMaskOp(::typeof(.-), flip::Bool, scale) = GenericAttenMaskOp(.+, flip, -scale)
GenericAttenMaskOp(::typeof(./), flip::Bool, scale) = GenericAttenMaskOp(.*, flip, inv(scale))

# softmax norm default value
GenericAttenMaskOp() = GenericAttenMaskOp(.+, static(true), -1e9)

getmask(m::AbstractAttenMask, score) = getmask(m, score, one(eltype(score)))
function getmask(m::AbstractAttenMask, score, scale)
    tmp = similar(score)
    @. tmp = m * scale
    return tmp
end

function apply_broadcast_mask(f, mask, score, scale)
    @. f(score, mask * scale)
end

"""
Equivalent to `op.apply(score, op.scale .* (op.flip ? .! mask : mask))`.

For example: `apply_generic_mask(GenericAttenMaskOp(.+, static(true), -1e9), mask, score) == @. score + (!mask * -1e9)`.
"""
function apply_mask(op::GenericAttenMaskOp, mask::AbstractAttenMask, score)
    scale = convert(eltype(score), op.scale)
    apply = op.apply
    m = Bool(op.flip) ? !mask : mask

    if apply isa Base.BroadcastFunction
        masked_score = apply_broadcast_mask(apply.f, m, score, scale)
    else
        tmp = getmask(m, score, scale)
        masked_score = apply(tmp, score)
    end
    return masked_score
end

abstract type AbstractDatalessMask <: AbstractAttenMask end
abstract type AbstractArrayMask <: AbstractAttenMask end
abstract type AbstractWrapperMask <: AbstractAttenMask end

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

GetIndexer(m::AbstractDatalessMask) = m

Base.@propagate_inbounds Base.getindex(m::AbstractAttenMask, i::CartesianIndex) = m[Tuple(i)]
Base.@propagate_inbounds Base.getindex(m::AbstractAttenMask, I::Tuple) = GetIndexer(m)[I...]
Base.@propagate_inbounds Base.getindex(m::M, I::Integer...) where {M <: Union{<:AbstractWrapperMask, <:AbstractArrayMask}} = m[I]
Base.@propagate_inbounds Base.getindex(m::MaskIndexer, i::CartesianIndex) = m[Tuple(i)]
Base.@propagate_inbounds Base.getindex(m::MaskIndexer, I::Tuple) = m[I...]

using Adapt
using CUDA
import Adapt: adapt_structure, adapt

Adapt.adapt(to::CUDA.Adaptor, m::AbstractArrayMask) = Indexer{typeof(m)}(map(Base.Fix1(Adapt.adapt, to), GetIndexer(m).__fields))
