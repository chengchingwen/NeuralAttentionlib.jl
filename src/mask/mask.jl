"""
    AbstractMaskOp

Trait-like abstract type for holding operation related argument, defined how the mask should be apply to input array
"""
abstract type AbstractMaskOp end

"""
    AbstractMask

Abstract type for mask data.
"""
abstract type AbstractMask end

"""
    AbstractSequenceMask <: AbstractMask

Abstract type for mask data specifically for sequence.
"""
abstract type AbstractSequenceMask <: AbstractMask end

"""
    AbstractAttenMask <: AbstractMask

Abstract type for mask data specifically for attention.
"""
abstract type AbstractAttenMask <: AbstractMask end

AttenMask(m::AbstractAttenMask) = m

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

abstract type AbstractWrapperMask <: AbstractMask end

abstract type AbstractDatalessMask <: AbstractAttenMask end
abstract type AbstractArrayMask <: AbstractAttenMask end

Broadcast.broadcastable(m::AbstractMask) = m
Base.eltype(::AbstractMask) = Bool
Base.@propagate_inbounds Broadcast.newindex(arg::AbstractMask, I::CartesianIndex) = I
Base.@propagate_inbounds Broadcast.newindex(arg::AbstractMask, I::Integer) = I

const MaskIndexer = Indexer{<:AbstractMask}
Base.@propagate_inbounds Broadcast.newindex(arg::MaskIndexer, I::CartesianIndex) = I
Base.@propagate_inbounds Broadcast.newindex(arg::MaskIndexer, I::Integer) = I
Base.eltype(::MaskIndexer) = Bool

GetIndexer(m::AbstractDatalessMask, dest_size = nothing) = m

Base.@propagate_inbounds Base.getindex(m::AbstractMask, i::CartesianIndex) = m[Tuple(i)]
Base.@propagate_inbounds Base.getindex(m::AbstractMask, I::Tuple) = GetIndexer(m)[I...]
Base.@propagate_inbounds Base.getindex(m::M, I::Integer...) where {M <: Union{<:AbstractWrapperMask, <:AbstractArrayMask}} = m[I]
Base.@propagate_inbounds Base.getindex(m::MaskIndexer, i::CartesianIndex) = m[Tuple(i)]
Base.@propagate_inbounds Base.getindex(m::MaskIndexer, I::Tuple) = m[I...]

Adapt.adapt(to::CUDA.Adaptor, m::AbstractArrayMask) = Indexer{typeof(m)}(map(Base.Fix1(Adapt.adapt, to), GetIndexer(m).__fields))

randomness(::AbstractMask) = static(false)
require_dest(::AbstractMask) = static(false)
