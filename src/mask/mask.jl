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

apply_mask(::Nothing, s) = s
apply_mask(_, ::Nothing, s) = s
apply_mask(m, s) = apply_mask(NaiveMaskOp(), m, s)

struct NaiveMaskOp <: AbstractMaskOp end

"""
    apply_mask(op::NaiveMaskOp, mask::AbstractAttenMask, score)

Directly broadcast multiply mask to attention score, i.e. `score .* mask`.
"""
apply_mask(op::NaiveMaskOp, mask::AbstractAttenMask, score) = score .* mask

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

function getmask(m::AbstractAttenMask, score, scale = one(eltype(score)))
    tmp = similar(score)
    @. tmp = m * scale
    return tmp
end

function apply_broadcast_mask(f, mask::AbstractAttenMask, score, scale)
    @. f(score, mask * scale)
end

"""
    apply_mask(op::GenericMaskOp, mask::AbstractAttenMask, score)

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
function apply_mask(op::GenericMaskOp, mask::AbstractAttenMask, score)
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
Base.eltype(::AbstractAttenMask) = Bool
Base.@propagate_inbounds Broadcast.newindex(arg::AbstractAttenMask, I::CartesianIndex) = I
Base.@propagate_inbounds Broadcast.newindex(arg::AbstractAttenMask, I::Integer) = I

const MaskIndexer = Indexer{<:AbstractAttenMask}
Base.@propagate_inbounds Broadcast.newindex(arg::MaskIndexer, I::CartesianIndex) = I
Base.@propagate_inbounds Broadcast.newindex(arg::MaskIndexer, I::Integer) = I
Base.eltype(::MaskIndexer) = Bool

GetIndexer(m::AbstractDatalessMask) = m

Base.@propagate_inbounds Base.getindex(m::AbstractAttenMask, i::CartesianIndex) = m[Tuple(i)]
Base.@propagate_inbounds Base.getindex(m::AbstractAttenMask, I::Tuple) = GetIndexer(m)[I...]
Base.@propagate_inbounds Base.getindex(m::M, I::Integer...) where {M <: Union{<:AbstractWrapperMask, <:AbstractArrayMask}} = m[I]
Base.@propagate_inbounds Base.getindex(m::MaskIndexer, i::CartesianIndex) = m[Tuple(i)]
Base.@propagate_inbounds Base.getindex(m::MaskIndexer, I::Tuple) = m[I...]

Adapt.adapt(to::CUDA.Adaptor, m::AbstractArrayMask) = Indexer{typeof(m)}(map(Base.Fix1(Adapt.adapt, to), GetIndexer(m).__fields))

randomness(::AbstractAttenMask) = static(false)
