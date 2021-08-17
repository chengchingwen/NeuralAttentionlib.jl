"""
    AbstractAttenMaskOp

Trait-like abstract type for holding operation related argument, defined how the mask should be apply to input array
"""
abstract type AbstractAttenMaskOp end

"""
    AbstractAttenMask

Abstract type for mask data, can be viewed as `AbstractArray{Bool}`
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
    apply_mask(op::NaiveAttenMaskOp, mask::AbstractAttenMask, score)

Directly broadcast multiply mask to attention score, i.e. `score .* mask`.
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
    apply_mask(op::GenericAttenMaskOp, mask::AbstractAttenMask, score)

Equivalent to `op.apply(score, op.scale .* (op.flip ? .! mask : mask))`.

# Example

```julia
julia> x = randn(10, 10);

julia> m = CausalMask()
CausalMask()

julia> apply_mask(GenericAttenMaskOp(.+, true, -1e9), m, x) ==  @. x + (!m * -1e9)
true

```
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

@init @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
    Adapt.adapt(to::CUDA.Adaptor, m::AbstractArrayMask) = Indexer{typeof(m)}(map(Base.Fix1(Adapt.adapt, to), GetIndexer(m).__fields))
end
