abstract type AbstractAttenMask end

struct GenericAttenMask{N, M<:AbstractArray{Bool, N}} <: AbstractAttenMask
    mask::M
end

function GenericAttenMask(mask)
    N = ndims(mask)
    bmask = convert(AbstractArray{Bool, N}, mask)
    return GenericAttenMask{N, typeof(bmask)}(bmask)
end

abstract type AbstractAttenMaskOp end

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
function apply_mask(op::NaiveAttenMaskOp, mask, score)
    gmask = convert(GenericAttenMask, mask)
    return apply_naive_mask(gmask.mask, score)
end

function apply_naive_mask(mask, score)
    return score .* mask
end

struct GenericAttenMaskOp{F, B<:StaticBool, T} <: AbstractAttenMaskOp
    apply::F
    flip::B
    scale::T
end

GenericAttenMaskOp(apply, flip::Bool, scale) = GenericAttenMaskOp(apply, static(flip), scale)

# softmax norm default value
GenericAttenMaskOp() = GenericAttenMaskOp(.+, static(true), -1e9)

function apply_mask(op::GenericAttenMaskOp, mask, score)
    gmask = convert(GenericAttenMask, mask)
    return apply_generic_mask(op, gmask.mask, score)
end

"""
Equivalent to `op.apply(score, op.scale .* (op.flip ? .! mask : mask))`.

For example: `apply_generic_mask(GenericAttenMaskOp(.+, static(true), -1e9), mask, score) == @. score + (!mask * -1e9)`.
"""
function apply_generic_mask(op::GenericAttenMaskOp, mask, score)
    scale = convert(eltype(score), op.scale)
    apply = op.apply
    m = Base.broadcasted(*, Bool(op.flip) ? Base.broadcasted(!, mask) : mask, scale)
    if apply isa Base.BroadcastFunction
        masked_score = Base.Broadcast.materialize(Base.broadcasted(apply.f, score, m))
    else
        masked_score = apply(score, Base.Broadcast.materialize(m))
    end
    return masked_score
end

struct LengthMaskOp <: AbstractAttenMaskOp end
MaskOp(::LengthMask) = LengthMaskOp()

struct SymmetricLengthAttenMask{N, L <: AbstractArray{Int32, N}} <: AbstractAttenMask
    max_length::Int32
    lengths::L
end

function SymmetricLengthAttenMask(lengths)
    N = ndims(lengths)
    max_length = maximum(lengths)
    lens = convert(AbstractArray{Int32, N}, lengths)
    return SymmetricLengthAttenMask{N, typeof(lens)}(max_length, lens)
end

loc(c::CartesianIndex) = CartesianIndex(Base.tail(Base.tail(c.I)))

function Base.convert(::Type{GenericAttenMask}, m::SymmetricLengthAttenMask)
    len = m.lengths
    cinds = CartesianIndices((m.max_length, m.max_length, size(len)...))
    refl = Ref(len)
    mask = @. (getindex(cinds, 1) <= getindex(refl, loc(cinds))) & (getindex(cinds, 2) <= getindex(refl, loc(cinds)))
    return GenericAttenMask(mask)
end


# struct LengthAttenMask{N, L <: AbstractArray{Int32, N}} <: AbstractAttenMask
#     max_q_length::Int32
#     max_k_length::Int32
#     q_lengths::L
#     k_lengths::L
# end

# function LengthAttenMask(q_lengths, k_lengths)
#     @assert ndims(q_lengths) == ndims(k_lengths)
#     N = ndims(q_lengths)
#     max_q_length = maximum(q_lengths)
#     max_k_length = maximum(k_lengths)
#     qlens = convert(AbstractArray{Int32, N}, q_lengths)
#     klens = convert(AbstractArray{Int32, N}, k_lengths)
#     return LengthAttenMask{N, typeof(qlens)}(max_q_length, max_k_length, qlens, klens)
# end

# function Base.convert(::Type{GenericAttenMask}, m::LengthAttenMask)
#     cinds = CartesianIndices((m.max_k_length, m.max_q_length, size(m.q_lengths)...))
#     refq = Ref(m.q_lengths)
#     refk = Ref(m.k_lengths)
#     mask = @. (getindex(cinds, 1) <= getindex(refk, loc(cinds))) & (getindex(cinds, 2) <= getindex(refq, loc(cinds)))
#     return GenericAttenMask(mask)
# end
