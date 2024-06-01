####################  Sequence Mask  ####################

AttenMask(q::AbstractSeqMask, k::AbstractSeqMask) = BiSequenceMask(q, k)

struct GenericSeqMask{N, M <: AbstractArray{Bool, N}} <: AbstractSeqMask{ARRAYDATA}
    mask::M
    function GenericSeqMask{N, M}(mask::M) where {N, M <: AbstractArray{Bool, N}}
        @assert size(mask, 1) == 1
        return new{N, M}(mask)
    end
end
const GenericSequenceMask = GenericSeqMask

GenericSeqMask{N}(mask::AbstractArray{Bool}) where N = GenericSeqMask{N, typeof(mask)}(mask)
function GenericSeqMask(mask::AbstractArray{Bool})
    if size(mask, 1) != 1
        mask = reshape(mask, (1, size(mask)...))
    end
    return GenericSeqMask{ndims(mask), typeof(mask)}(mask)
end
GenericSeqMask(mask) = GenericSeqMask(convert(AbstractArray{Bool}, mask))

Base.ndims(::GenericSeqMask{N}) where N = N

adapt_structure(to, x::GenericSeqMask{N}) where N = GenericSeqMask{N}(adapt(to, x.mask))

Base.@propagate_inbounds maskgetindex(::Dims, m::GenericSeqMask, I::Integer...) = m.mask[1, Base.tail(I)...]

AxesConstraint(m::GenericSeqMask) = (NDimConstraint(ndims(m)), ntuple(i->DimConstraint(i+1, size(m.mask, i+1), i < 2), ndims(m)-1)...)

AttenMask(m::GenericSeqMask) = GenericAttenMask(PermutedDimsArray(m.mask, ntuple(i-> i == 1 ? 2 : i == 2 ? 1 : i, Val(ndims(m)))) .* m.mask)

_tn(m, s) = ntuple(identity, static(ndims(m)) - s)
lengths(m::GenericSeqMask) = reshape(sum(m.mask; dims = _tn(m, static(1))), :)
lengths(m::GenericSeqMask{2}) = m.mask


struct LengthMask{N, L <: AbstractArray{Int32, N}} <: AbstractSeqMask{ARRAYDATA}
    len::L
end

Base.ndims(::LengthMask{N}) where N = N + 2

LengthMask(len::Integer) = LengthMask(Int32[len])
LengthMask(len::AbstractArray) = LengthMask(convert(AbstractArray{Int32}, len))

adapt_structure(to, x::LengthMask) = LengthMask(adapt(to, x.len))

Base.@propagate_inbounds function maskgetindex(::Dims, m::LengthMask, _::Integer, j::Integer, J::Integer...)
    l = m.len[J...]
    return j <= l
end

AxesConstraint(m::LengthMask{N}) where N = length(m.len) == 1 ? # only one mask
    (NDimConstraint(2, true), All1Constraint(3, ndims(m))) :
    (NDimConstraint(ndims(m)), ntuple(i->DimConstraint(i+2, size(m.len, i)), N)...)

AttenMask(m::LengthMask) = SymLengthMask(m.len)
AttenMask(q::LengthMask, k::LengthMask) = BiLengthMask(q.len, k.len)

Base.:(+)(m::LengthMask, l::Integer) = LengthMask(m.len .+ l)
Base.:(*)(m::LengthMask, l::Integer) = LengthMask(m.len .* l)
Base.:(-)(m::LengthMask, l::Integer) = LengthMask(m.len .- l)
Base.:(+)(l::Integer, m::LengthMask) = m + l
Base.:(*)(l::Integer, m::LengthMask) = m * l

lengths(m::LengthMask) = reshape(sum(m.len; dims = _tn(m, static(3))), :)
lengths(m::LengthMask{1}) = m.len


struct RevLengthMask{N, L <: AbstractArray{Int32, N}} <: AbstractSeqMask{ARRAYDATA}
    len::L
end

Base.ndims(::RevLengthMask{N}) where N = N + 2

RevLengthMask(len::Integer) = RevLengthMask(Int32[len])
RevLengthMask(len::AbstractArray) = RevLengthMask(convert(AbstractArray{Int32}, len))

adapt_structure(to, x::RevLengthMask) = RevLengthMask(adapt(to, x.len))

Base.@propagate_inbounds function maskgetindex(destsize::Dims, m::RevLengthMask, _::Integer, j::Integer, J::Integer...)
    cl = destsize[2]
    l = m.len[J...]
    return cl - l < j
end

AxesConstraint(m::RevLengthMask{N}) where N = length(m.len) == 1 ? # only one mask
    (NDimConstraint(2, true), All1Constraint(3, ndims(m))) :
    (NDimConstraint(ndims(m)), ntuple(i->DimConstraint(i+2, size(m.len, i)), N)...)

AttenMask(m::RevLengthMask) = RevSymLengthMask(m.len)
AttenMask(q::RevLengthMask, k::RevLengthMask) = RevBiLengthMask(q.len, k.len)

Base.:(+)(m::RevLengthMask, l::Integer) = RevLengthMask(m.len .+ l)
Base.:(*)(m::RevLengthMask, l::Integer) = RevLengthMask(m.len .* l)
Base.:(-)(m::RevLengthMask, l::Integer) = RevLengthMask(m.len .- l)
Base.:(+)(l::Integer, m::RevLengthMask) = m + l
Base.:(*)(l::Integer, m::RevLengthMask) = m * l

lengths(m::RevLengthMask) = reshape(sum(m.len; dims = _tn(m, static(3))), :)
lengths(m::RevLengthMask{1}) = m.len
