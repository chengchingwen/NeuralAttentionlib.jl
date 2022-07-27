####################  Array Mask  ####################

struct GenericMask{N, M <:AbstractArray{Bool, N}} <: AbstractArrayMask
    mask::M
end
GenericMask(mask::AbstractArray) = GenericMask(convert(AbstractArray{Bool}, mask))

Base.ndims(::GenericMask{N}) where N = N

GenericMask(mask) = GenericMask(convert(AbstractArray{Bool}, mask))

adapt_structure(to, x::GenericMask) = GenericMask(adapt(to, x.mask))

Base.@propagate_inbounds Base.getindex(m::Indexer{<:GenericMask}, I::Integer...) = m.mask[I...]

AxesConstraint(m::GenericMask) = (NDimConstraint(ndims(m)), ntuple(i->DimConstraint(i, size(m.mask, i)), ndims(m))...)

struct SymLengthMask{N, L <: AbstractArray{Int32, N}} <: AbstractArrayMask
    len::L
end

Base.ndims(::SymLengthMask{N}) where N = N + 2

SymLengthMask(len::Integer) = SymLengthMask(Int32[len])
SymLengthMask(len::AbstractArray) = SymLengthMask(convert(AbstractArray{Int32}, len))

adapt_structure(to, x::SymLengthMask) = SymLengthMask(adapt(to, x.len))

Base.@propagate_inbounds Base.getindex(m::Indexer{<:SymLengthMask}, i, j) = (l = m.len[]; i <= l && j <= l)
Base.@propagate_inbounds function Base.getindex(m::Indexer{<:SymLengthMask}, i, j, J::Integer...)
    l = m.len[J...]
    return i <= l && j <= l
end

AxesConstraint(m::SymLengthMask{N}) where N = length(m.len) == 1 ? # only one mask
    (NDimConstraint(2, true), All1Constraint(3, ndims(m))) :
    (NDimConstraint(ndims(m)), ntuple(i->DimConstraint(i+2, size(m.len, i)), N)...)

struct BiLengthMask{N, L <: AbstractArray{Int32, N}} <: AbstractArrayMask
    q_len::L
    k_len::L
end

Base.ndims(::BiLengthMask{N}) where N = N + 2

function BiLengthMask(q_len, k_len)
    @assert size(q_len) == size(k_len)
    return BiLengthMask(convert(AbstractArray{Int32}, q_len), convert(AbstractArray{Int32}, k_len))
end

adapt_structure(to, x::BiLengthMask) = BiLengthMask(adapt(to, x.q_len), adapt(to, x.k_len))

Base.@propagate_inbounds Base.getindex(m::Indexer{<:BiLengthMask}, i, j) = i <= m.k_len[] && j <= m.q_len[]
Base.@propagate_inbounds function Base.getindex(m::Indexer{<:BiLengthMask}, i, j, J::Integer...)
    ql = m.q_len[J...]
    kl = m.k_len[J...]
    return i <= kl && j <= ql
end

AxesConstraint(m::BiLengthMask{N}) where N = length(m.q_len) == 1 ? # only one mask
    (NDimConstraint(2, true), All1Constraint(3, ndims(m))) :
    (NDimConstraint(ndims(m)), ntuple(i->DimConstraint(i+2, size(m.q_len, i)), N)...)
