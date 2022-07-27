####################  Array Mask  ####################

struct GenericAttenMask{N, M <:AbstractArray{Bool, N}} <: AbstractArrayMask
    mask::M
end
GenericAttenMask(mask) = GenericAttenMask(convert(AbstractArray{Bool}, mask))

Base.ndims(::GenericAttenMask{N}) where N = N

adapt_structure(to, x::GenericAttenMask) = GenericAttenMask(adapt(to, x.mask))

Base.@propagate_inbounds Base.getindex(m::Indexer{<:GenericAttenMask}, I::Integer...) = m.mask[I...]

AxesConstraint(m::GenericAttenMask) = (NDimConstraint(ndims(m)), ntuple(i->DimConstraint(i, size(m.mask, i)), ndims(m))...)

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

Base.:(+)(m::SymLengthMask, l::Integer) = SymLengthMask(m.len .+ l)
Base.:(*)(m::SymLengthMask, l::Integer) = SymLengthMask(m.len .* l)
Base.:(-)(m::SymLengthMask, l::Integer) = SymLengthMask(m.len .- l)
Base.:(+)(l::Integer, m::SymLengthMask) = m + l
Base.:(*)(l::Integer, m::SymLengthMask) = m * l

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

Base.:(+)(m::BiLengthMask, l::Integer) = BiLengthMask(m.q_len .+ l, m.k_len .+ l)
Base.:(*)(m::BiLengthMask, l::Integer) = BiLengthMask(m.q_len .* l, m.k_len .* l)
Base.:(-)(m::BiLengthMask, l::Integer) = BiLengthMask(m.q_len .- l, m.k_len .- l)
Base.:(+)(l::Integer, m::BiLengthMask) = m + l
Base.:(*)(l::Integer, m::BiLengthMask) = m * l
