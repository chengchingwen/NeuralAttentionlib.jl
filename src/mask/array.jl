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

struct RevSymLengthMask{N, L <: AbstractArray{Int32, N}} <: AbstractArrayMask
    len::L
end

Base.ndims(::RevSymLengthMask{N}) where N = N + 2

require_dest(::RevSymLengthMask) = static(true)

RevSymLengthMask(len::Integer) = RevSymLengthMask(Int32[len])
RevSymLengthMask(len::AbstractArray) = RevSymLengthMask(convert(AbstractArray{Int32}, len))

adapt_structure(to, x::RevSymLengthMask) = RevSymLengthMask(adapt(to, x.len))

Base.@propagate_inbounds function Base.getindex(m::Indexer{<:RevSymLengthMask, <:Tuple}, i, j, J::Integer...)
    rl, cl = m.dest_size
    l = m.len[J...]
    return rl - l < i && cl - l < j
end

AxesConstraint(m::RevSymLengthMask{N}) where N = length(m.len) == 1 ? # only one mask
    (NDimConstraint(2, true), All1Constraint(3, ndims(m))) :
    (NDimConstraint(ndims(m)), ntuple(i->DimConstraint(i+2, size(m.len, i)), N)...)

Base.:(+)(m::RevSymLengthMask, l::Integer) = RevSymLengthMask(m.len .+ l)
Base.:(*)(m::RevSymLengthMask, l::Integer) = RevSymLengthMask(m.len .* l)
Base.:(-)(m::RevSymLengthMask, l::Integer) = RevSymLengthMask(m.len .- l)
Base.:(+)(l::Integer, m::RevSymLengthMask) = m + l
Base.:(*)(l::Integer, m::RevSymLengthMask) = m * l

struct RevBiLengthMask{N, L <: AbstractArray{Int32, N}} <: AbstractArrayMask
    q_len::L
    k_len::L
end

Base.ndims(::RevBiLengthMask{N}) where N = N + 2

require_dest(::RevBiLengthMask) = static(true)

function RevBiLengthMask(q_len, k_len)
    @assert size(q_len) == size(k_len)
    return RevBiLengthMask(convert(AbstractArray{Int32}, q_len), convert(AbstractArray{Int32}, k_len))
end

adapt_structure(to, x::RevBiLengthMask) = RevBiLengthMask(adapt(to, x.q_len), adapt(to, x.k_len))

Base.@propagate_inbounds function Base.getindex(m::Indexer{<:RevBiLengthMask, <:Tuple}, i, j, J::Integer...)
    rl, cl = m.dest_size
    ql = m.q_len[J...]
    kl = m.k_len[J...]
    return rl - kl < i && cl - ql < j
end

AxesConstraint(m::RevBiLengthMask{N}) where N = length(m.q_len) == 1 ? # only one mask
    (NDimConstraint(2, true), All1Constraint(3, ndims(m))) :
    (NDimConstraint(ndims(m)), ntuple(i->DimConstraint(i+2, size(m.q_len, i)), N)...)

Base.:(+)(m::RevBiLengthMask, l::Integer) = RevBiLengthMask(m.q_len .+ l, m.k_len .+ l)
Base.:(*)(m::RevBiLengthMask, l::Integer) = RevBiLengthMask(m.q_len .* l, m.k_len .* l)
Base.:(-)(m::RevBiLengthMask, l::Integer) = RevBiLengthMask(m.q_len .- l, m.k_len .- l)
Base.:(+)(l::Integer, m::RevBiLengthMask) = m + l
Base.:(*)(l::Integer, m::RevBiLengthMask) = m * l
