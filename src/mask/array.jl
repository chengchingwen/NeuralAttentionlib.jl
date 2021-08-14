####################  Array Mask  ####################

struct GenericMask{N, M <:AbstractArray{Bool, N}} <: AbstractArrayMask
    mask::M
end

GenericMask(mask) = GenericMask(convert(AbstractArray{Bool}, mask))

adapt_structure(to, x::GenericMask) = GenericMask(adapt(to, x.mask))

Base.@propagate_inbounds Base.getindex(m::Indexer{<:GenericMask}, I::Integer...) = m.mask[I...]

Base.axes(m::GenericMask{N}) where N = MaskAxes{N}(axes(m.mask))

struct SymLengthMask{N, L <: AbstractArray{Int32, N}} <: AbstractArrayMask
    len::L
end

SymLengthMask(len) = SymLengthMask(convert(AbstractArray{Int32}, len))

adapt_structure(to, x::SymLengthMask) = SymLengthMask(adapt(to, x.len))

Base.@propagate_inbounds Base.getindex(m::Indexer{<:SymLengthMask}, i, j) = (l = m.len[]; i <= l && j <= l)
Base.@propagate_inbounds function Base.getindex(m::Indexer{<:SymLengthMask{N}}, i, j, J::Vararg{Integer,N}) where N
    l = m.len[J...]
    return i <= l && j <= l
end

Base.axes(m::SymLengthMask{N}) where N = MaskAxes{N+2}(axes(m.len))


struct BiLengthMask{N, L <: AbstractArray{Int32, N}} <: AbstractArrayMask
    q_len::L
    k_len::L
end

function BiLengthMask(q_len, k_len)
    @assert axes(q_len) == axes(k_len)
    return BiLengthMask(convert(AbstractArray{Int32}, q_len), convert(AbstractArray{Int32}, k_len))
end

adapt_structure(to, x::BiLengthMask) = BiLengthMask(adapt(to, x.q_len), adapt(to, x.k_len))

Base.@propagate_inbounds Base.getindex(m::Indexer{<:BiLengthMask}, i, j) = i <= m.k_len[] && j <= m.q_len[]
Base.@propagate_inbounds function Base.getindex(m::Indexer{<:BiLengthMask{N}}, i, j, J::Vararg{Integer,N}) where N
    ql = m.q_len[J...]
    kl = m.k_len[J...]
    return i <= kl && j <= ql
end

Base.axes(m::BiLengthMask{N}) where N = MaskAxes{N+2}(axes(m.q_len))
