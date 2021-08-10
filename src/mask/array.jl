####################  Array Mask  ####################

struct SymLengthMask{N, L <: AbstractArray{Int32, N}} <: AbstractArrayMask
    len::L
end

SymLengthMask(len) = SymLengthMask(convert(AbstractArray{Int32}, len))

adapt_structure(to, x::SymLengthMask) = SymLengthMask(adapt(to, x.len))


# TODO: add boundcheck
Base.@propagate_inbounds function getmask_at(m::Indexer{<:SymLengthMask}, I::Tuple)
    i = I[1]
    j = I[2]
    l = m.len[Base.tail(Base.tail(I))...]
    return i <= l && j <= l
end

struct BiLengthMask{N, L <: AbstractArray{Int32, N}} <: AbstractArrayMask
    q_len::L
    k_len::L
end

BiLengthMask(q_len, k_len) = BiLengthMask(convert(AbstractArray{Int32}, q_len), convert(AbstractArray{Int32}, k_len))

adapt_structure(to, x::BiLengthMask) = BiLengthMask(adapt(to, x.q_len), adapt(to, x.k_len))

Base.@propagate_inbounds function getmask_at(m::Indexer{<:BiLengthMask}, I::Tuple)
    i = I[1]
    j = I[2]
    J = Base.tail(Base.tail(I))
    ql = m.q_len[J...]
    kl = m.k_len[J...]
    return i <= kl && j <= ql
end
