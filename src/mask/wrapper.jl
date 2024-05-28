####################  Wrapper Mask  ####################

struct FlipMask{D, T, M <: AbstractMask{D, T}} <: AbstractWrapperMask{D, T}
    mask::M
end

AttenMask(m::FlipMask) = FlipMask(AttenMask(m.mask))

Base.:!(m::AbstractMask) = FlipMask(m)
Base.:!(m::FlipMask) = m.mask

adapt_structure(to, x::FlipMask) = FlipMask(adapt(to, x.mask))

Base.@propagate_inbounds maskgetindex(destsize::Dims, m::FlipMask, I::Integer...) = !maskgetindex(destsize, m.mask, I...)

AxesConstraint(m::FlipMask) = AxesConstraint(m.mask)
randomness(m::FlipMask) = randomness(m.mask)

Base.show(io::IO, m::FlipMask) = (print(io, '!'); show(io, m.mask); io)

struct CombinedMask{C, D, T, Ts<:Tuple{Vararg{AbstractMask}}} <: AbstractWrapperMask{D, T}
    f::C
    masks::Ts
    function CombinedMask(f, masks::Tuple{Vararg{AbstractMask}})
        return new{typeof(f), combine_maskdatatag(masks...), combine_masktypetag(masks...), typeof(masks)}(f, masks)
    end
end

AttenMask(c::CombinedMask) = CombinedMask(c.f, map(AttenMask, c.masks))

Base.:|(m1::AbstractMask, m2::AbstractMask) = CombinedMask(|, (m1, m2))
Base.:|(m1::CombinedMask{typeof(|)}, m2::AbstractMask) = CombinedMask(|, (m1.masks..., m2))
Base.:|(m1::AbstractMask, m2::CombinedMask{typeof(|)}) = CombinedMask(|, (m1, m2.masks...))
Base.:|(m1::CombinedMask{typeof(|)}, m2::CombinedMask{typeof(|)}) = CombinedMask(|, (m1.masks..., m2.masks...))
Base.:&(m1::AbstractMask, m2::AbstractMask) = CombinedMask(&, (m1, m2))
Base.:&(m1::CombinedMask{typeof(&)}, m2::AbstractMask) = CombinedMask(&, (m1.masks..., m2))
Base.:&(m1::AbstractMask, m2::CombinedMask{typeof(&)}) = CombinedMask(&, (m1, m2.masks...))
Base.:&(m1::CombinedMask{typeof(&)}, m2::CombinedMask{typeof(&)}) = CombinedMask(&, (m1.masks..., m2.masks...))

Base.:|(m::AbstractMask, ::Nothing) = nothing
Base.:|(::Nothing, m::AbstractMask) = nothing
Base.:&(m::AbstractMask, ::Nothing) = m
Base.:&(::Nothing, m::AbstractMask) = m

adapt_structure(to, x::CombinedMask) = CombinedMask(x.f, map(m->adapt(to, m), x.masks))

@generated function maskgetindex(destsize::Dims, m::CombinedMask, I::Integer...)
    n = length(m.parameters[4].parameters)
    calls = [:(maskgetindex(destsize, m.masks[$i], I...)) for i in 1:n]
    expr = Expr(:call, :(m.f), calls[end-1], calls[end])
    for i = n-2:-1:1
        expr = Expr(:call, :(m.f), calls[i], expr)
    end
    return quote
        Base.@_propagate_inbounds_meta
        return $expr
    end
end

function AxesConstraint(m::CombinedMask)
    merge_constraint(map(AxesConstraint, m.masks)...)
end

randomness(m::CombinedMask) = static(any(map(randomness, m.masks)))

function Base.show(io::IO, m::CombinedMask)
    print(io, '(')
    show(io, first(m.masks))
    for mask in Base.tail(m.masks)
        print(io, ' ', m.f, ' ')
        show(io, mask)
    end
    print(io, ')')
    io
end

struct BatchedMask{D, T, I, M <: AbstractMask{D, T}} <: AbstractWrapperMask{D, T}
    mask::M
    batch_dim::I
end

AttenMask(b::BatchedMask) = BatchedMask(AttenMask(b.mask), b.batch_dim)

compute_batch_dim(::Tuple{}) = 0
compute_batch_dim(cs::Tuple{NDimConstraint}) = 0
compute_batch_dim(cs::Tuple{NDimConstraint, All1Constraint}) = 0
compute_batch_dim(cs::Tuple{NDimConstraint, Vararg{DimConstraint}}) = count(c->!c.fixed, Base.tail(cs))

BatchedMask(::Nothing) = nothing
BatchedMask(mask::BatchedMask) = mask
function BatchedMask(mask)
    batch_dim = compute_batch_dim(AxesConstraint(mask))
    return BatchedMask(mask, batch_dim)
end

adapt_structure(to, x::BatchedMask) = BatchedMask(adapt(to, x.mask), x.batch_dim)
adapt_structure(::Type{Indexer}, x::BatchedMask) = BatchedMask(adapt(Indexer, x.mask), static(x.batch_dim))

@inline function _tailtuples(I, dim)
    offset = static(length(I)) - dim
    return ntuple(i->I[i+offset], dim)
end

Base.@propagate_inbounds function maskgetindex(destsize::Dims, m::BatchedMask, I::Integer...)
    i = I[1]
    j = I[2]
    J = _tailtuples(I, m.batch_dim)
    return maskgetindex(destsize, m.mask, i, j, J...)
end

batch_constraint(::Tuple{}) = ()
batch_constraint(cs::Tuple{NDimConstraint}) = (NDimConstraint(cs[1].n, true),)
batch_constraint(cs::Tuple{NDimConstraint, All1Constraint}) = (NDimConstraint(cs[1].n, true),)
function batch_constraint(cs::Tuple{NDimConstraint, Vararg{DimConstraint}})
    dcs = Base.tail(cs)
    n = length(dcs)
    return (NDimConstraint(cs[1].n, true), ntuple(i-> dcs[i].fixed ? dcs[i] : DimConstraint(i-n-1, dcs[i].val), n)...)
end

AxesConstraint(m::BatchedMask) = batch_constraint(AxesConstraint(m.mask))
randomness(m::BatchedMask) = randomness(m.mask)

struct RepeatMask{D, T, M <: AbstractMask{D, T}} <: AbstractWrapperMask{D, T}
    mask::M
    num::Int
end

AttenMask(r::RepeatMask) = RepeatMask(AttenMask(r.mask), r.num)

adapt_structure(to, x::RepeatMask) = RepeatMask(adapt(to, x.mask), x.num)

Base.@propagate_inbounds function maskgetindex(destsize::Dims, m::RepeatMask, I::Integer...)
    dim = length(I)
    b = I[dim]
    J = Base.setindex(I, fld1(b, m.num), dim)
    return maskgetindex(destsize, m.mask, J...)
end

AxesConstraint(m::RepeatMask) = multiply_constraint(AxesConstraint(m.mask), m.num)

@inline multiply_constraint(::Tuple{}, _) = ()
@inline multiply_constraint(cs::Tuple{NDimConstraint}, n) = cs
@inline function multiply_constraint(cs::Tuple{NDimConstraint, All1Constraint}, n)
    c = cs[2]
    return (NDimConstraint(c.n), ntuple(c.n - c.from + 1) do i
        dim = i + c.from - 1
        DimConstraint(dim, dim != c.n ? 1 : n)
    end...)
end

@inline function multiply_constraint(cs::Tuple{NDimConstraint, Vararg{DimConstraint}}, n)
    h = Base.front(cs)
    c = cs[end]
    return (h..., DimConstraint(c.dim, c.val * n, c.fixed))
end

randomness(m::RepeatMask) = randomness(m.mask)

struct BiSequenceMask{D, QM<:AbstractSeqMask, KM<:AbstractSeqMask} <: AbstractWrapperMask{D, ATTENTION}
    q_mask::QM
    k_mask::KM
    function BiSequenceMask(q_mask::AbstractSeqMask{D1}, k_mask::AbstractSeqMask{D2}) where {D1, D2}
        return new{combine_maskdatatag(D1, D2), typeof(q_mask), typeof(k_mask)}(q_mask, k_mask)
    end
end

adapt_structure(to, x::BiSequenceMask) = BiSequenceMask(adapt(to, x.q_mask), adapt(to, x.k_mask))

function bi_destsize(destsize, is_q)
    if length(destsize) <= 2
        i, j = destsize
        J = ()
    else
        i, j, J... = destsize
    end
    return (1, ifelse(is_q, j, i), J...)
    # return is_q ? (1, j, J...) : (1, i, J...)
end

Base.@propagate_inbounds function maskgetindex(destsize::Dims, m::BiSequenceMask, i::Integer, j::Integer, J::Integer...)
    q = maskgetindex(destsize, m.q_mask, 1, j, J...)
    k = maskgetindex(destsize, m.k_mask, 1, i, J...)
    return q & k
end

seq_cs_transpose(c) = c
seq_cs_transpose(c::DimConstraint) = c.dim == 2 ? DimConstraint(1, c.val, c.fixed) : c
function seq_cs_transpose(cs::Tuple{NDimConstraint, All1Constraint})
    c, c1 = cs
    c1.from == 2 && return (c, DimConstraint(1, 1), All1Constraint(3, c1.n-1))
    return cs
end
function seq_cs_transpose(cs::Tuple{NDimConstraint, Vararg{DimConstraint}})
    c = first(cs)
    dcs = Base.tail(cs)
    dc2 = findfirst(dc->dc.dim == 2, dcs)
    !isnothing(dc2) && return (c, map(dc->dc.dim == 2 ? DimConstraint(1, dc.val, dc.fixed) : dc, dcs)...)
    return cs
end

function AxesConstraint(m::BiSequenceMask)
    qc = AxesConstraint(m.q_mask)
    kc = seq_cs_transpose(AxesConstraint(m.k_mask))
    return merge_constraint(qc, kc)
end

randomness(m::BiSequenceMask) = randomness(m.q_mask) | randomness(m.k_mask)
