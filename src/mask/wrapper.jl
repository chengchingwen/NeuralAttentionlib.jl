####################  Wrapper Mask  ####################

struct FlipMask{M} <: AbstractWrapperMask
    mask::M
end

AttenMask(m::FlipMask) = FlipMask(AttenMask(m.mask))

Base.:!(m::AbstractMask) = FlipMask(m)
Base.:!(m::FlipMask) = m.mask

adapt_structure(to, x::FlipMask) = FlipMask(adapt(to, x.mask))

Base.@propagate_inbounds Base.getindex(m::Indexer{<:FlipMask}, I::Tuple) = !m.mask[I]
Base.@propagate_inbounds Base.getindex(m::Indexer{<:FlipMask}, I::Integer...) = !m.mask[I...]

AxesConstraint(m::FlipMask) = AxesConstraint(m.mask)
randomness(m::FlipMask) = randomness(m.mask)

Base.show(io::IO, m::FlipMask) = (print(io, '!'); show(io, m.mask); io)

struct CombinedMask{C, Ts<:Tuple{Vararg{AbstractMask}}} <: AbstractWrapperMask
    f::C
    masks::Ts
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

adapt_structure(to, x::CombinedMask) = CombinedMask(x.f, adapt(to, x.masks))

@inline function _combine_getmask(f, masks, I)
    m1 = first(masks)[I]
    if length(masks) == 2
        m2 = masks[2][I]
        return f(m1, m2)
    else
        m2 = _combine_getmask(f, Base.tail(masks), I)
        return f(m1, m2)
    end
end

Base.@propagate_inbounds Base.getindex(m::Indexer{M}, I::Integer...) where M <: CombinedMask = m[I]
Base.@propagate_inbounds function Base.getindex(m::Indexer{M}, I::Tuple) where M <: CombinedMask
    return _combine_getmask(m.f, m.masks, I)
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

struct BatchedMask{M} <: AbstractWrapperMask
    mask::M
    batch_dim::Int
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
Indexer(m::BatchedMask, dest_size::Base.Dims) = Indexer{BatchedMask}((mask = Indexer(m.mask, dest_size), batch_dim = static(m.batch_dim)), dest_size)

@inline function _tailtuples(I, dim)
    offset = static(length(I)) - dim
    return ntuple(i->I[i+offset], dim)
end

Base.@propagate_inbounds Base.getindex(m::Indexer{M}, I::Integer...) where M <: BatchedMask = m[I]
Base.@propagate_inbounds function Base.getindex(m::Indexer{M}, I::Tuple) where M <: BatchedMask
    i = I[1]
    j = I[2]
    J = _tailtuples(I, m.batch_dim)
    return m.mask[(i, j, J...)]
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

struct RepeatMask{M} <: AbstractWrapperMask
    mask::M
    num::Int
end

AttenMask(r::RepeatMask) = RepeatMask(AttenMask(r.mask), r.num)

adapt_structure(to, x::RepeatMask) = RepeatMask(adapt(to, x.mask), x.num)

Base.@propagate_inbounds Base.getindex(m::Indexer{M}, I::Integer...) where M <: RepeatMask = m[I]
Base.@propagate_inbounds function Base.getindex(m::Indexer{M}, I::Tuple) where M <: RepeatMask
    dim = length(I)
    b = I[dim]
    J = Base.setindex(I, fld1(b, m.num), dim)
    return m.mask[J]
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

struct BiSequenceMask{QM<:AbstractMask, KM<:AbstractMask} <: AbstractWrapperMask
    q_mask::QM
    k_mask::KM
end

adapt_structure(to, x::BiSequenceMask) = BiSequenceMask(adapt(to, x.q_mask), adapt(to, x.k_mask))

bi_dest_size(::Nothing, is_q) = nothing
function bi_dest_size(dest_size, is_q)
    if length(dest_size) <= 2
        i, j = dest_size
        J = ()
    else
        i, j, J... = dest_size
    end
    return is_q ? (1, j, J...) : (1, i, J...)
end
Indexer(m::BiSequenceMask, dest_size::Base.Dims) = Indexer{BiSequenceMask}((
    q_mask = Indexer(m.q_mask, bi_dest_size(dest_size, true)),
    k_mask = Indexer(m.k_mask, bi_dest_size(dest_size, false))), dest_size)

Base.@propagate_inbounds function Base.getindex(m::Indexer{M}, i::Integer, j::Integer, J::Integer...) where M <: BiSequenceMask
    q = m.q_mask[1, j, J...]
    k = m.k_mask[1, i, J...]
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
