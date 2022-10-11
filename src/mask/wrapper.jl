####################  Wrapper Mask  ####################

struct FlipMask{M} <: AbstractWrapperMask
    mask::M
end

AttenMask(m::FlipMask) = FlipMask(AttenMask(m.mask))

Base.:!(m::AbstractMask) = FlipMask(m)
Base.:!(m::FlipMask) = m.mask

Adapt.adapt(to::CUDA.Adaptor, m::FlipMask) = Indexer{typeof(m)}((mask = adapt(to, m.mask),))

adapt_structure(to, x::FlipMask) = FlipMask(adapt(to, x.mask))
GetIndexer(m::FlipMask, dest_size = nothing) = Indexer{typeof(m)}((mask = GetIndexer(m.mask, dest_size),), dest_size)

Base.@propagate_inbounds Base.getindex(m::Indexer{<:FlipMask}, I::Tuple) = !m.mask[I]
Base.@propagate_inbounds Base.getindex(m::Indexer{<:FlipMask}, I::Integer...) = !m.mask[I...]

check_constraint(m::FlipMask, x) = check_constraint(m.mask, x)

AxesConstraint(m::FlipMask) = AxesConstraint(m.mask)
randomness(m::FlipMask) = randomness(m.mask)
require_dest(m::FlipMask) = require_dest(m.mask)

Base.show(io::IO, m::FlipMask) = (print(io, '!'); show(io, m.mask); io)

struct CombinedMask{C, Ts<:Tuple} <: AbstractWrapperMask
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

Base.:|(m::AbstractMask, ::Nothing) = m
Base.:|(::Nothing, m::AbstractMask) = m
Base.:&(m::AbstractMask, ::Nothing) = m
Base.:&(::Nothing, m::AbstractMask) = m

Adapt.adapt(to::CUDA.Adaptor, m::CombinedMask) = Indexer{typeof(m)}((f = adapt(to, m.f),
                                                                     masks = map(Base.Fix1(adapt, to), m.masks)))
adapt_structure(to, x::CombinedMask) = CombinedMask(x.f, adapt(to, x.masks))
GetIndexer(m::CombinedMask, dest_size = nothing) = Indexer{typeof(m)}((m.f, masks = map(Base.Fix2(GetIndexer, dest_size), m.masks)))

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

check_constraint(m::CombinedMask, x) = check_constraint(m.masks, x)

function AxesConstraint(m::CombinedMask)
    merge_constraint(map(AxesConstraint, m.masks)...)
end

randomness(m::CombinedMask) = static(any(map(randomness, m.masks)))
require_dest(m::CombinedMask) = static(any(map(require_dest, m.masks)))

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

struct BatchedMask{M, S<:StaticInt} <: AbstractWrapperMask
    mask::M
    batch_dim::S
end

AttenMask(b::BatchedMask) = BatchedMask(AttenMask(b.mask), b.batch_dim)

compute_batch_dim(::Tuple{}) = 0
compute_batch_dim(cs::Tuple{NDimConstraint}) = 0
compute_batch_dim(cs::Tuple{NDimConstraint, All1Constraint}) = 0
compute_batch_dim(cs::Tuple{NDimConstraint, Vararg{DimConstraint}}) = count(c->!c.fixed, Base.tail(cs))

BatchedMask(mask::BatchedMask) = mask
function BatchedMask(mask)
    batch_dim = compute_batch_dim(AxesConstraint(mask))
    return BatchedMask(mask, static(batch_dim))
end

Adapt.adapt(to::CUDA.Adaptor, m::BatchedMask) = Indexer{typeof(m)}((mask = adapt(to, m.mask), batch_dim = m.batch_dim))

adapt_structure(to, x::BatchedMask) = BatchedMask(adapt(to, x.mask), x.batch_dim)

GetIndexer(m::BatchedMask, dest_size = nothing) = Indexer{typeof(m)}((mask = GetIndexer(m.mask, dest_size), batch_dim = m.batch_dim))

@inline function _tailtuples(I, dim)
    offset = static(length(I)) - dim
    return ntuple(i->I[i+offset], dim)
end

Base.getindex(m::Indexer{M}, I::Integer...) where M <: BatchedMask = m[I]
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
require_dest(m::BatchedMask) = require_dest(m.mask)

struct RepeatMask{M} <: AbstractWrapperMask
    mask::M
    num::Int
end

AttenMask(r::RepeatMask) = RepeatMask(AttenMask(r.mask), r.num)

Adapt.adapt(to::CUDA.Adaptor, m::RepeatMask) = Indexer{typeof(m)}((mask = adapt(to, m.mask), num = m.num))

adapt_structure(to, x::RepeatMask) = RepeatMask(adapt(to, x.mask), x.num)

GetIndexer(m::RepeatMask, dest_size = nothing) = Indexer{typeof(m)}((mask = GetIndexer(m.mask, dest_size), num = m.num))

Base.@propagate_inbounds function Base.getindex(m::Indexer{M}, I::Integer...) where M <: RepeatMask
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
require_dest(m::RepeatMask) = require_dest(m.mask)
