####################  Wrapper Mask  ####################

struct FlipMask{M} <: AbstractAttenMask
    mask::M
end

Base.:!(m::AbstractAttenMask) = FlipMask(m)
Base.:!(m::FlipMask) = m.mask

Adapt.adapt(to::CUDA.Adaptor, m::FlipMask) = Indexer{typeof(m)}((mask = adapt(to, m.mask),))

adapt_structure(to, x::FlipMask) = FlipMask(adapt(to, x.mask))
GetIndexer(m::FlipMask) = Indexer{typeof(m)}((mask = GetIndexer(m.mask),))

Base.@propagate_inbounds getmask_at(m::Indexer{<:FlipMask}, I::Tuple) = !getmask_at(m.mask, I)

Base.show(io::IO, m::FlipMask) = (print(io, '!'); show(io, m.mask); io)

struct CombinedMask{C, Ts<:Tuple} <: AbstractAttenMask
    f::C
    masks::Ts
end

Base.:|(m1::AbstractAttenMask, m2::AbstractAttenMask) = CombinedMask(|, (m1, m2))
Base.:|(m1::CombinedMask{typeof(|)}, m2::AbstractAttenMask) = CombinedMask(|, (m1.masks..., m2))
Base.:|(m1::AbstractAttenMask, m2::CombinedMask{typeof(|)}) = CombinedMask(|, (m1, m2.masks...))
Base.:|(m1::CombinedMask{typeof(|)}, m2::CombinedMask{typeof(|)}) = CombinedMask(|, (m1.masks..., m2.masks...))
Base.:&(m1::AbstractAttenMask, m2::AbstractAttenMask) = CombinedMask(&, (m1, m2))
Base.:&(m1::CombinedMask{typeof(&)}, m2::AbstractAttenMask) = CombinedMask(&, (m1.masks..., m2))
Base.:&(m1::AbstractAttenMask, m2::CombinedMask{typeof(&)}) = CombinedMask(&, (m1, m2.masks...))
Base.:&(m1::CombinedMask{typeof(&)}, m2::CombinedMask{typeof(&)}) = CombinedMask(&, (m1.masks..., m2.masks...))

Adapt.adapt(to::CUDA.Adaptor, m::CombinedMask) = Indexer{typeof(m)}((f = adapt(to, m.f),
                                                                     masks = map(Base.Fix1(adapt, to), m.masks)))
adapt_structure(to, x::CombinedMask) = CombinedMask(x.f, adapt(to, x.masks))
GetIndexer(m::CombinedMask) = Indexer{typeof(m)}((m.f, masks = map(GetIndexer, m.masks)))

@inline function _combine_getmask(f, masks, I)
    if length(masks) == 2
        m1 = getmask_at(masks[1], I)
        m2 = getmask_at(masks[2], I)
        return f(m1, m2)
    else
        m1 = getmask_at(masks[1], I)
        m2 = _combine_getmask(f, Base.tail(masks), I)
        return f(m1, m2)
    end
end

Base.@propagate_inbounds function getmask_at(m::Indexer{M}, I::Tuple) where M <: CombinedMask
    return _combine_getmask(m.f, m.masks, I)
end

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

struct BatchedMask{M<:AbstractArrayMask, S<:StaticInt, B<:StaticBool} <: AbstractAttenMask
    mask::M
    batch_dim::S
    neg::B
end

function BatchedMask(mask, batch_dim::Integer)
    @assert batch_dim > 2 || batch_dim < 0
    if batch_dim < 0
        return BatchedMask(mask, static(-batch_dim), static(true))
    else
        return BatchedMask(mask, static(batch_dim), static(false))
    end
end

Adapt.adapt(to::CUDA.Adaptor, m::BatchedMask) = Indexer{typeof(m)}((mask = adapt(to, m.mask), batch_dim = m.batch_dim, neg = m.neg))

adapt_structure(to, x::BatchedMask) = BatchedMask(adapt(to, x.mask), x.batch_dim, x.neg)

GetIndexer(m::BatchedMask) = Indexer{typeof(m)}((mask = GetIndexer(m.mask), batch_dim = m.batch_dim, neg = m.neg))

@inline function _tailtuples(::True, I, dim)
    offset = length(I) - dim
    return ntuple(i->I[i+offset], dim)
end

@inline function _tailtuples(::False, I, _dim)
    dim = _dim - 1
    return ntuple(i->I[i+dim], length(I)-dim)
end

Base.@propagate_inbounds function getmask_at(m::Indexer{M}, I::Tuple) where M <: BatchedMask
    i = I[1]
    j = I[2]
    J = _tailtuples(m.neg, I, m.batch_dim)
    return getmask_at(m.mask, (i, j, J...))
end

struct RepeatMask{M<:AbstractArrayMask} <: AbstractAttenMask
    mask::M
    num::Int
end

Adapt.adapt(to::CUDA.Adaptor, m::RepeatMask) = Indexer{typeof(m)}((mask = adapt(to, m.mask), num = m.num))

adapt_structure(to, x::RepeatMask) = RepeatMask(adapt(to, x.mask), x.num)

GetIndexer(m::RepeatMask) = Indexer{typeof(m)}((mask = GetIndexer(m.mask), num = m.num))

Base.@propagate_inbounds function getmask_at(m::Indexer{M}, I::Tuple) where M <: RepeatMask
    dim = length(I)
    b = I[dim]
    J = Base.setindex(I,  fld1(b, m.num), dim)
    return getmask_at(m.mask, J)
end
