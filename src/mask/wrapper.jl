####################  Wrapper Mask  ####################

struct FlipMask{M} <: AbstractWrapperMask
    mask::M
end

Base.:!(m::AbstractAttenMask) = FlipMask(m)
Base.:!(m::FlipMask) = m.mask

Adapt.adapt(to::CUDA.Adaptor, m::FlipMask) = Indexer{typeof(m)}((mask = adapt(to, m.mask),))

adapt_structure(to, x::FlipMask) = FlipMask(adapt(to, x.mask))
GetIndexer(m::FlipMask) = Indexer{typeof(m)}((mask = GetIndexer(m.mask),))

Base.@propagate_inbounds Base.getindex(m::Indexer{<:FlipMask}, I::Integer...) = !m.mask[I...]

AxesConstrain(m::FlipMask) = AxesConstrain(m.mask)

Base.show(io::IO, m::FlipMask) = (print(io, '!'); show(io, m.mask); io)

struct CombinedMask{C, Ts<:Tuple} <: AbstractWrapperMask
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
        m1 = masks[1][I]
        m2 = masks[2][I]
        return f(m1, m2)
    else
        m1 = masks[1][I]
        m2 = _combine_getmask(f, Base.tail(masks), I)
        return f(m1, m2)
    end
end

Base.@propagate_inbounds Base.getindex(m::Indexer{M}, I::Integer...) where M <: CombinedMask = m[I]
Base.@propagate_inbounds function Base.getindex(m::Indexer{M}, I::Tuple) where M <: CombinedMask
    return _combine_getmask(m.f, m.masks, I)
end

check_constrain(m::CombinedMask, x) = check_constrain(m.masks, x)

function AxesConstrain(m::CombinedMask)
    merge_constrain(map(AxesConstrain, m.masks)...)
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

struct BatchedMask{M, S<:StaticInt} <: AbstractWrapperMask
    mask::M
    batch_dim::S
end

compute_batch_dim(::Tuple{}) = 0
compute_batch_dim(cs::Tuple{NDimConstrain}) = 0
compute_batch_dim(cs::Tuple{NDimConstrain, All1Constrain}) = 0
compute_batch_dim(cs::Tuple{NDimConstrain, Vararg{DimConstrain}}) = length(cs) - 1

function BatchedMask(mask)
    batch_dim = compute_batch_dim(AxesConstrain(mask))
    return BatchedMask(mask, static(batch_dim))
end

Adapt.adapt(to::CUDA.Adaptor, m::BatchedMask) = Indexer{typeof(m)}((mask = adapt(to, m.mask), batch_dim = m.batch_dim))

adapt_structure(to, x::BatchedMask) = BatchedMask(adapt(to, x.mask), x.batch_dim)

GetIndexer(m::BatchedMask) = Indexer{typeof(m)}((mask = GetIndexer(m.mask), batch_dim = m.batch_dim))

@inline function _tailtuples(I, dim)
    offset = length(I) - dim
    return ntuple(i->I[i+offset], dim)
end

Base.@propagate_inbounds function Base.getindex(m::Indexer{M}, I::Integer...) where M <: BatchedMask
    i = I[1]
    j = I[2]
    J = _tailtuples(I, m.batch_dim)
    return m.mask[(i, j, J...)]
end

batch_constrain(::Tuple{}) = ()
batch_constrain(cs::Tuple{NDimConstrain}) = (NDimConstrain(cs[1].n, true),)
batch_constrain(cs::Tuple{NDimConstrain, All1Constrain}) = (NDimConstrain(cs[1].n, true),)
function batch_constrain(cs::Tuple{NDimConstrain, Vararg{DimConstrain}})
    dcs = Base.tail(cs)
    n = length(dcs)
    return (NDimConstrain(cs[1].n, true), ntuple(i->DimConstrain(i-n-1, dcs[i].val), n)...)
end

AxesConstrain(m::BatchedMask) = batch_constrain(AxesConstrain(m.mask))

struct RepeatMask{M} <: AbstractWrapperMask
    mask::M
    num::Int
end

Adapt.adapt(to::CUDA.Adaptor, m::RepeatMask) = Indexer{typeof(m)}((mask = adapt(to, m.mask), num = m.num))

adapt_structure(to, x::RepeatMask) = RepeatMask(adapt(to, x.mask), x.num)

GetIndexer(m::RepeatMask) = Indexer{typeof(m)}((mask = GetIndexer(m.mask), num = m.num))

Base.@propagate_inbounds function Base.getindex(m::Indexer{M}, I::Integer...) where M <: RepeatMask
    dim = length(I)
    b = I[dim]
    J = Base.setindex(I,  fld1(b, m.num), dim)
    return m.mask[J]
end

@inline head(t::Tuple) = Base.reverse(Base.tail(Base.reverse(t)))

AxesConstrain(m::RepeatMask) = multiply_constrain(AxesConstrain(m.mask), m.num)

@inline multiply_constrain(::Tuple{}, _) = ()
@inline multiply_constrain(cs::Tuple{NDimConstrain}, n) = cs
@inline function multiply_constrain(cs::Tuple{NDimConstrain, All1Constrain}, n)
    c = cs[2]
    return (NDimConstrain(c.n), ntuple(c.n - c.from + 1) do i
        dim = i + c.from - 1
        DimConstrain(dim, dim != c.n ? 1 : n)
    end...)
end

@inline function multiply_constrain(cs::Tuple{NDimConstrain, Vararg{DimConstrain}}, n)
    h = head(cs)
    c = cs[end]
    return (h..., DimConstrain(c.dim, c.val * n))
end
