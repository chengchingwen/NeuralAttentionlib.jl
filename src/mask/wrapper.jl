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

@inline merge_constrain(cs1, cs2, cs...) = merge_constrain(merge_constrain(cs1, cs2), cs...)
@inline merge_constrain(::Tuple{}, ::Tuple{}) = ()
@inline merge_constrain(cs1::Tuple, ::Tuple{}) = cs1
@inline merge_constrain(::Tuple{}, cs2::Tuple) = cs2
@inline function merge_constrain(cs1::Tuple, cs2::Tuple)
    ndc = _merge_ndim_c(cs1[1], cs2[1])
    dc = _merge_dim_c(Base.tails(cs1, cs2)...)
    return (ndc, dc...)
end

_merge_dim_c(::Tuple{},::Tuple{}) = ()
_merge_dim_c(a::Tuple, ::Tuple{}) = a
_merge_dim_c(::Tuple{}, b::Tuple) = b
_merge_dim_c(a::Tuple{All1Constrain}, b::Tuple{All1Constrain}) = All1Constrain(min(a.from, b.from), max(a.n, b.n))

_merge_post_dim_c(::Tuple{}, ::Tuple{}) = ()
_merge_post_dim_c(::Tuple{}, b::Tuple) = b
_merge_post_dim_c(a::Tuple, ::Tuple{}) = a
function _merge_post_dim_c(a::Tuple, b::Tuple)::Tuple{Vararg{DimConstrain}}
    a1 = a[1]
    b1 = b[1]
    if a1.dim < b1.dim
        return (a1, _merge_post_dim_c(Base.tail(a), b)...)
    elseif a1.dim > b1.dim
        return (b1, _merge_post_dim_c(a, Base.tail(b))...)
    else
        a1.val != b1.val && thrdm("mask require $(a1.dim)-th dimension to be both $(a1.val) and $(b1.val)")
        return (a1, _merge_post_dim_c(Base.tails(a, b)...)...)
    end
end

function _merge_dim_c(a::Tuple{DimConstrain, Vararg{DimConstrain}}, b::Tuple{DimConstrain, Vararg{DimConstrain}})
    a1 = a[1]
    b1 = b[1]
    if (a1.dim > 0) == (b1.dim > 0)
        return _merge_post_dim_c(a, b)
    else
        if a1.dim < 0
            n = b[end].dim
            return _merge_post_dim_c(ntuple(i->DimConstrain(a[i].dim+n+1, a[i].val), length(a)), b)
        else
            n = a[end].dim
            return _merge_post_dim_c(a, ntuple(i->DimConstrain(b[i].dim+n+1, b[i].val), length(b)))
        end
    end
end

_merge_dim_c(a::Tuple{DimConstrain, Vararg{DimConstrain}}, b::Tuple{All1Constrain}) = _merge_dim_c(b, a)
function _merge_dim_c(a::Tuple{All1Constrain}, b::Tuple{DimConstrain, Vararg{DimConstrain}})
    c = a[]
    offset = c.from - 1
    return _merge_dim_c(ntuple(i->DimConstrain(i+offset, 1), b[end].dim - offset), b)
end

@inline function _merge_ndim_c(a, b)
    a.least || b.least || thrdm("mask require both ndims(A) == $(a.n) and ndim(A) == $(b.n)")
    n = max(a.n, b.n)
    a.least || a.n == n || thrdm("mask require both ndims(A) == $(a.n) and ndims(A) ≥ $(b.n)")
    b.least || b.n == n || thrdm("mask require both ndims(A) == $(b.n) and ndims(A) ≥ $(a.n)")
    return NDimConstrain(n, a.least == b.least)
end

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
