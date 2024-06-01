# using `adapt` as a free walk interface
using Random
struct IndexerAdaptor{R<:Union{Nothing, AbstractRNG}}; rng::R; end
IndexerAdaptor() = IndexerAdaptor(Random.default_rng())

abstract type AbstractIndexer{T, N} <: AbstractArray{T, N} end
struct Indexer{T, M <: AbstractMask, N} <: AbstractIndexer{T, N}
    scale::T
    mask::M
    destsize::Dims{N}
end
Indexer(mask::AbstractMask, destsize::Dims, scale = true) = Indexer(IndexerAdaptor(), mask, destsize, scale)
function Indexer(to::IndexerAdaptor, mask::AbstractMask, destsize::Dims{N}, scale = true) where N
    m = adapt(to, mask)
    return Indexer{typeof(scale), typeof(m), N}(scale, m, destsize)
end
GetIndexer(mask::AbstractMask, destsize::Dims, scale = true) = GetIndexer(IndexerAdaptor(), mask, destsize, scale)
function GetIndexer(to::IndexerAdaptor, mask::AbstractMask, destsize::Dims, scale = true)
    check_constraint(AxesConstraint(mask), destsize)
    return Indexer(to, mask, destsize, scale)
end
Base.length(I::Indexer) = prod(size(I))
Base.size(I::Indexer) = I.destsize

@inline Base.@propagate_inbounds Base.getindex(m::Indexer{Bool}, I::Integer...) = __maskgetindex__(m.destsize, m.mask, I...)
@inline Base.@propagate_inbounds Base.getindex(m::Indexer{Bool}, I::Tuple) = __maskgetindex__(m.destsize, m.mask, I...)
@inline Base.@propagate_inbounds Base.getindex(m::Indexer, I::Integer...) = m.scale * __maskgetindex__(m.destsize, m.mask, I...)
@inline Base.@propagate_inbounds Base.getindex(m::Indexer, I::Tuple) = m.scale * __maskgetindex__(m.destsize, m.mask, I...)

using Adapt
import Adapt: adapt_structure
adapt_structure(to, m::Indexer) = (mask = adapt(to, m.mask); Indexer{eltype(m), typeof(mask), ndims(m)}(m.scale, mask, m.destsize))
Base.print_array(io::IO, m::Indexer) = invoke(Base.print_array, Tuple{IO, AbstractArray{eltype(m), ndims(m)}}, io, Adapt.adapt(Array, m))

using FuncTransforms: FuncTransforms, FuncTransform, FA, VA
function _maskgetindex_generator(world, source, self, destsize, mask, I)
    caller = Core.Compiler.specialize_method(
        FuncTransforms.method_by_ftype(Tuple{self, destsize, mask, I...}, nothing, world))
    sig = Base.to_tuple_type((typeof(maskgetindex), destsize, mask, I...))
    ft = FuncTransform(sig, world, [FA(:maskgetindex, 1), FA(:destsize, 2), FA(:mask, 3), VA(:I, 3)]; caller)
    for (ssavalue, code) in FuncTransforms.FuncInfoIter(ft.fi)
        stmt = code.stmt
        newstmt = FuncTransforms.walk(stmt) do x
            FuncTransforms.resolve(x) isa typeof(maskgetindex) ? FuncTransforms.getparg(ft.fi, 1) : x
        end
        FuncTransforms.inlineflag!(code)
        code.stmt = newstmt
    end
    ci = FuncTransforms.toCodeInfo(ft; inline = true, propagate_inbounds = true)
    return ci
end
@eval function __maskgetindex__(destsize::Dims, mask::AbstractMask, I::Integer...)
    $(Expr(:meta, :generated, _maskgetindex_generator))
    $(Expr(:meta, :generated_only))
end
