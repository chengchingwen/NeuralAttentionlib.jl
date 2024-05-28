abstract type AbstractIndexer{N} <: AbstractArray{Bool, N} end

struct Indexer{M <: AbstractMask, N} <: AbstractIndexer{N}
    mask::M
    destsize::Dims{N}
    function Indexer(mask::AbstractMask, destsize::Dims{N}) where N
        m = adapt(Indexer, mask)
        return new{typeof(m), N}(m, destsize)
    end
end
function GetIndexer(mask::AbstractMask, destsize::Dims)
    check_constraint(AxesConstraint(mask), destsize)
    return Indexer(mask, destsize)
end
Base.length(I::Indexer) = prod(size(I))
Base.size(I::Indexer) = getfield(I, :destsize)
Base.eltype(::Indexer) = Bool

@inline Base.@propagate_inbounds Base.getindex(m::Indexer, I::Integer...) = __maskgetindex__(m.destsize, m.mask, I...)
@inline Base.@propagate_inbounds Base.getindex(m::Indexer, I::Tuple) = __maskgetindex__(m.destsize, m.mask, I...)

using Adapt
import Adapt: adapt_structure
adapt_structure(to, m::Indexer) = Indexer(adapt(to, m.mask), m.destsize)
Base.print_array(io::IO, m::Indexer) = invoke(Base.print_array, Tuple{IO, AbstractArray{Bool, ndims(m)}}, io, Adapt.adapt(Array, m))

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
