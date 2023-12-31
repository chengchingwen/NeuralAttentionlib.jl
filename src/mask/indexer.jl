abstract type AbstractIndexer{N} <: AbstractArray{Bool, N} end

struct Indexer{T <: AbstractMask, N, D<:Base.Dims{N}, Ns<:NamedTuple} <: AbstractIndexer{N}
    __fields::Ns
    dest_size::D
    function Indexer{T}(x::NamedTuple, dest_size::Base.Dims) where T
        N = length(dest_size)
        return new{T, N, typeof(dest_size), typeof(x)}(x, dest_size)
    end
end

function Indexer(m::AbstractMask, dest_size::Base.Dims)
    if @generated
        ex = Expr(:tuple)
        for i = 1:fieldcount(m)
            fn = fieldname(m, i)
            ft = fieldtype(m, i)
            expr = :(getfield(m, $(QuoteNode(fn))))
            if ft <: AbstractMask
                expr = :(Indexer($expr, dest_size))
            elseif ft <: Tuple{Vararg{AbstractMask}}
                expr = :(map(Base.Fix2(Indexer, dest_size), $expr))
            end
            push!(ex.args, Expr(:(=), fn, expr))
        end
        if isempty(ex.args)
            ex = :(NamedTuple())
        end
        T = Base.typename(m).wrapper
        ret = quote
            vs = $ex
            return Indexer{$T}(vs, dest_size)
        end
        return ret
    else
        T = typeof(m)
        vs = NamedTuple{fieldnames(T)}(
            ntuple(fieldcount(T)) do i
              v = getfield(m, i)
              if v isa AbstractMask
                v = Indexer(v, dest_size)
              elseif v isa Tuple{Vararg{AbstractMask}}
                v = map(Base.Fix2(Indexer, dest_size), v)
              end
              return v
            end
        )
        return Indexer{Base.typename(T).wrapper}(vs, dest_size)
    end
end

Base.length(I::Indexer) = prod(size(I))
Base.size(I::Indexer) = getfield(I, :dest_size)

IndexedType(::Indexer{T}) where T = T

set_dest_size(x, dest_size::Base.Dims) = x
set_dest_size(t::Tuple{Vararg{Indexer}}, dest_size::Base.Dims) = map(Base.Fix2(set_dest_size, dest_size), t)
set_dest_size(I::Indexer, dest_size::Base.Dims) =
    Indexer{IndexedType(I)}(map(Base.Fix2(set_dest_size, dest_size), getfield(I, :__fields)), dest_size)

function Base.getproperty(I::Indexer, x::Symbol)
    fs = getfield(I, :__fields)
    haskey(fs, x) && return fs[x]
    x == :__fields && return fs
    x == :dest_size && return getfield(I, :dest_size)
    error("type Indexer{$(IndexedType(I))} has no field $x")
end

const MaskIndexer = Indexer{<:AbstractMask}
Base.@propagate_inbounds Broadcast.newindex(arg::MaskIndexer, I::CartesianIndex) = I
Base.@propagate_inbounds Broadcast.newindex(arg::MaskIndexer, I::Integer) = I
Base.eltype(::MaskIndexer) = Bool
Base.@propagate_inbounds Base.getindex(m::MaskIndexer, i::CartesianIndex) = m[Tuple(i)]
Base.@propagate_inbounds Base.getindex(m::MaskIndexer, I::Tuple) = m[I...]

function GetIndexer(m::AbstractMask, dest_size::Base.Dims)
    check_constraint(AxesConstraint(m), dest_size)
    return Indexer(m, dest_size)
end


using Adapt
import Adapt: adapt_structure
adapt_structure(to, x::Indexer) = Indexer{IndexedType(x)}(adapt(to, getfield(x, :__fields)), getfield(x, :dest_size))
Base.print_array(io::IO, I::Indexer) = invoke(Base.print_array, Tuple{IO, AbstractArray}, io, Adapt.adapt(Array, I))
