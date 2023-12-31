abstract type AbstractIndexer{N} <: AbstractArray{Bool, N} end

struct Indexer{T, N, D<:Base.Dims{N}, Ns<:NamedTuple} <: AbstractIndexer{N}
    __fields::Ns
    dest_size::D
    function Indexer{T}(x::NamedTuple, dest_size::Base.Dims) where T
        N = length(dest_size)
        return new{T, N, typeof(dest_size), typeof(x)}(x, dest_size)
    end
end

Base.length(I::Indexer) = prod(size(I))
Base.size(I::Indexer) = getfield(I, :dest_size)

IndexedType(::Indexer{T}) where T = T

function Base.getproperty(I::Indexer, x::Symbol)
    fs = getfield(I, :__fields)
    haskey(fs, x) && return fs[x]
    x == :__fields && return fs
    x == :dest_size && return getfield(I, :dest_size)
    error("type Indexer{$(IndexedType(I))} has no field $x")
end

function GetIndexer(x, dest_size)
    if @generated
        fs = fieldnames(x)
        ex = Expr(:tuple)
        for i = 1:fieldcount(x)
            push!(ex.args,
                  Expr(:(=), fs[i], :(getfield(x, $i))))
        end
        if isempty(ex.args)
            ex = :(NamedTuple())
        end
        T = x
        ret = quote
            vs = $ex
            return Indexer{$T}(vs, dest_size)
        end
        return ret
    else
        T = typeof(x)
        vs = NamedTuple{fieldnames(T)}(ntuple(i->getfield(x, i), fieldcount(T)))
        vs = merge(vs, extra)
        return Indexer{T}(vs, dest_size)
    end
end

adapt_structure(to, x::Indexer) = Indexer{IndexedType(x)}(adapt(to, getfield(x, :__fields)), getfield(x, :dest_size))

function Base.show(io::IO, I::Indexer)
    print(io, "Indexer{", IndexedType(I), '}')
    show(io, getfield(I, :__fields))
    return io
end
