abstract type AbstractIndexer end

struct Indexer{T, D<:Union{Nothing, NTuple{2, Int}}, Ns<:NamedTuple} <: AbstractIndexer
    __fields::Ns
    dest_size::D
end

Indexer{T}(x::NamedTuple, dest_size = nothing) where T = Indexer{T, typeof(dest_size), typeof(x)}(x, dest_size)

IndexedType(::Indexer{T}) where T = T

function Base.getproperty(I::Indexer, x::Symbol)
    fs = getfield(I, :__fields)
    haskey(fs, x) && return fs[x]
    x == :__fields && return fs
    x == :dest_size && return getfield(I, :dest_size)
    error("type Indexer{$(IndexedType(I))} has no field $x")
end

function GetIndexer(x, dest_size = nothing)
    if @generated
        fs = fieldnames(x)
        ex = Expr(:tuple)
        for i = 1:fieldcount(x)
            push!(ex.args,
                  Expr(:(=), fs[i], :(getfield(x, $i))))
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

adapt_structure(to, x::Indexer) = Indexer{IndexedType(x)}(adapt(to, x.__fields), getfield(x, :dest_size))

function Base.show(io::IO, I::Indexer)
    print(io, "Indexer{", IndexedType(I), '}')
    show(io, I.__fields)
    return io
end
