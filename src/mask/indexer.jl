abstract type AbstractIndexer end

struct Indexer{T, Ns<:NamedTuple} <: AbstractIndexer
    __fields::Ns
end

Indexer{T}(x::NamedTuple) where T = Indexer{T, typeof(x)}(x)

IndexedType(::Indexer{T}) where T = T

function Base.getproperty(I::Indexer, x::Symbol)
    fs = getfield(I, :__fields)
    haskey(fs, x) && return fs[x]#getproperty(fs, x)
    x == :__fields && return fs
    error("type Indexer{$(IndexedType(I))} has no field $x")
end

function GetIndexer(x)
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
            return Indexer{$T}(vs)
        end
        return ret
    else
        T = typeof(x)
        vs = ntuple(i->getfield(x, i), fieldcount(T))
        return Indexer{T}(vs)
    end
end

adapt_structure(to, x::Indexer) = Indexer{IndexedType(x)}(adapt(to, x.__fields))


function Base.show(io::IO, I::Indexer)
    print(io, "Indexer{", IndexedType(I), '}')
    show(io, I.__fields)
    return io
end
