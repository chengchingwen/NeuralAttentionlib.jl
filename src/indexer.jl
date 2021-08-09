abstract type AbstractIndexer end

struct Indexer{T, Ns<:NamedTuple} <: AbstractIndexer
    __fields::Ns
end

Indexer{T}(x::NamedTuple) where T = Indexer{T, typeof(x)}(x)

IndexedType(::Indexer{T}) where T = T

function Base.getproperty(I::Indexer, x::Symbol)
    fs = getfield(I, :__fields)
    haskey(fs, x) && return getproperty(fs, x)
    x == :__fields && return fs
    error("type Indexer{$(IndexedType(I))} has no field $x")
end

function GetIndexer(x)
    T = typeof(x)
    fs = fieldnames(T)
    vs = map(f->getproperty(x, f), fs)
    Indexer{T}(NamedTuple{fs}(vs))
end

function Base.show(io::IO, I::Indexer)
    print(io, "Indexer{", IndexedType(I), '}')
    show(io, I.__fields)
    io
end
