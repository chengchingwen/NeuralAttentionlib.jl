as_bool(b::Bool) = b
as_bool(b::StaticBool) = Bool(b)

as_char(c::Char) = c
as_char(c::StaticInt) = Char(c)

struct PrefixedFunction{F, A<:Tuple} <: Function
    f::F
    arg::A
end

function Base.show(io::IO, f::PrefixedFunction)
    show(io, f.f)
    show(io, f.arg)
end
Base.show(io::IO, ::MIME"text/plain", f::PrefixedFunction) = show(io, f)

@inline (f::PrefixedFunction)(args...) = f.f(f.arg..., args...)

($)(f::Function, x) = PrefixedFunction(f, (x,))
($)(f::PrefixedFunction, x) = PrefixedFunction(f.f, (f.arg..., x))
