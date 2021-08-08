using Static

as_bool(b::Bool) = b
as_bool(b::StaticBool) = Bool(b)

function stable_tuple_prod(@nospecialize(x::Tuple))
    @assert length(x) > 0
    x1 = first(x)
    Base.@simd for xi in Base.tail(x)
        x1 *= xi
    end
    return x1
end

as_char(c::Char) = c
as_char(c::StaticInt) = Char(c)
