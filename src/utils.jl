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

struct StaticChar{C} <: AbstractChar
    StaticChar{C}() where {C} = new{C::Char}()
end

Base.show(io::IO, ::StaticChar{C}) where {C} = print(io, "static($C)")

StaticChar(C::Char) = StaticChar{C}()
StaticChar(C::AbstractChar) = StaticChar(convert(Char, C))
StaticChar(::StaticChar{C}) where {C} = StaticChar{C}()
StaticChar(::Val{C}) where {C} = StaticChar{C}()

Base.convert(::Type{T}, ::StaticChar{C}) where {T<:AbstractChar, C} = convert(T, C)
(::Type{T})(::StaticChar{C}) where {T<:AbstractChar, C} = T(C)
(::Type{T})(x::Char) where {T<:StaticChar} = StaticChar(x)

Base.convert(::Type{StaticChar{C}}, ::StaticChar{C}) where {C} = StaticChar{C}()

Base.codepoint(::StaticChar{C}) where {C} = static(codepoint(C))

@inline Base.:(==)(::StaticChar{C}, ::StaticChar{S}) where {C, S} = C == S
@inline Base.:(==)(::StaticChar{C}, c::AbstractChar) where {C} = C == c
@inline Base.:(==)(c::AbstractChar, ::StaticChar{C}) where {C} = C == c

# Base.promote_rule(::Type{T}, ::Type{<:StaticChar}) where {T >: Nothing} = promote_type(T, Char)
# Base.promote_rule(::Type{Nothing}, ::Type{<:StaticChar}) = Union{Nothing, Char}
# Base.promote_rule(::Type{<:StaticChar}, ::Type{T}) where {T <: AbstractChar} = promote_type(T, Char)

Static.static(c::Char) = StaticChar(c)
Static.is_static(::Type{T}) where {T<:StaticChar} = True()
Static.known(::Type{StaticChar{C}}) where {C} = C::Char

as_char(c::Char) = c
as_char(c::StaticChar) = Char(c)
as_char(c) = c
