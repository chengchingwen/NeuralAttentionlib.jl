as_bool(b::Bool) = b
as_bool(b::StaticBool) = Bool(b)

as_char(c::Char) = c
as_char(c::StaticInt) = Char(c)

"""
    PrefixedFunction(f, args::NTuple{N}) <: Function

A type representating a partially-applied version of the function `f`, with the first `N` arguments fixed to the
 values `args`. In other words, `PrefixedFunction(f, args)` behaves similarly to `(xs...)->f(args..., xs...)`.

See also [`NeuralAttentionlib.:\$`](@ref).
"""
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

"""
    f \$ x
    f \$ x \$ y \$ ...

Partially-applied function. Return a [`PrefixedFunction`](@ref).
"""
($)(f::Function, x) = PrefixedFunction(f, (x,))
($)(f::PrefixedFunction, x) = PrefixedFunction(f.f, (f.arg..., x))
($)(f::PrefixedFunction, g::PrefixedFunction) = PrefixedFunction(f.f, (f.arg..., g.f, g.arg...))

function first_n(s::Tuple, n)
    ntuple(i->s[i], n)
end

function last_n(s::Tuple, n)
    offset = static(length(s)) - n
    ntuple(i->s[offset + i], n)
end

struct PrefixedFunctionPullback{B, I, A}
    back::B
    num_input::I
    num_f_args::A
end
function (pb::PrefixedFunctionPullback)(Ȳ)
    ∂args = pb.back(Ȳ)
    ∂pf = first_n(∂args, pb.num_f_args)
    ∂f = first(∂pf)
    ∂arg = Base.tail(∂pf)
    return (Tangent{PrefixedFunction}(; f = ∂f, arg = ∂arg), last_n(∂args, pb.num_input)...)
end

function ChainRulesCore.rrule(config::RuleConfig, pf::PrefixedFunction, args...)
    f_tape = rrule(config, pf.f, pf.arg..., args...)
    isnothing(f_tape) && (f_tape = rrule_via_ad(config, pf.f, pf.arg..., args...))
    y, back = f_tape
    num_input = static(length(args))
    num_f_args = static(length(pf.arg)) + static(1)
    return y, PrefixedFunctionPullback(back, num_input, num_f_args)
end

shape2stride(shape::Tuple{T, Vararg{T}}) where T = _shape2stride((one(T),), shape)
_shape2stride(stride, shape::NTuple{1}) = stride
_shape2stride(stride, shape::Tuple{T, T, Vararg{T}}) where T =
    _shape2stride((stride..., last(stride) * first(shape)), Base.tail(shape))

# https://github.com/FluxML/NNlib.jl/blob/7369244c1a64317eef5b0a20c142316947a85bb3/src/utils.jl#L131-L141
function _fast_broadcast2!(f::F, dst::Array, x, yz...) where {F<:Function}
    bc = Broadcast.broadcasted(f, x, yz...)
    bc = Broadcast.instantiate(Broadcast.Broadcasted(bc.style, bc.f, bc.args, axes(dst)))
    @simd ivdep for I in eachindex(bc)
        @inbounds dst[I] = bc[I]
    end
    return dst
end
function _fast_broadcast2!(f::F, dst::AbstractArray, x, yz...) where {F<:Function}
    return broadcast!(f, dst, x, yz...)
end
using NNlib: _fast_broadcast!
@inline _fast_broadcast(f, x, yz...) = _fast_broadcast!(f, copy(x), yz...)
@inline _fast_broadcast2(f, x, yz...) = _fast_broadcast2!(f, similar(x), x, yz...)

# avoiding ChainRulesCore #684
using ChainRulesCore
@static if VERSION > v"1.10"
    using Base: replace_linenums!
else
    replace_linenums!(ex, ln::LineNumberNode) = ex
    function replace_linenums!(ex::Expr, ln::LineNumberNode)
        if ex.head === :block || ex.head === :quote
            # replace line number expressions from metadata (not argument literal or inert) position
            map!(ex.args, ex.args) do @nospecialize(x)
                isa(x, Expr) && x.head === :line && length(x.args) == 1 && return Expr(:line, ln.line)
                isa(x, Expr) && x.head === :line && length(x.args) == 2 && return Expr(:line, ln.line, ln.file)
                isa(x, LineNumberNode) && return ln
                return x
            end
        end
        # preserve any linenums inside `esc(...)` guards
        if ex.head !== :escape
            for subex in ex.args
                subex isa Expr && replace_linenums!(subex, ln)
            end
        end
        return ex
    end
end
macro _thunk(body)
    letargs = Base._lift_one_interp!(body)
    func = replace_linenums!(:(()->($(esc(body)))), __source__)
    return quote
        if ChainRulesCore._usethunks()
            let $(letargs...)
                Thunk($func)
            end
        else
            let $(letargs...)
                $(esc(body))
            end
        end
    end
end
