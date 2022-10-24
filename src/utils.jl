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
($)(f::PrefixedFunction, g::PrefixedFunction) = PrefixedFunction(f.f, (f.arg..., g.f, g.arg...))

function first_n(s::Tuple, n)
    ntuple(i->s[i], n)
end

function last_n(s::Tuple, n)
    offset = static(length(s)) - n
    ntuple(i->s[offset + i], n)
end

_pf_arg_pullback(::Tuple{Vararg{NoTangent}}) = NoTangent()
_pf_arg_pullback(∂args) = ∂args
_pf_pullback(∂f::NoTangent, ∂args::NoTangent) = NoTangent()
_pf_pullback(∂f, ∂args) = (f = ∂f, arg = ∂args)
function _pf_pullback(∂pf::Tuple)
    ∂f = first(∂pf)
    ∂args = _pf_arg_pullback(Base.tail(∂pf))
    return _pf_pullback(∂f, ∂args)
end

function ChainRulesCore.rrule(config::RuleConfig, pf::PrefixedFunction, args...)
    f_tape = rrule(config, pf.f, pf.arg..., args...)
    isnothing(f_tape) && (f_tape = rrule_via_ad(config, pf.f, pf.arg..., args...))
    y, back = f_tape
    num_input = static(length(args))
    num_f_args = static(length(pf.arg)) + static(1)
    function pf_pullback(Ȳ)
        ∂args = back(Ȳ)
        return (_pf_pullback(first_n(∂args, num_f_args)), last_n(∂args, num_input)...)
    end
    return y, pf_pullback
end
