abstract type AbstractAttenOp end
abstract type AbstractAttenScoreOp end
abstract type AbstractMixingOp end

get_attention_func(::AbstractAttenOp) = error("`get_attention_func` must be overloaded.")
get_attention_func_args(::AbstractAttenOp, args...) = error("`get_attention_func_args` must be overloaded.")
(op::AbstractAttenOp)(args...) = get_attention_func(op)(get_attention_func_args(op, args...)...)

function Base.show(io::IO, op::AbstractAttenOp)
    T = typeof(op)
    print(io, Base.typename(T).name)
    print(io, '(')
    names = propertynames(op)
    for (i, name) in enumerate(names)
        print(io, name)
        print(io, " = ")
        show(io, getproperty(op, name))
        i != length(names) && print(io, ", ")
    end
    print(io, ')')
end

struct NaiveQKVAttenOp{F} <: AbstractAttenOp
    p::F
end
NaiveQKVAttenOp() = NaiveQKVAttenOp(nothing)
get_attention_func(::NaiveQKVAttenOp) = naive_qkv_attention
get_attention_func_args(op::NaiveQKVAttenOp, q, k, v, mask = nothing) = (q, k, v, mask, op.p)

struct WithScore{A<:AbstractAttenOp} <: AbstractAttenOp
    op::A
end

Base.getproperty(op::WithScore, sym::Symbol) = getproperty(getfield(op, :op), sym)

function (opT::Type{<:WithScore})(args...)
    T = withscore_type(opT)
    return WithScore(T(args...))
end

function withscore_type(T::Type{<:WithScore})
    if isconcretetype(T)
        return T.parameters[1]
    else
        return T.body.parameters[1].name.wrapper
    end
end

get_attention_func(op::WithScore) = get_attention_func(getfield(op, :op))
get_attention_func_args(op::WithScore, args...) =
    (score_returning, get_attention_func_args(getfield(op, :op), args...)...)

function Base.show(io::IO, op::WithScore)
    print(io, "WithScore(")
    show(io, getfield(op, :op))
    print(io, ')')
end

"""
    struct MultiheadQKVAttenOp{F} <: AbstractAttenOp
        head::Int                   # number of head
        p::F  # dropout probability
    end

Structure for holding parameter of `multihead_qkv_attention`.

    (op::MultiheadQKVAttenOp)(q, k, v, mask = nothing)

Perform multihead attention.

"""
struct MultiheadQKVAttenOp{F} <: AbstractAttenOp
    head::Int
    p::F
end
MultiheadQKVAttenOp(head) = MultiheadQKVAttenOp(head, nothing)
get_attention_func(::MultiheadQKVAttenOp) = multihead_qkv_attention
get_attention_func_args(op::MultiheadQKVAttenOp, q, k, v, mask = nothing) =
    (op.head, q, k, v, BatchedMask(mask), op.p)

"Same as [`MultiheadQKVAttenOp`](@ref) but also return the attention score"
const MultiheadQKVAttenOpWithScore{F} = WithScore{MultiheadQKVAttenOp{F}}

"""
    struct CausalMultiheadQKVAttenOp{F} <: AbstractAttenOp
        head::Int                   # number of head
        p::F  # dropout probability
    end

Structure for holding parameter of `multihead_qkv_attention`.

    (op::CausalMultiheadQKVAttenOp)(q, k, v, mask = nothing)

Perform multihead attention where `mask` would be combined with a [`CausalMask`](@ref)
"""
struct CausalMultiheadQKVAttenOp{F} <: AbstractAttenOp
    head::Int
    p::F
end
CausalMultiheadQKVAttenOp(head) = CausalMultiheadQKVAttenOp(head, nothing)
get_attention_func(::CausalMultiheadQKVAttenOp) = multihead_qkv_attention
get_attention_func_args(op::CausalMultiheadQKVAttenOp, q, k, v, mask = nothing) =
    (op.head, q, k, v, BatchedMask(CausalMask() & mask), op.p)

"Same as [`CausalMultiheadQKVAttenOp`](@ref) but also return the attention score"
const CausalMultiheadQKVAttenOpWithScore{F} = WithScore{CausalMultiheadQKVAttenOp{F}}

