using ChainRulesCore

abstract type AbstractAttenOp end
abstract type AbstractAttenScoreOp end
abstract type AbstractMixingOp end

get_attention_func(::AbstractAttenOp) = error("`get_attention_func` must be overloaded.")
get_attention_func_args(::AbstractAttenOp, args...) = error("`get_attention_func_args` must be overloaded.")
(op::AbstractAttenOp)(args...) = get_attention_func(op)(get_attention_func_args(op, args...)...)

struct AttenOpPullback{B, F, A}
    f_pullback::B
    get_func_pullback::F
    get_arg_pullback::A
end
function (pb::AttenOpPullback)(Ȳ)
    ∂f, ∂op_args... = pb.f_pullback(Ȳ)
    _, ∂op1, ∂args... = pb.get_arg_pullback(∂op_args)
    _, ∂op2 = pb.get_func_pullback(∂f)
    ∂op = ∂op1 + ∂op2
    return (∂op, ∂args...)
end

function ChainRulesCore.rrule(config::RuleConfig, op::AbstractAttenOp, args...)
    get_func_tape = rrule(config, get_attention_func, op)
    isnothing(get_func_tape) && (get_func_tape = rrule_via_ad(config, get_attention_func, op))
    func, get_func_pullback = get_func_tape
    get_arg_tape = rrule(config, get_attention_func_args, op, args...)
    isnothing(get_arg_tape) && (get_arg_tape = rrule_via_ad(config, get_attention_func_args, op, args...))
    op_args, get_arg_pullback = get_arg_tape
    f_tape = rrule(config, func, op_args...)
    isnothing(f_tape) && (f_tape = rrule_via_ad(config, func, op_args...))
    y, f_pullback = f_tape
    return y, AttenOpPullback(f_pullback, get_func_pullback, get_arg_pullback)
end

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
        head::Int  # number of head
        p::F       # dropout probability
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
        head::Int  # number of head
        p::F       # dropout probability
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

"""
    struct GroupedQueryAttenOp{F} <: AbstractAttenOp
        head::Int
        group::Int
        p::F
    end

Structure for holding parameter of `grouped_query_attention`.

    (op::GroupedQueryAttenOp)(q, k, v, mask = nothing)

Perform grouped query attention.

"""
struct GroupedQueryAttenOp{F} <: AbstractAttenOp
    head::Int
    group::Int
    p::F
end
GroupedQueryAttenOp(head, group) = GroupedQueryAttenOp(head, group, nothing)
get_attention_func(::GroupedQueryAttenOp) = grouped_query_attention
get_attention_func_args(op::GroupedQueryAttenOp, q, k, v, mask = nothing) =
    (op.head, op.group, q, k, v, BatchedMask(mask), op.p)

"Same as [`GroupedQueryAttenOp`](@ref) but also return the attention score"
const GroupedQueryAttenOpWithScore{F} = WithScore{GroupedQueryAttenOp{F}}

"""
    struct CausalGroupedQueryAttenOp{F} <: AbstractAttenOp
        head::Int
        group::Int
        p::F
    end

Structure for holding parameter of `grouped_query_attention`.

    (op::CausalGroupedQueryAttenOp)(q, k, v, mask = nothing)

Perform grouped query attention where `mask` would be combined with a [`CausalMask`](@ref).

"""
struct CausalGroupedQueryAttenOp{F} <: AbstractAttenOp
    head::Int
    group::Int
    p::F
end
CausalGroupedQueryAttenOp(head, group) = CausalGroupedQueryAttenOp(head, group, nothing)
get_attention_func(::CausalGroupedQueryAttenOp) = grouped_query_attention
get_attention_func_args(op::CausalGroupedQueryAttenOp, q, k, v, mask = nothing) =
    (op.head, op.group, q, k, v, BatchedMask(CausalMask() & mask), op.p)

"Same as [`CausalGroupedQueryAttenOp`](@ref) but also return the attention score"
const CausalGroupedQueryAttenOpWithScore{F} = WithScore{CausalGroupedQueryAttenOp{F}}
