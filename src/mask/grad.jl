using Base.Broadcast: BroadcastFunction, broadcasted, materialize

ChainRulesCore.@non_differentiable Base.getindex(m::Indexer, I...)
ChainRulesCore.@non_differentiable maskgetindex(::Dims, ::AbstractMask, I::Integer...)
ChainRulesCore.@non_differentiable (::Type{<:AbstractMask})(args...)
ChainRulesCore.@non_differentiable (::Type{<:AbstractMaskOp})(args...)
ChainRulesCore.@non_differentiable getmask(arg...)
ChainRulesCore.@non_differentiable AttenMask(m)
ChainRulesCore.@non_differentiable AttenMask(m1, m2)
ChainRulesCore.@non_differentiable lengths(m)

function ChainRulesCore.rrule(::typeof(apply_mask), ::NaiveMaskOp, mask, score)
    m = GetIndexer(mask, size(score))
    naive_apply_mask_pullback(Ȳ) = (NoTangent(), NoTangent(), NoTangent(), _fast_broadcast(*, unthunk(Ȳ), m))
    return _fast_broadcast(*, score, m), naive_apply_mask_pullback
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(apply_mask), op::GenericMaskOp, mask, score)
    scale = convert(eltype(score), op.scale)
    apply = op.apply
    m = as_bool(op.flip) ? !mask : mask
    if apply isa Base.BroadcastFunction
        y, broadcast_pullback = rrule(config, apply_broadcast_mask, apply.f, m, score, scale)
        function broadcast_mask_pullback(Ȳ)
            _, _, _, ∂score, _ = broadcast_pullback(Ȳ)
            return (NoTangent(), NoTangent(), NoTangent(), ∂score)
        end
        return y, broadcast_mask_pullback
    else
        tmp = getmask(m, score, scale)
        apply_tape = rrule(config, apply, score, tmp)
        isnothing(apply_tape) && (apply_tape = rrule_via_ad(config, apply, score, tmp))
        y, apply_pullback = apply_tape
        function mask_pullback(Ȳ)
            _, ∂score, _ = apply_pullback(Ȳ)
            return (NoTangent(), NoTangent(), NoTangent(), ∂score)
        end
        return y, mask_pullback
    end
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(apply_broadcast_mask), f, mask, score, scale)
    tmp = GetIndexer(mask, size(score), convert(eltype(score), scale))
    Y, back = rrule_via_ad(config, broadcast, f, score, tmp)
    function fallback_apply_broadcast_mask_pullback(Ȳ)
        Ȳs = back(Ȳ)
        return (NoTangent(), Ȳs[2], NoTangent(), Ȳs[3], NoTangent())
    end
    Y, fallback_apply_broadcast_mask_pullback
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(apply_broadcast_mask), ::typeof(+), mask, score, scale)
    apply_broadcast_mask_pullback(Ȳ) = (NoTangent(), NoTangent(), NoTangent(), Ȳ, NoTangent())
    return apply_broadcast_mask(+, mask, score, scale), apply_broadcast_mask_pullback
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(apply_broadcast_mask), ::typeof(*), mask, score, scale)
    m = GetIndexer(mask, size(score), convert(eltype(score), scale))
    function apply_broadcast_mask_pullback(Ybar)
        Ȳ = unthunk(Ybar)
        thk = @thunk _fast_broadcast(*, Ȳ, m)
        return (NoTangent(), NoTangent(), NoTangent(), thk, NoTangent())
    end
    return _fast_broadcast(*, score, m), apply_broadcast_mask_pullback
end

function ChainRulesCore.rrule(config::RuleConfig, pf::PrefixedFunction{typeof(apply_mask), <:Tuple{<:AbstractMaskOp}}, m, s)
    y, pf_pullback = rrule(config, pf $ m, s)
    pullback(Ȳ) = (NoTangent(), pf_pullback(Ȳ)[2])
    return y, pullback
end

function ChainRulesCore.rrule(config::RuleConfig, pf::PrefixedFunction{typeof(apply_mask), <:Tuple{<:AbstractMaskOp, <:Union{Nothing, AbstractMask}}}, s)
    op, mask = pf.arg
    y, mask_pullback = rrule(config, apply_mask, op, mask, s)
    pullback(Ȳ) = (NoTangent(), mask_pullback(Ȳ)[4])
    return y, pullback
end
