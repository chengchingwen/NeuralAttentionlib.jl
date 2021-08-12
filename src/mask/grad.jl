using Base.Broadcast: BroadcastFunction, broadcasted, materialize

ChainRulesCore.@non_differentiable Base.getindex(m::AbstractAttenMask, I::Integer...)
ChainRulesCore.@non_differentiable Base.getindex(m::MaskIndexer, I::Integer...)
ChainRulesCore.@non_differentiable Base.getindex(m::AbstractAttenMask, I::Tuple)
ChainRulesCore.@non_differentiable Base.getindex(m::MaskIndexer, I::Tuple)
ChainRulesCore.@non_differentiable (::Type{<:AbstractAttenMask})(args...)
ChainRulesCore.@non_differentiable (::Type{<:AbstractAttenMaskOp})(args...)
ChainRulesCore.@non_differentiable getmask(arg...)

function ChainRulesCore.rrule(::typeof(apply_mask), op::NaiveAttenMaskOp, mask, score)
    naive_apply_mask_pullback(Ȳ) = (NoTangent(), NoTangent(), NoTangent(), @thunk unthunk(Ȳ) .* mask)
    return score .* mask, naive_apply_mask_pullback
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(apply_broadcast_mask), f, mask, score, scale)
    tmp = getmask(mask, score, scale)
    Y, back = rrule_via_ad(config, broadcast, f, tmp, score)
    function fallback_apply_broadcast_mask_pullback(Ȳ)
        Ȳs = (back∘unthunk)(Ȳ)
        return (NoTangent(), Ȳs[2], NoTangent(), Ȳs[4], NoTangent())
    end
    Y, fallback_apply_broadcast_mask_pullback
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(apply_broadcast_mask), ::typeof(+), mask, score, scale)
    apply_broadcast_mask_pullback(Ȳ) = (NoTangent(), NoTangent(), NoTangent(), Ȳ, NoTangent())
    apply_broadcast_mask(+, mask, score, scale), apply_broadcast_mask_pullback
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(apply_broadcast_mask), ::typeof(*), mask, score, scale)
    function apply_broadcast_mask_pullback(Ȳ)
        thk = @thunk begin
            tmp = unthunk(Ȳ)
            @. tmp * (mask * scale)
        end
        return (NoTangent(), NoTangent(), NoTangent(), thk, NoTangent())
    end
    apply_broadcast_mask(*, mask, score, scale), apply_broadcast_mask_pullback
end

#TODO: use Requires
using Zygote
Zygote.unbroadcast(x::AbstractAttenMask, _) = nothing

