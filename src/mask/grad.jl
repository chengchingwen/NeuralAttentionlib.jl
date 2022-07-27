using Base.Broadcast: BroadcastFunction, broadcasted, materialize

ChainRulesCore.@non_differentiable Base.getindex(m::AbstractMask, I::Integer...)
ChainRulesCore.@non_differentiable Base.getindex(m::MaskIndexer, I::Integer...)
ChainRulesCore.@non_differentiable Base.getindex(m::AbstractMask, I::Tuple)
ChainRulesCore.@non_differentiable Base.getindex(m::MaskIndexer, I::Tuple)
ChainRulesCore.@non_differentiable (::Type{<:AbstractMask})(args...)
ChainRulesCore.@non_differentiable (::Type{<:AbstractMaskOp})(args...)
ChainRulesCore.@non_differentiable getmask(arg...)

function ChainRulesCore.rrule(::typeof(apply_mask), op::NaiveMaskOp, mask, score)
    m = as_bool(randomness(mask)) ? getmask(mask, score) : mask
    naive_apply_mask_pullback(Ȳ) = (NoTangent(), NoTangent(), NoTangent(), @thunk unthunk(Ȳ) .* m)
    return score .* m, naive_apply_mask_pullback
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
    rnm = as_bool(randomness(mask))
    m = rnm ? getmask(mask, score, scale) : mask
    function apply_broadcast_mask_pullback(Ȳ)
        thk = @thunk begin
            tmp = unthunk(Ȳ)
            rnm ? tmp .* m : @. tmp * (mask * scale)
        end
        return (NoTangent(), NoTangent(), NoTangent(), thk, NoTangent())
    end
    fwd = rnm ? score .* m : apply_broadcast_mask(*, mask, score, scale)
    fwd, apply_broadcast_mask_pullback
end


@init @require Zygote="e88e6eb3-aa80-5325-afca-941959d7151f" begin
    Zygote.unbroadcast(x::AbstractMask, _) = nothing

    function Zygote._pullback(ctx::Zygote.AContext, ::typeof(Broadcast.broadcasted), ::MaskStyle{M}, f, args...) where M
        return y, ∇broadcasted = Zygote._pullback(ctx, Broadcast.broadcasted, M(), f, args...)
    end
end
