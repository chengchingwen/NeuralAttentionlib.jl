using ChainRulesCore

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(as_collapsed), x::AbstractVector)
    y, pullback = rrule(config, CollapsedDimsArray, x, static(0), static(0))
    collapsed_pullback(Ȳ) = (NoTangent(), pullback(Ȳ)[2])
    return y, collapsed_pullback
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(as_collapsed), x::AbstractMatrix)
    y, pullback = rrule(config, CollapsedDimsArray, x, static(1), static(0))
    collapsed_pullback(Ȳ) = (NoTangent(), pullback(Ȳ)[2])
    return y, collapsed_pullback
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(as_collapsed), x::AbstractArray)
    ni = static(ndims(x)) - static(2)
    nj = static(1)
    y, pullback = rrule(config, CollapsedDimsArray, x, collapsed_size(x, ni, nj), ni, nj)
    collapsed_pullback(Ȳ) = (NoTangent(), pullback(Ȳ)[2])
    return y, collapsed_pullback
end

ChainRulesCore.rrule(config::RuleConfig, ::typeof(as_collapsed), x::CollapsedDimsArray) = rrule(config, identity, x)

function ChainRulesCore.rrule(::typeof(split_head), head, x::CollapsedDimsArray)
    proj = ProjectTo(x)
    y = split_head(head, x)
    function split_head_pullback(Ȳ)
        return (NoTangent(), NoTangent(), proj(Ȳ))
    end
    return y, split_head_pullback
end

ChainRulesCore.@non_differentiable move_head_dim_out_perm(x...)

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(move_head_dim_out), x::CollapsedDimsArray)
    perm = move_head_dim_out_perm(x)
    y, back = rrule(config, permutedims, parent(x), perm)
    output_size = size(y)
    ni, nj = x.ni, x.nj
    proj = ProjectTo{CollapsedDimsArray}(; dims = size(parent(x)), ni = ni, nj = nj)
    function move_head_dim_out_pullback(Ybar)
        Ȳ = reshape(unthunk(Ybar), output_size)
        ∂x = proj(back(Ȳ)[2])
        return (NoTangent(), ∂x)
    end
    move_head_dim_out_pullback(::ZeroTangent) = (NoTangent(), ZeroTangent())
    return CollapsedDimsArray(y, x.ni, x.nj + static(1)), move_head_dim_out_pullback
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(move_head_dim_out), x)
    perm = move_head_dim_out_perm(x, static(false))
    y, back = rrule(config, permutedims, x, perm)
    pullback(Ȳ) = (NoTangent(), back(Ȳ)[2])
    pullback(::ZeroTangent) = (NoTangent(), ZeroTangent())
    return y, pullback
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(move_head_dim_out), x, nobatch)
    perm = move_head_dim_out_perm(x, nobatch)
    y, back = rrule(config, permutedims, x, perm)
    pullback(Ȳ) = (NoTangent(), back(Ȳ)[2], NoTangent())
    pullback(::ZeroTangent) = (NoTangent(), ZeroTangent(), NoTangent())
    return y, pullback
end

ChainRulesCore.@non_differentiable move_head_dim_in_perm(x...)

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(move_head_dim_in), x)
    perm = move_head_dim_in_perm(x, static(false))
    y, back = rrule(config, permutedims, x, perm)
    pullback(Ȳ) = (NoTangent(), back(Ȳ)[2])
    pullback(::ZeroTangent) = (NoTangent(), ZeroTangent())
    return y, pullback
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(move_head_dim_in), x, nobatch)
    perm = move_head_dim_in_perm(x, nobatch)
    y, back = rrule(config, permutedims, x, perm)
    pullback(Ȳ) = (NoTangent(), back(Ȳ)[2], NoTangent())
    pullback(::ZeroTangent) = (NoTangent(), ZeroTangent(), NoTangent())
    return y, pullback
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(merge_head), x)
    s = (:, Base.tail(Base.tail(size(x)))...)
    y, back = rrule(config, reshape, x, s)
    pullback(Ȳ) = (NoTangent(), back(Ȳ)[2])
    return y, pullback
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(_split_and_move_head), head, x)
    s, split_back = rrule(config, split_head, head, x)
    y, move_back = rrule(config, move_head_dim_out, s)
    @inline pullback(Ȳ) = (NoTangent(), NoTangent(), split_back(move_back(Ȳ)[2])[3])
    @inline pullback(::ZeroTangent) = (NoTangent(), NoTangent(), ZeroTangent())
    return y, pullback
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(_move_and_merge_head), t::NamedTuple)
    x = t.hidden_state
    y, back = rrule(config, _move_and_merge_head, x)
    function pullback(Ȳ)
        ∂x = back(Ȳ.hidden_state)
        ∂t = merge(Ȳ, (hidden_state = ∂x,))
        return (NoTangent(), ∂t)
    end
    return y, pullback
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(_move_and_merge_head), x)
    a, move_back = rrule(config, move_head_dim_in, x)
    y, merge_back = rrule(config, merge_head, a)
    @inline pullback(Ȳ) = (NoTangent(), move_back(merge_back(Ȳ)[2])[2])
    return y, pullback
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(dot_product_score), q, k)
    y, pullback = rrule(config, scaled_dot_product_score, q, k, true)
    function dot_product_score_pullback(Ȳ)
        ∂f, ∂q, ∂k, ∂s = pullback(Ȳ)
        return ∂f, ∂q, ∂k
    end
    return y, dot_product_score_pullback
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(scaled_dot_product_score), q, k)
    s = sqrt(inv(size(k, 1)))
    y, pullback = rrule(config, scaled_dot_product_score, q, k, s)
    function scaled_dot_product_score_pullback(Ȳ)
        ∂f, ∂q, ∂k, ∂s = pullback(Ȳ)
        return ∂f, ∂q, ∂k
    end
    return y, scaled_dot_product_score_pullback
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(scaled_dot_product_score), q, k, s)
    y, matmul_pullback = rrule(config, matmul, collapsed_adjoint(k), q, s)
    function scaled_dot_product_score_pullback(Ȳ)
        ∂f, ∂kt, ∂q, ∂s = matmul_pullback(Ȳ)
        ∂k = @thunk begin
            tmp = unthunk(∂kt)
            collapsed_adjoint(tmp)
        end
        return ∂f, ∂q, ∂k, ∂s
    end
    return y, scaled_dot_product_score_pullback
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(masked_score), maskop::AbstractMaskOp, mask, score, args...)
    score_tape = rrule(config, score, args...)
    isnothing(score_tape) && (score_tape = rrule_via_ad(config, score, args...))
    score_val, score_pullback = score_tape
    if isnothing(mask)
        function _score_pullback(Ȳ)
            ∂args = score_pullback(Ȳ)
            return (NoTangent(), NoTangent(), NoTangent(), ∂args...)
        end
        return score_val, _score_pullback
    else
        maskf = apply_mask $ maskop $ AttenMask(mask)
        mask_tape = rrule(config, collapseddims_nonbatch, maskf, score_val)
        y, mask_pullback = mask_tape
        function masked_score_pullback(Ȳ)
            _, _, ∂score = mask_pullback(Ȳ)
            ∂args = score_pullback(∂score)
            return (NoTangent(), NoTangent(), NoTangent(), ∂args...)
        end
        return y, masked_score_pullback
    end
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(normalized_score), norm, score, args...)
    score_tape = rrule(config, score, args...)
    isnothing(score_tape) && (score_tape = rrule_via_ad(config, score, args...))
    s, score_pullback = score_tape
    y, norm_pullback = rrule(config, collapseddims, norm, s)
    function normalized_score(Ȳ)
        _, ∂norm, ∂s = norm_pullback(Ȳ)
        ∂score, ∂args... = score_pullback(∂s)
        return (NoTangent(), ∂norm, ∂score, ∂args...)
    end
    return y, normalized_score
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(weighted_sum_mixing), s, v)
    y, pullback = rrule(config, matmul, v, s, true)
    function weighted_sum_mixing_pullback(Ȳ)
        _, ∂v, ∂s, _ = pullback(Ȳ)
        return (NoTangent(), ∂s, ∂v)
    end
    return y, weighted_sum_mixing_pullback
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(dropout), x, p)
    y, broadcast_pullback = rrule(config, apply_broadcast_mask, (*), RandomMask(p), x, inv(1 - p))
    dropout_pullback(Ȳ) = (NoTangent(), broadcast_pullback(Ȳ)[4], NoTangent())
    return y, dropout_pullback
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(dropout_score), p, score, args...)
    score_tape = rrule(config, score, args...)
    isnothing(score_tape) && (score_tape = rrule_via_ad(config, score, args...))
    score_val, score_pullback = score_tape
    if isnothing(p)
        function _score_pullback(Ȳ)
            ∂args = score_pullback(Ȳ)
            return (NoTangent(), NoTangent(), ∂args...)
        end
        return score_val, _score_pullback
    else
        dropf = Base.Fix2(dropout, p)
        drop_tape = rrule(config, collapseddims, dropf, score_val)
        isnothing(drop_tape) && (drop_tape = rrule_via_ad(config, collapseddims, dropf, score_val))
        y, drop_pullback = drop_tape
        function dropout_score_pullback(Ȳ)
            _, _, ∂score = drop_pullback(Ȳ)
            ∂args = score_pullback(∂score)
            return (NoTangent(), NoTangent(), ∂args...)
        end
        return y, dropout_score_pullback
    end
end

ChainRulesCore.@non_differentiable naive_attention_score(x...)

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(attention_score), f, args...)
    score_tape = rrule(config, f, args...)
    isnothing(score_tape) && (score_tape = rrule_via_ad(config, f, args...))
    score_val, score_pullback = score_tape
    pullback(Ȳ) = (NoTangent(), score_pullback(Ȳ)...)
    return score_val, pullback
end

_merge_grad(a::NoTangent, b::NoTangent) = NoTangent()
_merge_grad(a, b::NoTangent) = a
_merge_grad(a::NoTangent, b) = b
function _merge_grad(a, b)
    a′ = unthunk(a)
    b′ = unthunk(b)
    return @thunk a + b
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(mixing), sr::ScoreReturning, v, g, args...)
    score_tape = rrule(config, attention_score, g, args...)
    isnothing(score_tape) && (score_tape = rrule_via_ad(config, attention_score, g, args...))
    score_val, score_pullback = score_tape
    f_tape = rrule(config, f, score_val, v)
    isnothing(f_tape) && (f_tape = rrule_via_ad(config, f, score_val, v))
    y, f_pullback = f_tape
    y′, unwrap_pullback = rrule(config, unwrap_collapse, y)
    s, s_unwrap_pullback = rrule(config, unwrap_collapse, score_val)
    function pullback(Ȳ)
        _, ∂y = unwrap_pullback(Ȳ.hidden_state)
        _, ∂s1 = s_unwrap_pullback(Ȳ.attention_score)
        ∂f, ∂s2, ∂v = f_pullback(∂y)
        ∂s = _merge_grad(∂s1, ∂s2)
        _, ∂g, ∂args... = score_pullback(∂s)
        ∂sr = ∂f isa NoTangent ? NoTangent() : (f = ∂f,)
        return (NoTangent(), ∂sr, ∂v, ∂g, ∂args...)
    end
    return (hidden_state = y′, attention_score = s), pullback
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(mixing), f, v, g, args...)
    score_tape = rrule(config, attention_score, g, args...)
    isnothing(score_tape) && (score_tape = rrule_via_ad(config, attention_score, g, args...))
    score_val, score_pullback = score_tape
    f_tape = rrule(config, f, score_val, v)
    isnothing(f_tape) && (f_tape = rrule_via_ad(config, f, score_val, v))
    y, f_pullback = f_tape
    y′, unwrap_pullback = rrule(config, unwrap_collapse, y)
    function mixing_pullback(Ȳ)
        _, ∂y = unwrap_pullback(Ȳ)
        ∂f, ∂s, ∂v = f_pullback(∂y)
        _, ∂g, ∂args... = score_pullback(∂s)
        return (NoTangent(), ∂f, ∂v, ∂g, ∂args...)
    end
    return y′, mixing_pullback
end
