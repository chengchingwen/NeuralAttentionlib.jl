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
    proj = ProjectTo(x)
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
    function pullback(Ybar)
        Ȳ = ChainRulesCore.backing(unthunk(Ybar))
        _, ∂x = back(Ȳ.hidden_state)
        ∂t = merge(Ȳ, (hidden_state = ∂x,))
        return (NoTangent(), ∂t)
    end
    y′ = merge(t, (hidden_state = y,))
    return y′, pullback
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

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(mixing), sr::ScoreReturning, v, g, args...)
    score_tape = rrule(config, attention_score, g, args...)
    isnothing(score_tape) && (score_tape = rrule_via_ad(config, attention_score, g, args...))
    score_val, score_pullback = score_tape
    f = sr.f
    f_tape = rrule(config, f, score_val, v)
    isnothing(f_tape) && (f_tape = rrule_via_ad(config, f, score_val, v))
    y, f_pullback = f_tape
    y′, unwrap_pullback = rrule(config, unwrap_collapse, y)
    s, s_unwrap_pullback = rrule(config, unwrap_collapse, score_val)
    function pullback(Ybar)
        Ȳ = unthunk(Ybar)
        _, ∂y = unwrap_pullback(Ȳ.hidden_state)
        _, ∂s1 = s_unwrap_pullback(Ȳ.attention_score)
        ∂f, ∂s2, ∂v = f_pullback(∂y)
        ∂s = unthunk(∂s1) + unthunk(∂s2)
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

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(generic_qkv_attention), mixingf, scoref, q, k, v, args...)
    v′, v_back = rrule(config, as_collapsed, v)
    q′, q_back = rrule(config, as_collapsed, q)
    k′, k_back = rrule(config, as_collapsed, k)
    y, pullback = rrule(config, mixing, mixingf, v′, scoref, q′, k′, args...)
    function generic_qkv_attention_pullback(Ȳ)
        _, ∂mixingf, ∂v′, ∂scoref, ∂q′, ∂k′, ∂args... = pullback(Ȳ)
        _, ∂k = k_back(∂k′)
        _, ∂q = q_back(∂q′)
        _, ∂v = v_back(∂v′)
        return (NoTangent(), ∂mixingf, ∂scoref, ∂q, ∂k, ∂v, ∂args...)
    end
    return y, generic_qkv_attention_pullback
end

function ChainRulesCore.rrule(
    config::RuleConfig, ::typeof(generic_multihead_qkv_attention),
    mixingf, scoref, head::Integer, q::AbstractArray, k::AbstractArray, v::AbstractArray, args...
)
    q′, q_back = rrule(config, as_collapsed, q)
    k′, k_back = rrule(config, as_collapsed, k)
    v′, v_back = rrule(config, as_collapsed, v)
    y, atten_back = rrule(config, generic_multihead_qkv_attention, mixingf, scoref, head, q′, k′, v′, args...)
    function generic_multihead_qkv_attention_pullback(Ȳ)
        _, ∂mixingf, ∂scoref, _, ∂q′, ∂k′, ∂v′, ∂args... = atten_back(Ȳ)
        _, ∂v = v_back(∂v′)
        _, ∂k = k_back(∂k′)
        _, ∂q = q_back(∂q′)
        return (NoTangent(), ∂mixingf, ∂scoref, NoTangent(), ∂q, ∂k, ∂v, ∂args...)
    end
    return y, generic_multihead_qkv_attention_pullback
end

function ChainRulesCore.rrule(
    config::RuleConfig, ::typeof(generic_multihead_qkv_attention),
    mixingf, scoref, head::Integer, q::CollapsedDimsArray, k::CollapsedDimsArray, v::CollapsedDimsArray, args...
)
    hq, hq_back = rrule(config, _split_and_move_head, head, q)
    hk, hk_back = rrule(config, _split_and_move_head, head, k)
    hv, hv_back = rrule(config, _split_and_move_head, head, v)
    t, atten_back = rrule(config, generic_qkv_attention, mixingf, scoref, hq, hk, hv, args...)
    y, back = rrule(config, _move_and_merge_head, t)
    @inline function generic_multihead_qkv_attention_pullback(Ȳ)
        _, ∂t = back(Ȳ)
        _, ∂mixingf, ∂scoref, ∂hq, ∂hk, ∂hv, ∂args... = atten_back(∂t)
        _, _, ∂v = hv_back(∂hv)
        _, _, ∂k = hk_back(∂hk)
        _, _, ∂q = hq_back(∂hq)
        return (NoTangent(), ∂mixingf, ∂scoref, NoTangent(), ∂q, ∂k, ∂v, ∂args...)
    end
    return y, generic_multihead_qkv_attention_pullback
end
