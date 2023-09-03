using ChainRulesCore

_alibi_base(cp2) = 2 ^ (- (2 ^ (3 - log2(cp2))))
function _alibi_slope(base, ebase, cp2, i)
    if i <= cp2
        b = base
        p = i
    else
        b = ebase
        p = ((i - cp2) << 1) - 1
    end
    return b ^ p
end
_alibi(base, ebase, cp2, h, c::CartesianIndex) = _alibi(base, ebase, cp2, h, first(Tuple(c)))
function _alibi(base, ebase, cp2, h, i)
    s = _alibi_slope(base, ebase, cp2, h)
    return s * (i - 1)
end


_alibi_pattern_path(pat::Union{LengthMask, RevLengthMask, GenericSequenceMask, Nothing}) = true
_alibi_pattern_path(pat) = false
_alibi_mask_pattern(mask::Union{CausalMask, SymLengthMask, BiLengthMask}) = nothing
_alibi_mask_pattern(mask::RevSymLengthMask) = RevLengthMask(mask.len)
_alibi_mask_pattern(mask::RevBiLengthMask) = RevLengthMask(mask.k_len)
function _alibi_mask_pattern(mask::BatchedMask)
    pat = _alibi_mask_pattern(mask.mask)
    return _alibi_pattern_path(pat) ? pat : mask
end
_alibi_mask_pattern(mask) = mask
_alibi_and_mask_pattern(masks) = _alibi_and_mask_pattern(_alibi_mask_pattern(first(masks)), Base.tail(masks))
_alibi_and_mask_pattern(pat, ::Tuple{}) = pat
function _alibi_and_mask_pattern(pat, masks::Tuple)
    pat1 = _alibi_mask_pattern(first(masks))
    return pat == pat1 && _alibi_pattern_path(pat) ? _alibi_and_mask_pattern(pat, Base.tail(masks)) : ()
end
function _alibi_mask_pattern(mask::CombinedMask{typeof(&)})
    pat = _alibi_and_mask_pattern(mask.masks)
    return _alibi_pattern_path(pat) ? pat : mask
end

_build_alibi(::LengthMask, score) = _build_alibi(nothing, score)
function _build_alibi(::Nothing, score)
    klen, qlen, bh = size(score)
    bs = noncollapsed_size(score, 3)
    # Assume head dim exist, we treat the first number of batch dim as head dim, and others are still batch dims
    b = ntuple(_->1, length(Base.tail(bs)))
    h = first(bs)
    cp2 = prevpow(2, h)
    base = _alibi_base(cp2)
    ebase = _alibi_base(cp2 << 1)
    return Broadcast.broadcasted(_alibi, base, ebase, cp2,
                                 LinearIndices((1, 1, h, b...)), CartesianIndices((klen, 1, 1, b...)))
end
function _build_alibi(mask::GenericSequenceMask, score)
    klen, qlen, bh = size(score)
    bs = noncollapsed_size(score, 3)
    b = ntuple(_->1, length(Base.tail(bs)))
    h = first(bs)
    cp2 = prevpow(2, h)
    base = _alibi_base(cp2)
    ebase = _alibi_base(cp2 << 1)
    ms = Base.tail(size(mask.mask))
    # add an extra head dim here, we are assuming that GenericSequenceMask would NOT contain head mask.
    _mask = reshape(mask.mask, (first(ms), 1, 1 #= head dim =#, Base.tail(ms)...))
    indices = similar(_mask, Int32)
    cumsum!(indices, _mask; dims = 1)
    return Broadcast.broadcasted(_alibi, base, ebase, cp2, LinearIndices((1, 1, h, b...)), indices)
end
function _build_alibi(mask::RevLengthMask, score)
    klen, qlen, bh = size(score)
    bs = noncollapsed_size(score, 3)
    m = similar(score, Bool, 1, klen, Base.tail(bs)...)
    m .= first.(tuple.(mask, m))
    return build_alibi(GenericSequenceMask(m), score)
end
function _build_alibi(mask::AbstractMask, score)
    klen, qlen, bh = size(score)
    bs = noncollapsed_size(score, 3)
    b = ntuple(_->1, length(Base.tail(bs)))
    h = first(bs)
    cp2 = prevpow(2, h)
    base = _alibi_base(cp2)
    ebase = _alibi_base(cp2 << 1)
    indices = similar(score, Int32, klen, qlen, bs...)
    indices .= first.(tuple.(mask, collapseddims_nonbatch(score)))
    cumsum!(indices, indices; dims=1)
    return Broadcast.broadcasted(_alibi, base, ebase, cp2, LinearIndices((1, 1, h, b...)), indices)
end

function build_alibi(mask::Union{AbstractMask, Nothing}, score)
    pat = _alibi_mask_pattern(mask)
    return _build_alibi(pat, score)
end


alibi_position_embedding(mask::Union{AbstractMask, Nothing}) = alibi_position_embedding $ mask
alibi_position_embedding(score, args...) = alibi_position_embedding(nothing, score, args...)
function alibi_position_embedding(mask::Union{AbstractMask, Nothing}, score, args...)
    score_val = score(args...)
    alibi = build_alibi(mask, score_val)
    return collapseddims_nonbatch(Base.Fix1(.+, alibi), score_val)
end
alibi_position_embedding(::typeof(build_alibi), alibi, score, args...) =
    collapseddims_nonbatch(Base.Fix1(.+, alibi), score(args...))

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(alibi_position_embedding), score, args...)
    y, pullback = rrule(config, alibi_position_embedding, nothing, score, args...)
    alibi_pullback(Ȳ) = Base.tail(pullback(Ȳ))
    return y, alibi_pullback
end
function ChainRulesCore.rrule(config::RuleConfig, ::typeof(alibi_position_embedding),
                              mask::Union{AbstractMask, Nothing}, score, args...)
    score_tape = rrule(config, score, args...)
    isnothing(score_tape) && (score_tape = rrule_via_ad(config, score, args...))
    score_val, score_pullback = score_tape
    alibi = build_alibi(mask, score_val)
    function alibi_pullback(Ȳ)
        ∂args = score_pullback(Ȳ)
        return (NoTangent(), NoTangent(), ∂args...)
    end
    return collapseddims_nonbatch(Base.Fix1(.+, alibi), score_val), alibi_pullback
end
function ChainRulesCore.rrule(config::RuleConfig, ::typeof(alibi_position_embedding),
                              ::typeof(build_alibi), alibi, score, args...)
    score_tape = rrule(config, score, args...)
    isnothing(score_tape) && (score_tape = rrule_via_ad(config, score, args...))
    score_val, score_pullback = score_tape
    function alibi_pullback(Ȳ)
        ∂args = score_pullback(Ȳ)
        return (NoTangent(), NoTangent(), NoTangent(), ∂args...)
    end
    return collapseddims_nonbatch(Base.Fix1(.+, alibi), score_val), alibi_pullback
end
