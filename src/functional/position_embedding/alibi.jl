using ChainRulesCore

_alibi_base(cp2) = Float32(2 ^ (- (2 ^ (3 - log2(cp2)))))
function _alibi_slope(base, ebase, cp2, e0, i)
    i = unsafe_trunc(Int32, i)
    le = i <= cp2
    b = ifelse(le, base, ebase)
    e = (i << one(i)) - e0
    p = oftype(b, ifelse(le, i, e))
    return b ^ p
end
function _alibi(base, ebase, cp2, e0, h, i)
    s = _alibi_slope(base, ebase, cp2, e0, h)
    return s * (i - one(i))
end

# fast path for a few mask types:
#  based on the translation invariance property of softmax, the alibi value for each query are shift to have 0 on
#  the first key. As long as there is no "gaps" between keys, the alibi values with different mask can be the same.
#  for other kind of masks, we compute the "indices" that ingores all the "gaps" between keys hence allocating a
#  Int32 array with the same size of the score array.
_alibi_mask_pattern(::Nothing, _) = nothing
_alibi_mask_pattern(mask::AbstractAttenMask, _) = mask
_alibi_mask_pattern(mask::BatchedMask, score) = (pat = _alibi_mask_pattern(mask.mask, score); isnothing(pat) ? nothing : mask)
_alibi_mask_pattern(mask::CombinedMask{typeof(&)}, score) = all(isnothing ∘ (Base.Fix2(_alibi_mask_pattern, score)), mask.masks) ? nothing : mask
_alibi_mask_pattern(::CausalMask, _) = nothing
_alibi_mask_pattern(::LocalMask, _) = nothing
_alibi_mask_pattern(::BandPartMask, _) = nothing
_alibi_mask_pattern(::SymLengthMask, _) = nothing
_alibi_mask_pattern(::BiLengthMask, _) = nothing
_alibi_mask_pattern(::RevSymLengthMask, _) = nothing
_alibi_mask_pattern(::RevBiLengthMask, _) = nothing
_alibi_mask_pattern(mask::BiSeqMask, score) = (pat = _alibi_k_seqmask_pattern(mask.k_mask, score); isnothing(pat) ? nothing : mask)
_alibi_k_seqmask_pattern(::LengthMask, _) = nothing
_alibi_k_seqmask_pattern(::RevLengthMask, _) = nothing

# alibi is designed for multi-head attention and thus have a head specific scalar
# we assume the head dim exist and treat the first dim of batch dims as head dim, and others are still batch dims
function _build_alibi(::Nothing, score)
    klen, qlen, bh = size(score)
    bs = noncollapsed_size(score, 3)
    b = ntuple(one, Val(length(Base.tail(bs))))
    h = first(bs)
    cp2 = Int32(prevpow(2, h))
    rp2 = cp2 << 1
    base = _alibi_base(cp2)
    ebase = _alibi_base(rp2)
    e0 = rp2 + one(rp2)
    return Broadcast.broadcasted(_alibi, base, ebase, cp2, e0, LinearIndices((1, 1, h, b...)), Base.OneTo{Int32}(klen))
end
function _build_alibi(mask::AbstractAttenMask, score)
    klen, qlen, bh = size(score)
    bs = noncollapsed_size(score, 3)
    b = ntuple(one, length(Base.tail(bs)))
    h = first(bs)
    cp2 = Int32(prevpow(2, h))
    rp2 = cp2 << 1
    base = _alibi_base(cp2)
    ebase = _alibi_base(rp2)
    e0 = rp2 + one(rp2)
    shape = (klen, qlen, bs...)
    indices = _fast_broadcast2!(identity, similar(score, Int32, shape), GetIndexer(mask, shape))
    cumsum!(indices, indices; dims=1)
    return Broadcast.broadcasted(_alibi, base, ebase, cp2, e0, LinearIndices((1, 1, h, b...)), indices)
end
build_alibi(mask::Union{AbstractAttenMask, Nothing}, score) = _build_alibi(_alibi_mask_pattern(mask, score), score)

alibi_position_embedding(mask::Union{AbstractAttenMask, Nothing}) = alibi_position_embedding $ mask
alibi_position_embedding(score, args...) = alibi_position_embedding(nothing, score, args...)
function alibi_position_embedding(mask::Union{AbstractAttenMask, Nothing}, score, args...)
    score_val = score(args...)
    alibi = build_alibi(mask, score_val)
    score_val2 = similar(score_val)
    _fast_broadcast2!(+, collapseddims_nonbatch(score_val2), alibi, collapseddims_nonbatch(score_val))
    return score_val2
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(alibi_position_embedding), score, args...)
    y, pullback = rrule(config, alibi_position_embedding, nothing, score, args...)
    alibi_pullback(Ȳ) = Base.tail(pullback(Ȳ))
    return y, alibi_pullback
end
function ChainRulesCore.rrule(config::RuleConfig, ::typeof(alibi_position_embedding),
                              mask::Union{AbstractAttenMask, Nothing}, score, args...)
    score_tape = rrule(config, score, args...)
    isnothing(score_tape) && (score_tape = rrule_via_ad(config, score, args...))
    score_val, score_pullback = score_tape
    alibi = build_alibi(mask, score_val)
    score_val2 = similar(score_val)
    _fast_broadcast2!(+, collapseddims_nonbatch(score_val2), alibi, collapseddims_nonbatch(score_val))
    alibi_pullback(Ȳ) = (NoTangent(), NoTangent(), score_pullback(Ȳ)...)
    return score_val2, alibi_pullback
end
