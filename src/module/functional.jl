module Functional

using ..NeuralAttentionlib: @imexport

@imexport import ..NeuralAttentionlib:
    generic_qkv_attention, generic_multihead_qkv_attention,
    naive_qkv_attention, multihead_qkv_attention,
    mixing, weighted_sum_mixing, attention_score,
    scaled_dot_product_score, dot_product_score,
    masked_score, normalized_score

import ..NeuralAttentionlib:
    split_head, move_head_dim_out_perm, move_head_dim_out,
    move_head_dim_in_perm, move_head_dim_in, merge_head

using ..NeuralAttentionlib: SymLengthMask, BiLengthMask

"""
    generic_qkv_attention(mixingf, scoref, q, k, v, args...)

Generic version of [`naive_qkv_attention`](@ref). Need to specify mixing and score function.
"""
generic_qkv_attention

"""
    generic_multihead_qkv_attention(mixingf, scoref, head, q, k, v, args...)

Generic version of [`multihead_qkv_attention`](@ref). Need to specify mixing and score function.
"""
generic_multihead_qkv_attention

"""
    multihead_qkv_attention(head, q, k, v, mask=nothing)

Multihead version of [`naive_qkv_attention`](@ref). The core operation for implement a regular transformer layer.
"""
multihead_qkv_attention

@doc raw"""
    naive_qkv_attention(q, k, v, mask=nothing)

The scaled dot-product attention of a regular transformer layer.

``Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V``

It's equivalent to `generic_qkv_attention(weighted_sum_mixing, normalized_score(NNlib.softmax) $ masked_score(mask) $ scaled_dot_product_score, q, k, v)`.

#Example

```julia
julia> fdim, ldim, bdim = 32, 10, 4;

julia> x = randn(fdim, ldim, bdim);

julia> y = naive_qkv_attention(x, x, x); # simple self attention

# no mask here
julia> z = generic_qkv_attention(weighted_sum_mixing, normalized_score(NNlib.softmax) $ scaled_dot_product_score, x, x, x);

julia> y ≈ z
true

```

See also: [`generic_qkv_attention`](@ref)
"""
naive_qkv_attention

"""
    mixing(f, v, g, args...) = f(attention_score(g, args...), v)

`Mixing` function api. Can be overload for doing custom implementation with [`generic_qkv_attention`](@ref).
 `f` is the mixing function and `g` is score function.

See also: [`generic_qkv_attention`](@ref), [`generic_multihead_qkv_attention`](@ref), [`attention_score`](@ref)
"""
mixing

"""
    weighted_sum_mixing(s, v)

The mixing function of a regular transformer layer. `s` is the attention score and `v` is the value of QKV attention.
"""
weighted_sum_mixing

"""
    attention_score(f, args...) = f(args...)

Attention score api. Can be overload for doing custom implementation with [`generic_qkv_attention`](@ref).
 `f` is the score function.

See also: [`generic_qkv_attention`](@ref), [`generic_multihead_qkv_attention`](@ref), [`mixing`](@ref)
"""
attention_score

@doc raw"""
    normalized_score(norm) = normalized_score $ norm
    normalized_score(norm, score, args...)

Normalized attenion score api. `norm` is the normalize function (like `softmax`) and `score` is the function
 that compute attention score from `args...`.

See also: [`naive_qkv_attention`](@ref)
"""
normalized_score

@doc raw"""
    masked_score(mask) = masked_score $ mask
    masked_score(maskop::AbstractAttenMaskOp, mask::AbstractAttenMask, score, args...)

Masked attention score api. Applying the `mask` according to `maskop` on the attention score
 compute from `score(args...)`.

See also: [`naive_qkv_attention`](@ref), [`SymLengthMask`](@ref), [`BiLengthMask`](@ref)
"""
masked_score

@doc raw"""
     scaled_dot_product_score(q, k, s = sqrt(inv(size(k, 1))))

The scaled dot-product attention score function of a regular transformer layer.

``Score(Q, K) = \frac{QK^T}{\sqrt{d_k}}``

See also: [`naive_qkv_attention`](@ref)
"""
scaled_dot_product_score

"""
    dot_product_score(q, k)

Dot-product attention score function. Equivalent to `scaled_dot_product_score(q, k, 1)`.

See also: [`scaled_dot_product_score`](@ref)
"""
dot_product_score

"""
    split_head(head::Int, x)

Split the first dimension into `head` piece of small vector. Equivalent to 
 `reshape(x, :, head, tail(size(x))...)`.
"""
split_head

"""
    move_head_dim_out_perm(x::AbstractArray{T, N}, nobatch=false)
    move_head_dim_out_perm(N::Int, nobatch=false)

Dimension order for `permutedims` to move the `head` dimension (created by [`split_head`](@ref)) to batch dimension.
 Return a tuple of integer of length `n`. `nobatch` specify where `x` is a batch of data.

# Example

```julia
julia> Functional.move_head_dim_out_perm(5, false)
(1, 3, 4, 2, 5)

julia> Functional.move_head_dim_out_perm(5, true)
(1, 3, 4, 5, 2)

```

See also: [`split_head`](@ref), [`move_head_dim_out`](@ref)
"""
move_head_dim_out_perm

"""
    move_head_dim_out(x::AbstractArray, nobatch=false)

Equivanlent to `permutedims(x, move_head_dim_out_perm(x, nobatch)))`

See also: [`split_head`](@ref), [`move_head_dim_out_perm`](@ref)
"""
move_head_dim_out

"""
    move_head_dim_in_perm(x::AbstractArray{T, N}, nobatch=false)
    move_head_dim_in_perm(N::Int, nobatch=false)

Dimension order for `permutedims` to move the `head` dimension (created by [`split_head`](@ref)) from batch dimension
 to feature dimension (for [`merge_head`](@ref)). Return a tuple of integer of length `n`.
 `nobatch` specify where `x` is a batch of data.

# Example

```julia
julia> Functional.move_head_dim_in_perm(5, false)
(1, 4, 2, 3, 5)

julia> Functional.move_head_dim_in_perm(5, true)
(1, 5, 2, 3, 4)

```

See also: [`merge_head`](@ref), [`move_head_dim_in`](@ref)
"""
move_head_dim_in_perm

"""
    move_head_dim_in(x::AbstractArray, nobatch=false)

Equivanlent to `permutedims(x, move_head_dim_in_perm(x, nobatch)))`

See also: [`merge_head`](@ref), [`move_head_dim_in_perm`](@ref)
"""
move_head_dim_in

"""
    merge_head(x)

merge the `head` dimension split by [`split_head`](@ref).
"""
merge_head

end
