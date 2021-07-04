using ChainRulesCore

#=
  score: dot product, scale dot product, similarity, concat-linear, linear
=#
function attention_score(q, k) end

dot_product_score(q::AbstractMatrix, k::AbstractMatrix) = q' * k
dot_product_score(q::AbstractArray{T, 3}, k::AbstractArray{T, 3}) where T = NNlib.batched_mul(NNlib.batched_transpose(k), q)

scale_dot_product_score(q::AbstractMatrix, k::AbstractMatrix, s=size(k, 1)) = (q' * k) ./ sqrt(s)
scale_dot_product_score(q::AbstractArray{T, 3}, k::AbstractArray{T, 3}, s=size(k,1)) where T = NNlib.batched_mul(NNlib.batched_transpose(k), q) ./ sqrt(s)


#=
  normailze: softmax, l2-norm, l1-norm
=#
function normailze end

#=
  dropout rate: (0, 1)
=#
function dropout end


# mask 
function attention(q, k, v, score=dot_product_score, normalize=softmax, dropout=identity)
  s = dropout(score(q, k))
  s_ = normalize(s)
  # TODO: apply mask
  y = NNlib.batched_mul(v, s_)
  return y
end

# function ChainRulesCore.rrule(::typeof(attention), q, k, v, score, normalize, dropout)
#   y = attention(q, k, v, score, normalize, dropout)
  
#   function attention_pullback(Ŷ)
    
#     return (NoTangent(), )
#   end
#   return y, attention_pullback
# end


