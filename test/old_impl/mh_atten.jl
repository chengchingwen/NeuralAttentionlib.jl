using ChainRulesCore
using Flux
using Flux.Functors: Functors, @functor
using MacroTools: @forward, @capture

macro toNd(ex, outref::Int=1)
    if @capture ex f_(xs__; kw__)
        kwe = Expr(:parameters, kw...)
    else
        @capture ex f_(xs__)
    end
    rxs = map(xs) do x
        :(reshape($x, size($x, 1), :))
    end
    newcall = kw === nothing ? Expr(:call, f, rxs...) : Expr(:call, f, kwe, rxs...)
    :(reshape($newcall, :, Base.tail(size($(xs[outref])))...)) |> esc
end

abstract type AbstractAttention end

create_atten_mask(T::Type, score::AbstractArray, ::Nothing, future) = create_atten_mask(T, score, fill!(similar(score, size(score,1), size(score, 2), 1), one(T)), future)
function create_atten_mask(T::Type, score::AbstractArray, _mask::AbstractArray, future::Bool=false)
  #size(mask) == (q, k, n, b)

  # ql, kl = size(mask)
  mask = copy(_mask)

  maskval = convert(T, -1e9)
  !future && batched_triu!(mask, 0)
  mask .= (1 .- mask) .* maskval
  return mask
end
ChainRulesCore.@non_differentiable create_atten_mask(T, score, _mask, future)

struct MultiheadAttention{Q<:Dense, K<:Dense, V<:Dense, O<:Dense, DP} <: AbstractAttention
    head::Int
    future::Bool
    iqproj::Q
    ikproj::K
    ivproj::V
    oproj::O
    drop::DP
end

Functors.functor(mh::MultiheadAttention) = (mh.iqproj, mh.ikproj, mh.ivproj, mh.oproj), m -> MultiheadAttention(mh.head, mh.future, m..., mh.drop)

MultiheadAttention(head::Int, is::Int, hs::Int, os::Int; future::Bool=true, drop = nothing) =
    MultiheadAttention(head, future,
                       Dense(is, hs*head), Dense(is, hs*head), Dense(is, hs*head), Dense(hs*head, os),
                       isnothing(drop) ? nothing : NeuralAttentionlib._dropout_func(drop))


function Base.show(io::IO, mh::MultiheadAttention)
    hs = div(size(mh.iqproj.W)[1], mh.head)
    is = size(mh.iqproj.W)[end]
    os = size(mh.oproj.W)[1]

    print(io, "MultiheadAttention(")
    print(io, "head=$(mh.head), ")
    print(io, "head_size=$(hs), ")
    print(io, "$(is)=>$(os)")

    if Flux.istraining()
        print(io, ", dropout=$(mh.drop.p))")
    else
        print(io, ")")
    end
end

function (mh::MultiheadAttention)(query::A1,
                                  key::A2,
                                  value::A3;
                                  mask=nothing, return_score=false) where {T,
                                                       A1 <: Abstract3DTensor{T},
                                                       A2 <: Abstract3DTensor{T},
                                                       A3 <: Abstract3DTensor{T}}
  qs = size(query)
  ks = size(key)
  vs = size(value)

  #size(ipq) == (h, q_seq_len, batch)
  ipq = @toNd mh.iqproj(query)
  ipk = @toNd mh.ikproj(key)
  ipv = @toNd mh.ivproj(value)

  h = size(ipq, 1)
  hs = div(h, mh.head)

  #size(ipq) == (hs, q_seq_len, head, batch)
  ipq = permutedims(reshape(ipq, hs, mh.head, qs[2], qs[3]), [1, 3, 2, 4])
  ipk = permutedims(reshape(ipk, hs, mh.head, ks[2], ks[3]), [1, 3, 2, 4])
  ipv = permutedims(reshape(ipv, hs, mh.head, vs[2], vs[3]), [1, 3, 2, 4])

  #size(ipq) == (hs, q_seq_len, head * batch)
  ipq = reshape(ipq, hs, qs[2], :)
  ipk = reshape(ipk, hs, ks[2], :)
  ipv = reshape(ipv, hs, vs[2], :)

  atten, score = attention(ipq,ipk,ipv,
                    mask,
                    mh.future,
                    mh.drop,
                    return_score)

  atten = permutedims(reshape(atten, hs, qs[2], mh.head, qs[3]), [1, 3, 2, 4]) #size(atten) == (hs, head, ql, b)
  atten = reshape(atten, h, qs[2], qs[3]) #size(atten) == (h, ql, b)

  out = @toNd mh.oproj(atten)
  return_score ? (out, score) : out #size(out) == (h, q_seq_len, batch)
end

function (mh::MultiheadAttention)(query::A1,
                                  key::A2,
                                  value::A3;
                                  mask=nothing, return_score=false) where {T,
                                                       A1 <: AbstractMatrix{T},
                                                       A2 <: AbstractMatrix{T},
                                                       A3 <: AbstractMatrix{T}}

  # size(query) == (dims, seq_len)
  ipq = mh.iqproj(query)
  ipk = mh.ikproj(key)
  ipv = mh.ivproj(value)

  h = size(ipq)[1] #h == hs * head
  hs = div(h, mh.head)

  #size(hq) == (hs, seq_len, head)
  hq = permutedims(reshape(ipq, hs, mh.head, :), [1, 3, 2])
  hk = permutedims(reshape(ipk, hs, mh.head, :), [1, 3, 2])
  hv = permutedims(reshape(ipv, hs, mh.head, :), [1, 3, 2])

  atten, score = attention(hq, hk, hv,
                    mask,
                    mh.future,
                    mh.drop, return_score)

  # size(atten) == (head*hs, seq_len)
  atten = reshape(permutedims(atten, [1, 3, 2]), h, :)

  result = mh.oproj(atten)
  return_score ? (result, score) : result
end

function apply_mask(score, mask)
  s = size(score)
  ms = size(mask)
  bxn = s[end]
  b = ms[end]
  if bxn == b || b == 1
    return score .+ mask
  else
    return reshape(reshape(score, s[1:end-1]..., :, b) .+
                   reshape(mask, ms[1:end-1]..., 1, b), s)
  end
end

apply_mask(score::AbstractArray{T}, ::Nothing, future) where T = future ? score : apply_mask(score, create_atten_mask(T, score, nothing, future))
apply_mask(score::AbstractArray{T}, mask, future) where T = apply_mask(score, create_atten_mask(T, score, mask, future))

function attention(query::A1,
                   key::A2,
                   value::A3,
                   mask, future::Bool,
                   drop, return_score::Bool) where {T,
                                   A1 <: Abstract3DTensor{T},
                                   A2 <: Abstract3DTensor{T},
                                   A3 <: Abstract3DTensor{T}}
  #size(query) == (dims, {q,k}_seq_len, batch) == size(key) == size(value)
  #size(score) == (k_seq_len, q_seq_len, batch)
  dk = size(key, 1)
  score = batchedmul(key, query; transA = true)
  score = score ./ convert(T, sqrt(dk))

  score = apply_mask(score, mask, future)
  score = softmax(score; dims=1)
  !isnothing(drop) && (score = drop(score))
  result = batchedmul(value, score) #size(return) == (dims, q_seq_len, batch)
  return return_score ? (result, score) : (result, nothing)
end
