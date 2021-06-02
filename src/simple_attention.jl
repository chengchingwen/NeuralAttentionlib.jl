struct Attention{Q, K, V}
  D_Q::Q
  D_K::K
  D_V::V
end

@functor Attention

function extract_param(F::Attention)
  D_Q, D_K, D_V = Flux.trainable(F)
  W_Q, b_Q = Flux.trainable(D_Q)
  W_K, b_K = Flux.trainable(D_K)
  W_V, b_V = Flux.trainable(D_V)
  return (W_Q, W_K, W_V, b_Q, b_K, b_V)
end

function flux_attention_forward(F::Attention, Q, K, V)
  HQ = F.D_Q(Q)
  HK = F.D_K(K)
  HV = F.D_V(V)
  A = HK' * HQ
  d_k = size(HK, 1)
  As = A ./ convert(eltype(A), √d_k)
  S = softmax(As)
  Y = HV * S
  return Y
end

attentionFW(Q, K, V, W_Q, W_K, W_V, b_Q, b_K, b_V) = attentionAD(Q, K, V, W_Q, W_K, W_V, b_Q, b_K, b_V, Val(false))
function attentionAD(Q, K, V, W_Q, W_K, W_V, b_Q, b_K, b_V, ::Val{BW} = Val(true)) where BW
  HQ = W_Q * Q .+ b_Q
  HK = W_K * K .+ b_K
  HV = W_V * V .+ b_V
  A = HK' * HQ
  #As = A ./ √d_k
  #C, C_idx = findmax(As, dims=1) #maximum(As, dims=1)
  #T = As .- C  
  #E = exp.(As .- C)
  d_k = size(HK, 1)
  s = convert(eltype(A), √d_k)
  if BW
    C, C_idx = findmax(A, dims=1)
    E = Broadcast.instantiate(@~ exp.((A .- C) ./ s))
    Es = sum(E, dims=1)
    S = E ./ Es
    Broadcast.instantiate(S)
  else
    #C = maximum(A, dims=1)
    S = softmax(A ./ s)
  end
  Y = HV * S
  if !BW
    return Y
  else
    return Y, function (dY)
      dS = HV' * dY
      dHV = dY * S'
      _Es = Broadcast.instantiate(@~(dS .* E ./ (.- Es .^ 2)))
      dEs = sum(_Es, dims=1)
      #dE2 = dS ./ Es
      #dE1 = repeat(dEs, size(E, 1))
      #dE = dEs .+ dE2
      dE = Broadcast.instantiate(@~ dEs .+ (dS ./ Es))
      dT = Broadcast.instantiate(@~ dE .* E)
      dC = sum(dT, dims=1)
      #dAs2 = dT
      #dAs1 = setindex!(zero(As), dC, C_idx) # zero(As)[C_idx] = dC
      #dAs = dAs1 + dAs2
      #dAs = dAs1 .+ (dE .* E)
      #dA = dAs ./ √d_k
      dAs1 = setindex!(zero(A), dC, C_idx)
      dA = (dAs1 .+ (dE .* E)) ./ s
      dHQ = HK * dA
      dHK = HQ * dA'
      dV = W_V' * dHV
      dW_V = dHV * V'
      db_V = sum(dHV, dims=ndims(dHV))
      dK = W_K' * dHK
      dW_K = dHK * K'
      db_K = sum(dHK, dims=ndims(dHK))
      dQ = W_Q' * dHQ
      dW_Q = dHQ * Q'
      db_Q = sum(dHQ, dims=ndims(dHQ))
      return (dQ, dK, dV, dW_Q, dW_K, dW_V, db_Q, db_K, db_V)
    end
  end
end


# function attention(Q, K, V, W_Q, W_K, W_V, b_Q, b_K, b_V, d_k)
#   # Dense_Q
#   HQ = W_Q * Q .+ b_Q
#   db_Q = sum(dHQ, dims=-1)
#   dW_Q = dHQ * Q'
#   dQ = W_Q' * dHQ
#   # Dense_K
#   HK = W_K * K .+ b_K
#   db_K = sum(dHK, dims=-1)
#   dW_K = dHK * K'
#   dK = W_K' * dHK
#   # Dense_V
#   HV = W_V * V .+ b_V
#   db_V = sum(dHV, dims=-1)
#   dW_V = dHV * V'
#   dV = W_V' * dHV
#   # Attention Score
#   A = HK' * HQ
#   dHK = HQ * dA'
#   dHQ = HK * dA
#   As = A ./ √d_k
#   dA = dAs ./ √d_k
#   # Softmax Normalization
#   C, C_idx = findmax(As, dims=1) #maximum(As, dims=1)
#   dAs1 = setindex!(zero(As), dC, C_idx) # zero(As)[C_idx] = dC
#   T = As .- C
#   dAs2 = dT
#   dC = sum(dT, dims=1)
#   E = exp.(T)
#   dT = dE .* E
#   Es = sum(E, dims=1)
#   dE1 = repeat(dEs, size(E, 1))
#   S = E ./ Es
#   dE2 = dS ./ Es
#   dEs = sum(dS .* E ./ (- Es .^ 2), dims=1)
#   dE = dE1 + dE2
#   # mixer
#   Y = HV * S
#   dV = dY * S'
#   dS = HV' * dY
#   return Y, (dQ, dK, dV, dW_Q, dW_K, dW_V, db_Q, db_K, db_V)
# end


function compare_grad(flux_grad, ad_grad)
  @show flux_grad[2] ≈ ad_grad[1]
  @show flux_grad[3] ≈ ad_grad[2]
  @show flux_grad[4] ≈ ad_grad[3]
  @show flux_grad[1].D_Q.weight ≈ ad_grad[4]
  @show flux_grad[1].D_K.weight ≈ ad_grad[5]
  @show flux_grad[1].D_V.weight ≈ ad_grad[6]
  @show flux_grad[1].D_Q.bias ≈ ad_grad[7]
  @show flux_grad[1].D_K.bias ≈ ad_grad[8]
  @show flux_grad[1].D_V.bias ≈ ad_grad[9]
  return
end

function check_grad(test_cuda=false)
  device = test_cuda ? gpu : identity
  F = device(Attention(Dense(10, 5), Dense(10, 5), Dense(10, 5)))
  q = device(randn(Float32, 10, 3))
  k = device(randn(Float32, 10, 3))
  v = device(randn(Float32, 10, 3))
  flux_grad = Flux.gradient((F, q, k, v)-> sum(sin, flux_attention_forward(F, q, k, v)), F, q, k, v)
  y, back = NeuralAttentionlib.attentionAD(q, k, v, NeuralAttentionlib.extract_param(F)...)
  ad_grad = back(cos.(y))
  @show flux_attention_forward(F, q, k, v) ≈ y
  compare_grad(flux_grad, ad_grad)
  return F, q, k, v
end
