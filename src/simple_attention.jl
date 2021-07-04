using LinearAlgebra

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

function attentionAD_navie(Q, K, V, W_Q, W_K, W_V, b_Q, b_K, b_V)
  HQ = W_Q * Q .+ b_Q
  HK = W_K * K .+ b_K
  HV = W_V * V .+ b_V
  
  A = HK' * HQ
  d_k = size(HK, 1)
  s = convert(eltype(A), √d_k)
  C, C_idx = findmax(A, dims=1)
  E = Broadcast.instantiate(@~ exp.((A .- C) ./ s))
  Es = sum(E, dims=1)
  S = E ./ Es
  Broadcast.instantiate(S)
  Y = HV * S
  return Y, function (dY)
    dS = HV' * dY
    dHV = dY * S'
    _Es = Broadcast.instantiate(@~(dS .* E ./ (.- Es .^ 2)))
    dEs = sum(_Es, dims=1)
    dE = Broadcast.instantiate(@~ dEs .+ (dS ./ Es))
    dT = Broadcast.instantiate(@~ dE .* E)
    dC = sum(dT, dims=1)
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

attentionAD(arg...) = attentionAD_prealloc(arg...)
function attentionAD_prealloc(Q, K, V, W_Q, W_K, W_V, b_Q, b_K, b_V, ::Val{Back} = Val(true)) where Back
  HQ = similar(W_Q, size(W_Q, 1), size(Q, 2)) # @b
  HK = similar(W_K, size(W_K, 1), size(K, 2)) # @b
  HV = similar(W_V, size(W_V, 1), size(V, 2)) # @b
  A = similar(HK, size(HK, 2), size(HQ, 2))
  C = similar(A, 1, size(A, 2))
  C_idx = similar(C, CartesianIndex{2}) # @b
  if Back
    S = similar(A)
  else
    S = A
  end
  Y = similar(HV, size(HV, 1), size(A, 2))
  return attentionAD_kernel((HQ, HK, HV, A, C, C_idx, S, Y), Q, K, V, W_Q, W_K, W_V, b_Q, b_K, b_V)
end

function attentionAD_kernel((HQ, HK, HV, A, C, C_idx, S, Y), Q, K, V, W_Q, W_K, W_V, b_Q, b_K, b_V)
  #HQ = W_Q * Q .+ b_Q
  #HQ = similar(size(W_Q, 1), size(Q, 2))
  HQ .= b_Q
  mul!(HQ, W_Q, Q, true, true)
  #HK = W_K * K .+ b_K
  #HK = similar(size(W_K, 1), size(K, 2))
  HK .= b_K
  mul!(HK, W_K, K, true, true)
  #HV = W_V * V .+ b_V
  #HV = similar(size(W_V, 1), size(V, 2))
  HV .= b_V
  mul!(HV, W_V, V, true, true)

  # A = HK' * HQ
  mul!(A, HK', HQ)
  
  #As = A ./ √d_k
  #C, C_idx = findmax(As, dims=1) #maximum(As, dims=1)
  #T = As .- C  
  #E = exp.(As .- C)

  d_k = size(HK, 1)
  s = convert(eltype(A), √d_k)
  #mul!(A, HK', HQ, inv(s), false)
  
  #C, C_idx = findmax(A, dims=1)
  findmax!(C, C_idx, A)
  
  #E = Broadcast.instantiate(@~ exp.((A .- C) ./ s))
  E = A#similar(A)
  Broadcast.materialize!(E, @~ exp.((A .- C) ./ s))
  
  #Es = sum(E, dims=1)
  Es = C
  sum!(Es, E)
  
  #S = Broadcast.instantiate(E ./ Es)
  Broadcast.materialize!(S, @~ E ./ Es)

  #Y = HV * S
  mul!(Y, HV, S)
  return Y, function (dY)
    # forward buffers: HV, S, E, Es, C_idx, HK, HQ, #Q, K, V
    #dS = HV' * dY
    dS = similar(dY, size(HV, 2), size(dY, 2))
    mul!(dS, HV', dY)

    #dHV = dY * S' # @S
    dHV = similar(dY, size(dY, 1), size(S, 1))
    mul!(dHV, dY, S')
    
    _Es = @~(dS .* E ./ (.- Es .^ 2))
    dEs = sum(_Es, dims=1)
    
    #dE2 = dS ./ Es
    #dE1 = repeat(dEs, size(E, 1))
    #dE = dEs .+ dE2
    dE = Broadcast.instantiate(@~ dEs .+ (dS ./ Es)) # @Es
    dT = Broadcast.instantiate(@~ dE .* E)
    dC = sum(dT, dims=1)
    #dAs2 = dT
    #dAs1 = setindex!(zero(As), dC, C_idx) # zero(As)[C_idx] = dC
    #dAs = dAs1 + dAs2
    #dAs = dAs1 .+ (dE .* E)
    #dA = dAs ./ √d_k
    dAs1 = setindex!(zero(A), dC, C_idx) # @C_idx
    dA = (dAs1 .+ (dE .* E)) ./ s # @E
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
