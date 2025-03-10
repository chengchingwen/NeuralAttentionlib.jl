_n_m_s(x) = (one(Int32), x, zero(x))
_n_m_s(x::Float16) = (one(Int32), Float32(x), zero(Float32))

function _chan_update(a, b)
    n1, m1, s1 = a
    n2, m2, s2 = b
    iszero(n1) && return b
    iszero(n2) && return a
    n = n1 + n2
    delta = m1 - m2
    n2′ = oftype(delta, n2) / n
    m = fma(m1, oftype(delta, n1) / n, m2 * n2′) #(m1 * n1 + m2 * n2) / n
    s = fma(delta^2, n1 * n2′, s1 + s2) #s1 + s2 + delta^2 * n1 * n2′
    return (n, m, s)
end

_x_y2(x, y) = (x, x * y)

function _normalize(inN::T, ϵ::T, x::T, mean_M2_::Tuple{Int32, T, T}) where T
    _, μ, M2 = mean_M2_
    v = M2 * inN
    σ₀ = sqrt(v)
    σ = max(σ₀, ϵ)
    return (x - μ) / σ
end

function _normalize(inN::Float16, ϵ::Float16, x::Float16, mean_M2_::Tuple{Int32, Float32, Float32})
    _, μ, M2 = mean_M2_
    v = Float16(M2) * inN
    σ₀ = sqrt(v)
    σ = max(σ₀, ϵ)
    return Float16(Float32(x) - μ) / σ
end

function _rstd(inN::T, ϵ::T, mean_M2_::Tuple{Int32, T, T}) where T
    _, μ, M2 = mean_M2_
    v = M2 * inN
    σ₀ = sqrt(v)
    σ = max(σ₀, ϵ)
    return inv(σ)
end

function _rstd(inN::Float16, ϵ::Float16, mean_M2_::Tuple{Int32, Float32, Float32})
    _, μ, M2 = mean_M2_
    v = Float16(M2) * inN
    σ₀ = sqrt(v)
    σ = max(σ₀, ϵ)
    return inv(σ)
end

_mean_M2_init(::Type{T}) where T = (zero(Int32), zero(T), zero(T))
_mean_M2_init(::Type{Float16}) = _mean_M2_init(Float32)

layer_norm(alpha, beta, x) = layer_norm(1e-5, alpha, beta, x)
function layer_norm(epsilon, alpha, beta, x)
    T = eltype(x)
    N = size(x, 1)
    ϵ = convert(T, epsilon)
    α = isnothing(alpha) ? one(T) : alpha
    β = isnothing(beta) ? zero(T) : beta
    mean_M2_ = mapreduce(_n_m_s, _chan_update, x; dims=1, init = _mean_M2_init(T))
    return fma.(α, _normalize.(convert(T, 1//N), ϵ, x, mean_M2_), β)
end

_dya_init(::Type{T}) where T = (zero(T), zero(T))
_dya_init(::Type{Float16}) = _dya_init(Float32)

_fma2(dy::T, dya::NTuple{2, T}, n::T, inN::T, is::T) where T = fma(fma(n, last(dya), first(dya)), inN, dy) * is
_fma2(dy::Float16, dya::NTuple{2, Float32}, n::Float16, inN::Float16, is::Float16) =
    fma(fma(n, Float16(last(dya)), Float16(first(dya))), inN, dy) * is
function Δlayer_norm_dx(Ȳ, ϵ, α, n, x, mean_M2_)
    T = eltype(x)
    N = size(x, 1)
    is = Broadcast.instantiate(Broadcast.broadcasted(_rstd, convert(T, 1//N), ϵ, mean_M2_))
    dy = Broadcast.instantiate(Broadcast.broadcasted(*, Ȳ, α))
    dya = mapreduce(_x_y2, .+, dy, n; dims = 1, init = _dya_init(T))
    ∂x = _fma2.(dy, dya, n, -convert(T, 1//N), is)
    return ∂x
end

_taildims(Ȳ) = Base.tail(ntuple(identity, Val(ndims(Ȳ))))

function ChainRulesCore.rrule(::typeof(layer_norm), alpha, beta, x)
    y, pullback = rrule(layer_norm, 1e-5, alpha, beta, x)
    layer_norm_pullback(Ȳ) = (NoTangent(), last_n(pullback(Ȳ), static(3))...)
    return y, layer_norm_pullback
end
function ChainRulesCore.rrule(::typeof(layer_norm), epsilon, alpha, beta, x)
    T = eltype(x)
    N = size(x, 1)
    ϵ = convert(T, epsilon)
    cα = static(isnothing(alpha))
    cβ = static(isnothing(beta))
    aα = static(alpha isa AbstractArray)
    aβ = static(beta isa AbstractArray)
    α = as_bool(cα) ? one(T) : alpha
    β = as_bool(cβ) ? zero(T) : beta
    mean_M2_ = mapreduce(_n_m_s, _chan_update, x; dims=1, init = _mean_M2_init(T))
    n = _normalize.(convert(T, 1//N), ϵ, x, mean_M2_)
    y = fma.(α, n, β)
    function layer_norm_pullback(Ybar)
        Ȳ = unthunk(Ybar)
        ∂α = as_bool(cα) ? NoTangent() : @_thunk sum(
            Broadcast.instantiate(Broadcast.broadcasted(*, Ȳ, n));
            dims = as_bool(aα) ? _taildims(Ȳ) : :, init = zero(eltype(Ȳ))
        )
        ∂β = as_bool(cβ) ? NoTangent() : @_thunk sum(Ȳ; dims = as_bool(aβ) ? _taildims(Ȳ) : :)
        ∂x = @_thunk Δlayer_norm_dx(Ȳ, ϵ, α, n, x, mean_M2_)
        return (NoTangent(), NoTangent(), ∂α, ∂β, ∂x)
    end
    return y, layer_norm_pullback
end

_rrms(iN, ϵ, sum2) = inv(max(sqrt(sum2 * iN), ϵ))

function _rms_norm(iN, ϵ, x, sum2)
    rms₀ = sqrt(sum2 * iN)
    rms = max(rms₀, ϵ)
    return x / rms
end

rms_layer_norm(alpha, x) = rms_layer_norm(1e-5, alpha, x)
function rms_layer_norm(epsilon, alpha, x)
    T = eltype(x)
    N = size(x, 1)
    ϵ = convert(T, epsilon)
    α = isnothing(alpha) ? one(T) : alpha
    sum2 = sum(abs2, x; dims=1, init = zero(T))
    return α .* _rms_norm.(convert(T, 1//N), ϵ, x, sum2)
end

_fma1(ϵ, dy, da, x, sum2, irms) = fma(-da / max(sum2, ϵ), x, dy) * irms
function Δrms_layer_norm_dx(Ȳ, ϵ, α, x, sum2)
    T = eltype(x)
    N = size(x, 1)
    dy = Broadcast.instantiate(Broadcast.broadcasted(*, Ȳ, α))
    irms = Broadcast.instantiate(Broadcast.broadcasted(_rrms, convert(T, 1//N), ϵ, sum2))
    da = mapreduce(*, +, x, dy; dims = 1, init = zero(T))
    ∂x = _fma1.(ϵ, dy, da, x, sum2, irms)
    return ∂x
end

function ChainRulesCore.rrule(::typeof(rms_layer_norm), alpha, x)
    y, pullback = rrule(rms_layer_norm, 1e-5, alpha, x)
    rms_layer_norm_pullback(Ȳ) = (NoTangent(), last_n(pullback(Ȳ), static(2))...)
    return y, rms_layer_norm_pullback
end
function ChainRulesCore.rrule(::typeof(rms_layer_norm), epsilon, alpha, x)
    T = eltype(x)
    N = size(x, 1)
    ϵ = convert(T, epsilon)
    cα = static(isnothing(alpha))
    aα = static(alpha isa AbstractArray)
    α = as_bool(cα) ? one(T) : alpha
    sum2 = sum(abs2, x; dims=1, init = zero(T))
    y = α .* _rms_norm.(convert(T, 1//N), ϵ, x, sum2)
    function rms_layer_norm_pullback(Ybar)
        Ȳ = unthunk(Ybar)
        ∂α = as_bool(cα) ? NoTangent() : @_thunk sum(
            Broadcast.instantiate(Broadcast.broadcasted(*, Ȳ,
                                                        Broadcast.broadcasted(_rms_norm, convert(T, 1//N), ϵ, x, sum2)));
            dims = as_bool(aα) ? _taildims(Ȳ) : :, init = zero(eltype(Ȳ))
        )
        ∂x = @_thunk Δrms_layer_norm_dx(Ȳ, ϵ, α, x, sum2)
        return (NoTangent(), NoTangent(), ∂α, ∂x)
    end
    return y, rms_layer_norm_pullback
end
