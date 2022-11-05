_x_x2(x) = (x, x * x)
_x_y2(x, y) = (x, x * y)

function _normalize(inN::T, ϵ::T, x::T, sum_sum2::NTuple{2, T}) where T
    μ, s = sum_sum2 .* inN
    σ₀ = sqrt(fma(μ, -μ, s))
    σ = max(σ₀, ϵ)
    return (x - μ) / σ
end

function _rstd(inN::T, ϵ::T, sum_sum2::NTuple{2, T}) where T
    μ, s = sum_sum2 .* inN
    σ₀ = sqrt(fma(μ, -μ, s))
    σ = max(σ₀, ϵ)
    return inv(σ)
end

layer_norm(alpha, beta, x) = layer_norm(1e-5, alpha, beta, x)
function layer_norm(epsilon, alpha, beta, x)
    T = eltype(x)
    N = size(x, 1)
    ϵ = convert(T, epsilon)
    α = isnothing(alpha) ? one(T) : alpha
    β = isnothing(beta) ? zero(T) : beta
    sum_sum2 = mapreduce(_x_x2, .+, x; dims=1, init = (zero(T), zero(T)))
    return fma.(α, _normalize.(convert(T, 1//N), ϵ, x, sum_sum2), β)
end

_fma2(dy::T, dya::NTuple{2, T}, n::T, inN::T, is::T) where T = fma(fma(n, last(dya), first(dya)), inN, dy) * is
function Δlayer_norm_dx(Ȳ, ϵ, α, n, x, sum_sum2)
    T = eltype(x)
    N = size(x, 1)
    is = Broadcast.instantiate(Broadcast.broadcasted(_rstd, convert(T, 1//N), ϵ, sum_sum2))
    dy = Broadcast.instantiate(Broadcast.broadcasted(*, Ȳ, α))
    dya = mapreduce(_x_y2, .+, dy, n; dims=1, init=(zero(T), zero(T)))
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
    sum_sum2 = mapreduce(_x_x2, .+, x; dims=1, init = (zero(T), zero(T)))
    n = _normalize.(convert(T, 1//N), ϵ, x, sum_sum2)
    y = fma.(α, n, β)
    function layer_norm_pullback(Ybar)
        Ȳ = unthunk(Ybar)
        ∂α = as_bool(cα) ? NoTangent() : @thunk sum(
            Broadcast.instantiate(Broadcast.broadcasted(*, Ȳ, n));
            dims = as_bool(aα) ? _taildims(Ȳ) : :, init = zero(eltype(Ȳ))
        )
        ∂β = as_bool(cβ) ? NoTangent() : @thunk sum(Ȳ; dims = as_bool(aβ) ? _taildims(Ȳ) : :)
        ∂x = @thunk Δlayer_norm_dx(Ȳ, ϵ, α, n, x, sum_sum2)
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
        ∂α = as_bool(cα) ? NoTangent() : @thunk sum(
            Broadcast.instantiate(Broadcast.broadcasted(*, Ȳ,
                                                        Broadcast.broadcasted(_rms_norm, convert(T, 1//N), ϵ, x, sum2)));
            dims = as_bool(aα) ? _taildims(Ȳ) : :, init = zero(eltype(Ȳ))
        )
        ∂x = @thunk Δrms_layer_norm_dx(Ȳ, ϵ, α, x, sum2)
        return (NoTangent(), NoTangent(), ∂α, ∂x)
    end
    return y, rms_layer_norm_pullback
end
