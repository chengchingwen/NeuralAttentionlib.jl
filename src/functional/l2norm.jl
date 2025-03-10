using Base.Broadcast: broadcasted, instantiate
using ChainRulesCore

l2norm(x) = l2norm(1e-5, x)
function l2norm(epsilon, x)
    ϵ = convert(eltype(x), epsilon)
    x2 = sum(abs2, x; dims=1)
    return _fast_broadcast2(/, x, broadcasted(max, broadcasted(sqrt, x2), ϵ))
end

_nxdx1_div_x2(dx1, x, x2, ϵ) = -x * dx1 / max(x2, ϵ)
Base.has_fast_linear_indexing(::Broadcast.Broadcasted{<:Union{Nothing, Broadcast.BroadcastStyle}, A, typeof(_nxdx1_div_x2)}) where {A} = false

function ChainRulesCore.rrule(::typeof(l2norm), x)
    y, l2norm_pullback = rrule(l2norm, 1e-5, x)
    pullback(Ȳ) = Base.tail(l2norm_pullback(Ȳ))
    return y, pullback
end
function ChainRulesCore.rrule(::typeof(l2norm), epsilon, x)
    ϵ = convert(eltype(x), epsilon)
    x2 = sum(abs2, x; dims=1)
    y = _fast_broadcast2(/, x, broadcasted(max, broadcasted(sqrt, x2), ϵ))
    function l2norm_pullback(Ybar)
        Ȳ = unthunk(Ybar)
        dx = @_thunk begin
            # dx1 = inv.(sqrt.(x2)) .* dy
            # dx2s = sum(.- x ./ x2 .* dy ./ sqrt.(x2); dims=1)
            # dx = dx1 .+ dx2s .* x
            dx = Broadcast.materialize(instantiate(broadcasted(/, Ȳ, broadcasted(max, broadcasted(sqrt, x2), ϵ))))
            bc = instantiate(broadcasted(_nxdx1_div_x2, dx, x, x2, ϵ))
            dx2s = mapreduce(identity, +, bc; dims=1, init=zero(eltype(x)))
            dx .+= dx2s .* x
            return dx
        end
        return (NoTangent(), NoTangent(), dx)
    end
    return y, l2norm_pullback
end
