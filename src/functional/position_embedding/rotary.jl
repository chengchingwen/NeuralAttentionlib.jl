using Static
using ChainRulesCore

with_rotary_position_embedding(::Nothing) = with_rotary_position_embedding
with_rotary_position_embedding(dim::Integer) = Base.Fix1(with_rotary_position_embedding, dim)
with_rotary_position_embedding(f, dim::Union{Integer, Nothing}) = with_rotary_position_embedding $ f $ dim
with_rotary_position_embedding(::Nothing, dim::Union{Integer, Nothing}) = with_rotary_position_embedding(dim)

with_rotary_position_embedding(x) = with_rotary_position_embedding(nothing, x)
with_rotary_position_embedding(::Nothing, x) = with_rotary_position_embedding(nothing, nothing, x)
function with_rotary_position_embedding(f, ::Nothing, x)
    y = copy(x)
    with_rotary_position_embedding!(f, y)
    return y
end

with_rotary_position_embedding(dim::Integer, x) = with_rotary_position_embedding(nothing, dim, x)
function with_rotary_position_embedding(f, dim::Integer, x)
    y = copy(x)
    @static if VERSION < v"1.9"
        y′ = @view y[begin:dim, ntuple(i->Colon(), static(ndims(x)) - static(1))...]
        with_rotary_position_embedding!(f, y′)
    else # array view failed to compile on gpu above v1.9, so we copy the array and copy back
        y′ = y[begin:dim, ntuple(i->Colon(), static(ndims(x)) - static(1))...]
        with_rotary_position_embedding!(f, y′)
        y[begin:dim, ntuple(i->Colon(), static(ndims(x)) - static(1))...] .= y′
    end
    return y
end

@inline function rotary_sincos_position_embed(f, hidden_size::Val, pos, nindices::NTuple{2}, normalized::Val)
    a = sincos_position_embed(f, hidden_size, pos, nindices[1], normalized)
    b = sincos_position_embed(f, hidden_size, pos, nindices[2], normalized)
    return (a, b)
end

function _rotary(x1_x2, sin_cos)
    x1, x2 = x1_x2
    sinᵢ, cosᵢ = oftype.(x1, sin_cos)
    return (x1 * cosᵢ - x2 * sinᵢ, x2 * cosᵢ + x1 * sinᵢ)
end

function ∇_rotary(∂y1_y2, sin_cos)
    ∂y1, ∂y2 = ∂y1_y2
    sinᵢ, cosᵢ = oftype.(∂y1, sin_cos)
    return (∂y1 * cosᵢ + ∂y2 * sinᵢ, ∂y2 * cosᵢ - ∂y1 * sinᵢ)
end

_rotary_apply!(x, hidden_size, len) = _rotary_apply!(default_position_func(hidden_size), x, hidden_size, len)
_rotary_apply!(::Nothing, x, hidden_size, len) = _rotary_apply!(x, hidden_size, len)
function _rotary_apply!(f, x, hidden_size, len)
    @static if VERSION < v"1.9"
        sincos = Broadcast.broadcasted(rotary_sincos_position_embed, f,
                                       Val(Int32(hidden_size)), Base.OneTo{Int32}(len)',
                                       reinterpret(NTuple{2, CartesianIndex{2}}, CartesianIndices((hidden_size, len))),
                                       Val(false))
        x′ = reinterpret(NTuple{2, eltype(x)}, x)
        x′ .= _rotary.(x′, sincos)
    else
        _sincos = Broadcast.broadcasted(sincos_position_embed, f,
                                       Val(Int32(hidden_size)), Base.OneTo{Int32}(len)',
                                       CartesianIndices((hidden_size, len)),
                                       Val(false))
        x′ = reinterpret(NTuple{2, eltype(x)}, x)
        sincos = similar(x, (hidden_size, len))
        sincos .= _sincos
        x′ .= _rotary.(x′, reinterpret(NTuple{2, eltype(sincos)}, sincos))
    end
    return x, sincos
end

with_rotary_position_embedding!(x) = with_rotary_position_embedding!(nothing, x)
function with_rotary_position_embedding!(f, x)
    hidden_size = size(x, 1)
    @assert iseven(hidden_size) "rotary position embedding require the feature dim is even."
    len = size(x, 2)
    return first(_rotary_apply!(f, x, hidden_size, len))
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(with_rotary_position_embedding), x)
    y, pullback = rrule(config, with_rotary_position_embedding, nothing, x)
    _pullback(Ȳ) = (NoTangent(), last(pullback(Ȳ)))
    return y, _pullback
end
function ChainRulesCore.rrule(config::RuleConfig, ::typeof(with_rotary_position_embedding), dim, x)
    y, pullback = rrule(config, with_rotary_position_embedding, nothing, dim, x)
    _pullback(Ȳ) = (NoTangent(), NoTangent(), last(pullback(Ȳ)))
    return y, _pullback
end
function ChainRulesCore.rrule(config::RuleConfig, ::typeof(with_rotary_position_embedding), f, ::Nothing, x)
    hidden_size = size(x, 1)
    @assert iseven(hidden_size) "rotary position embedding require the feature dim is even."
    len = size(x, 2)
    y, sincos = _rotary_apply!(f, copy(x), hidden_size, len)
    function with_rotary_position_embedding_pullback(Ybar)
        Ȳ = unthunk(Ybar)
        ∂y = reinterpret(NTuple{2, eltype(Ȳ)}, Ȳ)
        @static if VERSION < v"1.9"
            ∂x′ = ∇_rotary.(∂y, sincos)
        else
            ∂x′ = ∇_rotary.(∂y, reinterpret(NTuple{2, eltype(sincos)}, sincos))
        end
        ∂x = reshape(reinterpret(eltype(Ȳ), ∂x′), size(Ȳ))
        return (NoTangent(), ∂x)
    end
    return y, with_rotary_position_embedding_pullback
end
function ChainRulesCore.rrule(config::RuleConfig, ::typeof(with_rotary_position_embedding), f, dim::Integer, x)
    y = copy(x)
    hidden_size = dim
    @assert iseven(hidden_size) "rotary position embedding require the feature dim is even."
    len = size(y, 2)
    @static if VERSION < v"1.9"
        y′ = @view y[begin:dim, ntuple(i->Colon(), static(ndims(x)) - static(1))...]
        _, sincos = _rotary_apply!(f, y′, hidden_size, len)
    else # array view failed to compile on gpu above v1.9, so we copy the array and copy back
        y′ = y[begin:dim, ntuple(i->Colon(), static(ndims(x)) - static(1))...]
        _, sincos = _rotary_apply!(f, y′, hidden_size, len)
        y[begin:dim, ntuple(i->Colon(), static(ndims(x)) - static(1))...] .= y′
    end
    function with_rotary_position_embedding_pullback(Ybar)
        Ȳ = unthunk(Ybar)
        ∂x = copy(Ȳ)
        @static if VERSION < v"1.9"
            Ȳ′ = @view ∂x[begin:dim, ntuple(i->Colon(), static(ndims(∂x)) - static(1))...]
            ∂y = reinterpret(NTuple{2, eltype(Ȳ′)}, Ȳ′)
            ∂y .= ∇_rotary.(∂y, sincos)
        else
            Ȳ′ = ∂x[begin:dim, ntuple(i->Colon(), static(ndims(∂x)) - static(1))...]
            ∂y = reinterpret(NTuple{2, eltype(Ȳ′)}, Ȳ′)
            ∂y .= ∇_rotary.(∂y, reinterpret(NTuple{2, eltype(sincos)}, sincos))
            ∂x[begin:dim, ntuple(i->Colon(), static(ndims(∂x)) - static(1))...] .= Ȳ′
        end
        return (NoTangent(), NoTangent(), ∂x)
    end
    return y, with_rotary_position_embedding_pullback
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(with_rotary_position_embedding), dim::Union{Integer, Nothing})
    pullback(_) = (NoTangent(), NoTangent())
    return with_rotary_position_embedding(dim), pullback
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(with_rotary_position_embedding), f, dim::Union{Integer, Nothing})
    pullback(_) = (NoTangent(), NoTangent(), NoTangent())
    return with_rotary_position_embedding(f, dim), pullback
end
