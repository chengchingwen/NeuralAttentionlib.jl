using Static
using ChainRulesCore

with_rotary_position_embedding(::Nothing) = with_rotary_position_embedding
with_rotary_position_embedding(dim::Integer) = Base.Fix1(with_rotary_position_embedding, dim)
with_rotary_position_embedding(::Nothing, x) = with_rotary_position_embedding(x)
with_rotary_position_embedding(x) = with_rotary_position_embedding!(copy(x))
function with_rotary_position_embedding(dim::Integer, x)
    y = copy(x)
    y′ = @view y[begin:dim, ntuple(i->Colon(), static(ndims(x)) - static(1))...]
    with_rotary_position_embedding!(y′)
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

function with_rotary_position_embedding!(x)
    hidden_size = size(x, 1)
    @assert iseven(hidden_size) "rotary position embedding require the feature dim is even."
    len = size(x, 2)
    half = hidden_size >> 1
    sincos = Broadcast.broadcasted(rotary_sincos_position_embed, default_position_func(hidden_size),
                                   Val(Int32(hidden_size)), Base.OneTo{Int32}(len)',
                                   reinterpret(NTuple{2, CartesianIndex{2}}, CartesianIndices((hidden_size, len))),
                                   Val(false))
    x′ = reinterpret(NTuple{2, eltype(x)}, x)
    x′ .= _rotary.(x′, sincos)
    return x
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(with_rotary_position_embedding), x)
    hidden_size = size(x, 1)
    @assert iseven(hidden_size) "rotary position embedding require the feature dim is even."
    len = size(x, 2)
    half = hidden_size >> 1
    sincos = Broadcast.broadcasted(rotary_sincos_position_embed, default_position_func(hidden_size),
                                   Val(Int32(hidden_size)), Base.OneTo{Int32}(len)',
                                   reinterpret(NTuple{2, CartesianIndex{2}}, CartesianIndices((hidden_size, len))),
                                   Val(false))
    x′ = reinterpret(NTuple{2, eltype(x)}, x)
    y = reshape(reinterpret(eltype(x), _rotary.(x′, sincos)), size(x))
    function with_rotary_position_embedding_pullback(Ybar)
        Ȳ = unthunk(Ybar)
        ∂y = reinterpret(NTuple{2, eltype(Ȳ)}, Ȳ)
        ∂x′ = ∇_rotary.(∂y, sincos)
        ∂x = reshape(reinterpret(eltype(Ȳ), ∂x′), size(Ȳ))
        return (NoTangent(), ∂x)
    end
    return y, with_rotary_position_embedding_pullback
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(with_rotary_position_embedding), ::Nothing, x)
    y, pullback = rrule(config, with_rotary_position_embedding, x)
    _pullback(Ȳ) = (NoTangent(), NoTangent(), pullback(Ȳ)[2])
    return y, _pullback
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(with_rotary_position_embedding), dim::Integer, x)
    y = copy(x)
    x = @view y[begin:dim, ntuple(i->Colon(), static(ndims(x)) - static(1))...]
    hidden_size = size(x, 1)
    @assert iseven(hidden_size) "rotary position embedding require the feature dim is even."
    len = size(x, 2)
    half = hidden_size >> 1
    sincos = Broadcast.broadcasted(rotary_sincos_position_embed, default_position_func(hidden_size),
                                   Val(Int32(hidden_size)), Base.OneTo{Int32}(len)',
                                   reinterpret(NTuple{2, CartesianIndex{2}}, CartesianIndices((hidden_size, len))),
                                   Val(false))
    x′ = reinterpret(NTuple{2, eltype(x)}, x)
    x′ .= _rotary.(x′, sincos)
    function with_rotary_position_embedding_pullback(Ybar)
        Ȳ = unthunk(Ybar)
        ∂x = copy(Ȳ)
        Ȳ′ = @view ∂x[begin:dim, ntuple(i->Colon(), static(ndims(∂x)) - static(1))...]
        ∂y = reinterpret(NTuple{2, eltype(Ȳ′)}, Ȳ′)
        ∂y .= ∇_rotary.(∂y, sincos)
        return (NoTangent(), NoTangent(), ∂x)
    end
    return y, with_rotary_position_embedding_pullback
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(with_rotary_position_embedding), dim::Integer)
    pullback(_) = (NoTangent(), NoTangent())
    return with_rotary_position_embedding(dim), pullback
end
