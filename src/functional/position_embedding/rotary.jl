using ChainRulesCore

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

function with_rotary_position_embedding(x)
    hidden_size = size(x, 1)
    @assert iseven(hidden_size) "rotary position embedding require the feature dim is even."
    len = size(x, 2)
    half = hidden_size >> 1
    sincos = Broadcast.broadcasted(rotary_sincos_position_embed, default_position_func(hidden_size),
                                   Val(Int32(hidden_size)), Base.OneTo{Int32}(len)',
                                   reinterpret(reshape, NTuple{2, CartesianIndex{2}},
                                               reshape(CartesianIndices((hidden_size, len)), 2, half, len)),
                                   Val(false))
    x′ = reinterpret(reshape, NTuple{2, eltype(x)}, reshape(x, 2, half, Base.tail(size(x))...))
    y = _rotary.(x′, sincos)
    return reshape(reinterpret(reshape, eltype(x), y), size(x))
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(with_rotary_position_embedding), x)
    hidden_size = size(x, 1)
    @assert iseven(hidden_size) "rotary position embedding require the feature dim is even."
    len = size(x, 2)
    half = hidden_size >> 1
    sincos = Broadcast.broadcasted(rotary_sincos_position_embed, default_position_func(hidden_size),
                                   Val(Int32(hidden_size)), Base.OneTo{Int32}(len)',
                                   reinterpret(reshape, NTuple{2, CartesianIndex{2}},
                                               reshape(CartesianIndices((hidden_size, len)), 2, half, len)),
                                   Val(false))
    x′ = reinterpret(reshape, NTuple{2, eltype(x)}, reshape(x, 2, half, Base.tail(size(x))...))
    y = reshape(reinterpret(reshape, eltype(x), _rotary.(x′, sincos)), size(x))
    function with_rotary_position_embedding_pullback(Ybar)
        Ȳ = unthunk(Ybar)
        ∂y = reinterpret(reshape, NTuple{2, eltype(Ȳ)}, reshape(Ȳ, 2, half, Base.tail(size(Ȳ))...))
        ∂x′ = ∇_rotary.(∂y, sincos)
        ∂x = reshape(reinterpret(reshape, eltype(Ȳ), ∂x′), size(Ȳ))
        return (NoTangent(), ∂x)
    end
    return y, with_rotary_position_embedding_pullback
end
