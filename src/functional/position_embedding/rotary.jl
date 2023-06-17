using Static
using ChainRulesCore

with_rotary_position_embedding(::Nothing) = with_rotary_position_embedding
with_rotary_position_embedding(dim::Integer) = Base.Fix1(with_rotary_position_embedding, dim)
with_rotary_position_embedding(::Nothing, x) = with_rotary_position_embedding(x)
with_rotary_position_embedding(x) = with_rotary_position_embedding!(copy(x))
function with_rotary_position_embedding(dim::Integer, x)
    y = copy(x)
    @static if VERSION < v"1.9"
        y′ = @view y[begin:dim, ntuple(i->Colon(), static(ndims(x)) - static(1))...]
        with_rotary_position_embedding!(y′)
    else
        y′ = y[begin:dim, ntuple(i->Colon(), static(ndims(x)) - static(1))...]
        with_rotary_position_embedding!(y′)
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

function with_rotary_position_embedding!(x)
    hidden_size = size(x, 1)
    @assert iseven(hidden_size) "rotary position embedding require the feature dim is even."
    len = size(x, 2)
    half = hidden_size >> 1
    @static if VERSION < v"1.9"
        sincos = Broadcast.broadcasted(rotary_sincos_position_embed, default_position_func(hidden_size),
                                       Val(Int32(hidden_size)), Base.OneTo{Int32}(len)',
                                       reinterpret(NTuple{2, CartesianIndex{2}}, CartesianIndices((hidden_size, len))),
                                       Val(false))
        x′ = reinterpret(NTuple{2, eltype(x)}, x)
        x′ .= _rotary.(x′, sincos)
    else
        sincos = Broadcast.broadcasted(sincos_position_embed, default_position_func(hidden_size),
                                       Val(Int32(hidden_size)), Base.OneTo{Int32}(len)',
                                       CartesianIndices((hidden_size, len)),
                                       Val(false))
        x′ = reinterpret(NTuple{2, eltype(x)}, x)
        _sincos = similar(x, (hidden_size, len))
        _sincos .= sincos
        x′ .= _rotary.(x′, reinterpret(NTuple{2, eltype(_sincos)}, _sincos))
    end
    return x
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(with_rotary_position_embedding), x)
    hidden_size = size(x, 1)
    @assert iseven(hidden_size) "rotary position embedding require the feature dim is even."
    len = size(x, 2)
    half = hidden_size >> 1
    @static if VERSION < v"1.9"
        sincos = Broadcast.broadcasted(rotary_sincos_position_embed, default_position_func(hidden_size),
                                       Val(Int32(hidden_size)), Base.OneTo{Int32}(len)',
                                       reinterpret(NTuple{2, CartesianIndex{2}}, CartesianIndices((hidden_size, len))),
                                       Val(false))
        x′ = reinterpret(NTuple{2, eltype(x)}, x)
        y = reshape(reinterpret(eltype(x), _rotary.(x′, sincos)), size(x))
    else
        sincos = Broadcast.broadcasted(sincos_position_embed, default_position_func(hidden_size),
                                       Val(Int32(hidden_size)), Base.OneTo{Int32}(len)',
                                       CartesianIndices((hidden_size, len)),
                                       Val(false))
        x′ = reinterpret(NTuple{2, eltype(x)}, x)
        _sincos = similar(x, (hidden_size, len))
        _sincos .= sincos
        y = reshape(reinterpret(eltype(x), _rotary.(x′, reinterpret(NTuple{2, eltype(_sincos)}, _sincos))), size(x))
    end
    function with_rotary_position_embedding_pullback(Ybar)
        Ȳ = unthunk(Ybar)
        ∂y = reinterpret(NTuple{2, eltype(Ȳ)}, Ȳ)
        @static if VERSION < v"1.9"
            ∂x′ = ∇_rotary.(∂y, sincos)
        else
            ∂x′ = ∇_rotary.(∂y, reinterpret(NTuple{2, eltype(_sincos)}, _sincos))
        end
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
    @static if VERSION < v"1.9"
        x = @view y[begin:dim, ntuple(i->Colon(), static(ndims(x)) - static(1))...]
    else
        x = y[begin:dim, ntuple(i->Colon(), static(ndims(x)) - static(1))...]
    end
    hidden_size = size(x, 1)
    @assert iseven(hidden_size) "rotary position embedding require the feature dim is even."
    len = size(x, 2)
    half = hidden_size >> 1
    @static if VERSION < v"1.9"
        sincos = Broadcast.broadcasted(rotary_sincos_position_embed, default_position_func(hidden_size),
                                       Val(Int32(hidden_size)), Base.OneTo{Int32}(len)',
                                       reinterpret(NTuple{2, CartesianIndex{2}}, CartesianIndices((hidden_size, len))),
                                       Val(false))
        x′ = reinterpret(NTuple{2, eltype(x)}, x)
        x′ .= _rotary.(x′, sincos)
    else
        sincos = Broadcast.broadcasted(sincos_position_embed, default_position_func(hidden_size),
                                       Val(Int32(hidden_size)), Base.OneTo{Int32}(len)',
                                       CartesianIndices((hidden_size, len)),
                                       Val(false))
        x′ = reinterpret(NTuple{2, eltype(x)}, x)
        _sincos = similar(x, (hidden_size, len))
        _sincos .= sincos
        x′ .= _rotary.(x′, reinterpret(NTuple{2, eltype(_sincos)}, _sincos))
        y[begin:dim, ntuple(i->Colon(), static(ndims(x)) - static(1))...] .= x
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
            ∂y .= ∇_rotary.(∂y, reinterpret(NTuple{2, eltype(_sincos)}, _sincos))
            ∂x[begin:dim, ntuple(i->Colon(), static(ndims(∂x)) - static(1))...] .= Ȳ′
        end
        return (NoTangent(), NoTangent(), ∂x)
    end
    return y, with_rotary_position_embedding_pullback
end

function ChainRulesCore.rrule(config::RuleConfig, ::typeof(with_rotary_position_embedding), dim::Integer)
    pullback(_) = (NoTangent(), NoTangent())
    return with_rotary_position_embedding(dim), pullback
end
