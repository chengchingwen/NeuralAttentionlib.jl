using Static
using ChainRulesCore

default_position_func(hidden_size) = Base.Fix1(default_position_func, static(Int32(hidden_size)))
@inline function default_position_func(hidden_size, i)
    j = (0x1 - i) << 0x3 #8 * (1 - i)
    return 1f1 ^ (Float32(j) / hidden_size)
end

function sincos_position_embed(f, ::Val{hidden_size}, pos, indices, ::Val{normalized}) where {hidden_size, normalized}
    idx = Int32(first(Tuple(indices)))
    i = (idx + 0x1) >> 0x1
    _pos = pos - 0x1
    w = _pos * f(i)
    y = idx & 0x1 > 0x0 ? sin(w) : cos(w)
    if normalized
        half = hidden_size >> 0x1
        if half << 0x1 != hidden_size
            r = sin(_pos * f(half + 0x1))
            return y * inv(sqrt(Float32(half + r)))
        else
            return y * inv(sqrt(Float32(half)))
        end
    else
        return y
    end
end

function compute_sincos_position_embedding!(pos_func, hidden_size::Integer, normalized::Bool, y, x)
    y .= sincos_position_embed.(pos_func, Val(Int32(hidden_size)), x, CartesianIndices(y), Val(normalized))
    return y
end

get_sincos_position_embeddings(hidden_size::Integer, normalized::Bool, x) =
    get_sincos_position_embeddings(default_position_func(hidden_size), hidden_size, normalized, x)
function get_sincos_position_embeddings(pos_func, hidden_size::Integer, normalized::Bool, len::Integer)
    y = Matrix{Float32}(undef, hidden_size, len)
    compute_sincos_position_embedding!(pos_func, hidden_size, normalized, y, Base.OneTo{Int32}(len)')
    return y
end
function get_sincos_position_embeddings(pos_func, hidden_size::Integer, normalized::Bool, x)
    len = size(x, 2)
    y = reshape(similar(x, hidden_size, len), Val(ndims(x)))
    compute_sincos_position_embedding!(pos_func, hidden_size, normalized, y, Base.OneTo{Int32}(len)')
    return y
end
function get_sincos_position_embeddings(pos_func, hidden_size::Integer, normalized::Bool, x::AbstractArray{<:Integer})
    y = similar(x, Float32, hidden_size, size(x)...)
    compute_sincos_position_embedding!(pos_func, hidden_size, normalized, y, reshape(x, 1, size(x)...))
    return y
end

ChainRulesCore.@non_differentiable get_sincos_position_embeddings(args...)
