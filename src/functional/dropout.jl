# Implementing dropout with random mask.
# no need and can't use `NNlib._rng_from_array` since the `rng` is only used for generate initial seeds on CPU side
dropout(x::AbstractArray, p::Real, dims = :) = dropout(Random.default_rng(), x, p, dims)
dropout(rng::AbstractRNG, x::AbstractArray, p::Real, dims = :) = _dropout!(rng, similar(x), x, p, dims)

dropout!(x::AbstractArray, p::Real, dims = :) = dropout!(Random.default_rng(), x, x, p, dims)
dropout!(y::AbstractArray, x::AbstractArray, p::Real, dims = :) = dropout!(Random.default_rng(), y, x, p, dims)
dropout!(rng::AbstractRNG, x::AbstractArray, p::Real, dims = :) = _dropout!(rng, x, x, p, dims)
dropout!(rng::AbstractRNG, y::AbstractArray, x::AbstractArray, p::Real, dims = :) = _dropout!(rng, y, x, p, dims)

_dropout_masksize(src, ::Colon) = size(src)
_dropout_masksize(src, dims) = ntuple(d->d in dims ? size(src, d) : 1, Val(ndims(src)))
function _dropout!(rng::AbstractRNG, dst::AbstractArray, src::AbstractArray, p::Real, dims)
    scale = convert(eltype(src), inv(one(p) - p))
    m = GetIndexer(IndexerAdaptor(rng), RandomMask(p), _dropout_masksize(src, dims), scale)
    return _fast_broadcast2!(*, dst, src, m)
end

function _dropout_rrule!(rng::AbstractRNG, y::AbstractArray, x::AbstractArray, p::Real, dims)
    scale = convert(eltype(x), inv(one(p) - p))
    m = GetIndexer(IndexerAdaptor(rng), RandomMask(p), _dropout_masksize(x, dims), scale)
    _fast_broadcast2!(*, y, x, m)
    dropout_dx!(dx, dy) = _fast_broadcast2!(*, dx, dy, m)
    return y, dropout_dx!
end
function ChainRulesCore.rrule(config::RuleConfig, ::typeof(dropout), x::AbstractArray, p::Real)
    y, back = rrule(config, dropout, x, p, :)
    dropout_pullback(Ȳ) = Base.front(back(Ȳ))
    return y, dropout_pullback
end
function ChainRulesCore.rrule(config::RuleConfig, ::typeof(dropout), x::AbstractArray, p::Real, dims)
    rng = Random.default_rng()
    y, back = rrule(config, dropout, rng, x, p, dims)
    dropout_pullback(Ȳ) = (NoTangent(), back(Ȳ)[3], NoTangent(), NoTangent())
    return y, dropout_pullback
end
function ChainRulesCore.rrule(config::RuleConfig, ::typeof(dropout), rng::AbstractRNG, x::AbstractArray, p::Real)
    y, back = rrule(config, dropout, rng, x, p, :)
    dropout_pullback(Ȳ) = Base.front(back(Ȳ))
    return y, dropout_pullback
end
function ChainRulesCore.rrule(config::RuleConfig, ::typeof(dropout), rng::AbstractRNG, x::AbstractArray, p::Real, dims)
    y, dropout_dx! = _dropout_rrule!(rng, similar(x), x, p, dims)
    function dropout_pullback(Ybar)
        Ȳ = unthunk(Ybar)
        thk = @_thunk dropout_dx!(similar(x), Ȳ)
        return (NoTangent(), NoTangent(), thk, NoTangent(), NoTangent())
    end
    return y, dropout_pullback
end

struct dropoutF{R <: Union{Nothing, AbstractRNG}, P <: Union{Nothing, Real}, D} <: Function
    rng::R
    p::P
    dims::D
end
dropoutF(; rng = nothing, p = nothing, dims = :) = dropoutF(rng, p, dims)
(f::dropoutF{Nothing, Nothing})(x::AbstractArray, p::Real) = dropout(x, p, f.dims)
(f::dropoutF{Nothing})(x::AbstractArray) = dropout(x, f.p, f.dims)
(f::dropoutF{R, Nothing})(x::AbstractArray, p::Real) where R <: AbstractRNG = dropout(f.rng, x, p, f.dims)
(f::dropoutF{R})(x::AbstractArray) where R <: AbstractRNG = dropout(f.rng, x, f.p, f.dims)

function ChainRulesCore.rrule(config::RuleConfig, f::dropoutF{R, Nothing}, x::AbstractArray, p::Real) where R
    if R <: AbstractRNG
        y, back_rng = rrule(config, dropout, f.rng, x, p, f.dims)
        pullback_rng(Ȳ) = (NoTangent(), back_rng(Ȳ)[3], NoTangent())
        return y, pullback_rng
    else
        y, back = rrule(config, dropout, x, p, f.dims)
        pullback(Ȳ) = (NoTangent(), back(Ȳ)[2], NoTangent())
        return y, pullback
    end
end
function ChainRulesCore.rrule(config::RuleConfig, f::dropoutF{R}, x::AbstractArray) where R
    y, back = rrule(config, dropoutF(f.rng, nothing, f.dims), x, f.p)
    pullback(Ȳ) = Base.front(back(Ȳ))
    return y, pullback
end

struct dropoutF!{R <: Union{Nothing, AbstractRNG}, P <: Union{Nothing, Real}, D} <: Function
    rng::R
    p::P
    dims::D
end
dropoutF!(; rng = nothing, p = nothing, dims = :) = dropoutF!(rng, p, dims)
dropoutF!(f::dropoutF) = dropoutF!(f.rng, f.p, f.dims)
(f::dropoutF!{Nothing, Nothing})(x::AbstractArray, p::Real) = dropout!(x, p, f.dims)
(f::dropoutF!{Nothing})(x::AbstractArray) = dropout!(x, f.p, f.dims)
(f::dropoutF!{R, Nothing})(x::AbstractArray, p::Real) where R <: AbstractRNG = dropout!(f.rng, x, p, f.dims)
(f::dropoutF!{R})(x::AbstractArray) where R <: AbstractRNG = dropout!(f.rng, x, f.p, f.dims)


"""
    dropout(x::AbstractArray, p::Real, dims = :)
    dropout([rng::AbstractRNG,] x::AbstractArray, p::Real, dims = :)

Drop `p` percent of element and multiply the rest with `1/(1-p)`. See also [`dropout!`](@ref)
"""
dropout

"""
    dropout!([rng::AbstractRNG,] x::AbstractArray, p::Real, dims = :)
    dropout!([rng::AbstractRNG,] y::AbstractArray, x::AbstractArray, p::Real, dims = :)

Inplace version of [`dropout`](@ref), storing result in `y`. If `y` is not provided, inplace update `x`.
"""
dropout!

"""
    dropoutF(; rng = nothing, p = nothing, dims = :)

Return a dropout function with the specific arguments set.
"""
dropoutF
