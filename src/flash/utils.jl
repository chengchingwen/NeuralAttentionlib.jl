using CUDA
using CUDA.WMMA

@generated function tuplesplit(t::NTuple{N}, ::Val{M}) where {N, M}
    expr = Expr(:tuple)
    for indices in Iterators.partition(1:N, M)
        ti = Expr(:tuple)
        for idx in indices
            push!(ti.args, :(t[$idx]))
        end
        push!(expr.args, ti)
    end
    return quote
        @inbounds $expr
    end
end

@generated function tuplesplitp(t::NTuple{N}, ::Val{M}) where {N, M}
    N % M == 0 || error("Cannot split tuple of length $N into $M pieces")
    n = cld(N, M)
    return :(tuplesplit(t, Val{$n}()))
end

tuplejoin(a::Tuple, b::Tuple, c::Tuple...) = tuplejoin((a..., b...), c...)
tuplejoin(a::Tuple, b::Tuple) = (a..., b...)

@generated function _fast_mul(x, ::Val{V}) where V
    ispow2(V) || error("No fast mod for $V")
    v = Int32(trailing_zeros(V))
    return isone(V) ? :x : :(x << $v)
end
@generated function _fast_fld(x, ::Val{V}) where V
    ispow2(V) || error("No fast mod for $V")
    v = Int32(trailing_zeros(V))
    return isone(V) ? :x : :(x >> $v)
end
@generated function _fast_mod(x, ::Val{V}) where V
    ispow2(V) || error("No fast mod for $V")
    v = Int32(V) - 1i32
    return :(x & $v)
end
_fast_fldmod(x, v::Val{V}) where V = (_fast_fld(x, v), _fast_mod(x, v))

@generated function _uint(::Type{T}) where T
    s = sizeof(T)
    if s == 1
        return UInt8
    elseif s == 2
        return UInt16
    elseif s == 4
        return UInt32
    elseif s == 8
        return UInt64
    else
        error("no corresponding unsigned type for $T")
    end
end

@generated function _pack_4byte(x::NTuple{N, UT}) where {N, T, UT <: Union{T, VecElement{T}}}
    nb = N * sizeof(T)
    n, r = fldmod(nb, 4)
    x′ = Expr(:tuple)
    for i = 1:N
        xi = UT <: VecElement ? :(x[$i].value) : :(x[$i])
        xi = :(reinterpret($(_uint(T)),  $xi))
        push!(x′.args, xi)
    end
    expr = Expr(:tuple)
    for i = 1:n
        push!(expr.args, :(reinterpret(UInt32, Vec{4, UInt8}(xs[$i]))))
    end
    if !iszero(r)
        rest = Expr(:tuple, Expr(:..., :(xs[$(n+1)])), [:(zero(T)) for _ in 1:(4 - r)]...)
        push!(expr.args, :(reinterpret(UInt32, Vec{4, UInt8}($rest))))
    end
    return quote
        @inbounds begin
            x′ = $x′
            xs = tuplesplit(WMMA.flatten(reinterpret(Vec{$nb, UInt8}, Vec(x′)).data), Val(4))
            $expr
        end
    end
end

@generated function _unpack_4byte(::Type{NTuple{N, UT}}, xs::NTuple{M, UInt32}) where {N, M, T, UT <: Union{T, VecElement{T}}}
    expr = Expr(:tuple)
    for i = 1:N
        xi = :(reinterpret($T, reinterpret($(_uint(T)), Vec(xs[$i]))))
        xi = UT <: VecElement ? :(VecElement{$T}($xi)) : xi
        push!(expr.args, xi)
    end
    return quote
        @inbounds begin
            xs = tuplesplit(WMMA.flatten(reinterpret(Vec{$(4M), UInt8}, Vec(xs)).data), Val{$(sizeof(T))}())
            return $expr
        end
    end
end

@generated function _shfl(shfl_op, mask, vals::NTuple{N, UT}, src) where {N, T, UT <: Union{T, VecElement{T}}}
    nb = N * sizeof(T)
    n = cld(nb, 4)
    expr = Expr(:tuple)
    for i = 1:n
        push!(expr.args, :(shfl_op(mask, xs[$i], src)::UInt32))
    end
    return quote
        @inbounds begin
            xs = _pack_4byte(vals)
            xs = $expr
            return _unpack_4byte(NTuple{N, UT}, xs)
        end
    end
end
_shfl_sync(mask, vals::NTuple{N, UT}, src) where {N, T, UT <: Union{T, VecElement{T}}} = _shfl(shfl_sync, mask, vals, src)
_shfl_sync(mask, vals::F, src) where F <: Fragment = F(_shfl_sync(mask, vals.x, src))
_shfl_xor_sync(mask, vals::NTuple{N, UT}, src) where {N, T, UT <: Union{T, VecElement{T}}} = _shfl(shfl_xor_sync, mask, vals, src)
_shfl_xor_sync(mask, vals::F, src) where F <: Fragment = F(_shfl_xor_sync(mask, vals.x, src))

@generated function _dot(::Type{T}, a::NTuple{N, MT}, b::NTuple{N, MT}, c::Union{T, VecElement{T}, Nothing} = nothing) where {T, N, MT}
    ab = Expr(:tuple)
    for i = 1:N
        push!(ab.args, MT <: VecElement ? :((a[$i].value * b[$i].value)) : :(a[$i] * b[$i]))
    end
    expr = :($T(ab[1]))
    for i = 2:N
        expr = Expr(:call, :(+), expr, :($T(ab[$i])))
    end
    if !(c <: Nothing)
        expr = Expr(:call, :(+), c <: VecElement ? :(c.value) : :c, expr)
    end
    if c <: VecElement
        expr = :(VecElement{$T}($expr))
    end
    return quote
        @inbounds begin
            ab = $ab
            return $expr
        end
    end
end
