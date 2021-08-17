"""
    AxesConstrain

Abstract type for define mask's constrain. return a tuple of concrete subtype of AxesConstrain.
"""
abstract type AxesConstrain end

struct All1Constrain <: AxesConstrain
    from::Int
    n::Int
end

struct DimConstrain <: AxesConstrain
    dim::Int
    val::Int
end

struct NDimConstrain <: AxesConstrain
    n::Int
    least::Bool
end
NDimConstrain(n) = NDimConstrain(n, false)

@noinline thrdm(s) = throw(DimensionMismatch("arrays could not be broadcast to a common size; $s"))

axislength(x::Integer) = x
axislength(x) = length(x)

function check_constrain(c::All1Constrain, x)
    foreach(c.from:length(x)) do i
        if axislength(x[i]) != 1
            thrdm("mask require $i-th dimension to be 1, but get $(axislength(x[i]))")
        end
    end
    return x
end

function check_constrain(c::DimConstrain, x)
    if c.dim > 0
        dim = c.dim
        xi = length(x) ≥ dim ? axislength(x[dim]) : 1
        xi != c.val && thrdm("mask require $dim-th dimension to be $(c.val), but get $xi")
    else
        dim = length(x) + c.dim + 1
        xi = axislength(x[dim])
        xi != c.val && thrdm("mask require $dim-th dimension to be $(c.val), but get $xi")
    end
    return x
end

function check_constrain(c::NDimConstrain, x)
    cmp = c.least ? (≥) : (==)
    cmp(length(x), c.n) || thrdm("mask require ndims(A) $(cmp) $(c.n)")
    return x
end

check_constrain(cs::Tuple{}, x) = x
check_constrain(cs::Tuple, x) = check_constrain(Base.tail(cs), check_constrain(cs[1], x))

Base.iterate(c::AxesConstrain) = (c, nothing)
Base.iterate(c::AxesConstrain, ::Nothing) = nothing


@inline merge_constrain(cs1, cs2, cs...) = merge_constrain(merge_constrain(cs1, cs2), cs...)
@inline merge_constrain(::Tuple{}, ::Tuple{}) = ()
@inline merge_constrain(cs1::Tuple, ::Tuple{}) = cs1
@inline merge_constrain(::Tuple{}, cs2::Tuple) = cs2
@inline function merge_constrain(cs1::Tuple, cs2::Tuple)
    ndc = _merge_ndim_c(cs1[1], cs2[1])
    dc = _merge_dim_c(Base.tails(cs1, cs2)...)
    return (ndc, dc...)
end

_merge_dim_c(::Tuple{},::Tuple{}) = ()
_merge_dim_c(a::Tuple, ::Tuple{}) = a
_merge_dim_c(::Tuple{}, b::Tuple) = b
_merge_dim_c(a::Tuple{All1Constrain}, b::Tuple{All1Constrain}) = All1Constrain(min(a.from, b.from), max(a.n, b.n))

_merge_post_dim_c(::Tuple{}, ::Tuple{}) = ()
_merge_post_dim_c(::Tuple{}, b::Tuple) = b
_merge_post_dim_c(a::Tuple, ::Tuple{}) = a
function _merge_post_dim_c(a::Tuple, b::Tuple)::Tuple{Vararg{DimConstrain}}
    a1 = a[1]
    b1 = b[1]
    if a1.dim < b1.dim
        return (a1, _merge_post_dim_c(Base.tail(a), b)...)
    elseif a1.dim > b1.dim
        return (b1, _merge_post_dim_c(a, Base.tail(b))...)
    else
        a1.val != b1.val && thrdm("mask require $(a1.dim)-th dimension to be both $(a1.val) and $(b1.val)")
        return (a1, _merge_post_dim_c(Base.tails(a, b)...)...)
    end
end

function _merge_dim_c(a::Tuple{DimConstrain, Vararg{DimConstrain}}, b::Tuple{DimConstrain, Vararg{DimConstrain}})
    a1 = a[1]
    b1 = b[1]
    if (a1.dim > 0) == (b1.dim > 0)
        return _merge_post_dim_c(a, b)
    else
        if a1.dim < 0
            n = b[end].dim
            return _merge_post_dim_c(ntuple(i->DimConstrain(a[i].dim+n+1, a[i].val), length(a)), b)
        else
            n = a[end].dim
            return _merge_post_dim_c(a, ntuple(i->DimConstrain(b[i].dim+n+1, b[i].val), length(b)))
        end
    end
end

_merge_dim_c(a::Tuple{DimConstrain, Vararg{DimConstrain}}, b::Tuple{All1Constrain}) = _merge_dim_c(b, a)
function _merge_dim_c(a::Tuple{All1Constrain}, b::Tuple{DimConstrain, Vararg{DimConstrain}})
    c = a[]
    offset = c.from - 1
    return _merge_dim_c(ntuple(i->DimConstrain(i+offset, 1), b[end].dim - offset), b)
end

@inline function _merge_ndim_c(a, b)
    a.least || b.least || thrdm("mask require both ndims(A) == $(a.n) and ndim(A) == $(b.n)")
    n = max(a.n, b.n)
    a.least || a.n == n || thrdm("mask require both ndims(A) == $(a.n) and ndims(A) ≥ $(b.n)")
    b.least || b.n == n || thrdm("mask require both ndims(A) == $(b.n) and ndims(A) ≥ $(a.n)")
    return NDimConstrain(n, a.least == b.least)
end
