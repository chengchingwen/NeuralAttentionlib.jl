@init @require FiniteDifferences="26cc04aa-876d-5657-8c51-4c34ba976000" begin
    using .FiniteDifferences

    function FiniteDifferences.to_vec(X::CollapsedDimsArray)
        x_vec, back = to_vec(collapseddims(X))
        s = size(parent(X))
        ni = X.ni
        nj = X.nj
        function CollapsedDimsArray_from_vec(x_vec)
            return CollapsedDimsArray(reshape(back(x_vec), s), ni, nj)
        end
        return x_vec, CollapsedDimsArray_from_vec
    end
end

@inline function _sumbatch(ca::CollapsedDimsArray)
    offset = static(ndims(parent(ca))) - ca.nj
    dims = ntuple(identity, ca.nj) .+ offset
    y = sum(parent(ca); dims = dims)
    return CollapsedDimsArray(y, ca.ni, ca.nj)
end

using ChainRulesCore
using ChainRulesCore: NoTangent, @thunk
import ChainRulesCore: ProjectTo
function ChainRulesCore.rrule(::Type{CollapsedDimsArray}, x, dims, ni, nj)
    s = size(x)
    function CollapsedDimsArray_pullback(Ybar)
        Ȳ = unthunk(Ybar)
        ∂x = @thunk begin
            tmp = unwrap_collapse(Ȳ)
            size(tmp) == s ? tmp : reshape(tmp, s)
        end
        return (NoTangent(), ∂x, NoTangent(), NoTangent(), NoTangent())
    end
    CollapsedDimsArray_pullback(::ZeroTangent) = (NoTangent(), ZeroTangent(), NoTangent(), NoTangent(), NoTangent())
    return CollapsedDimsArray(x, dims, ni, nj), CollapsedDimsArray_pullback
end

function ChainRulesCore.rrule(::Type{CollapsedDimsArray}, x::AbstractArray, ni, nj)
    ni, nj = static(ni), static(nj)
    y, pullback = rrule(CollapsedDimsArray, x, collapsed_size(x, ni, nj), ni, nj)
    collapsed_array_pullback(Ȳ) = (NoTangent(), pullback(Ȳ)[2], NoTangent(), NoTangent())
    return y, collapsed_array_pullback
end

function ChainRulesCore.rrule(::Type{CollapsedDimsArray}, x::AbstractVector)
    ni = nj = static(0)
    dims = collapsed_size(parent, ni, nj)
    y, pullback = rrule(CollapsedDimsArray, x, dims, ni, nj)
    collapsed_array_pullback(Ȳ) = (NoTangent(), pullback(Ȳ)[2])
    return y, collapsed_array_pullback
end

function ChainRulesCore.rrule(::Type{CollapsedDimsArray}, x)
    ni = static(1)
    nj = static(ndims(x)) - static(2)
    dims = collapsed_size(x, ni, nj)
    y, pullback = rrule(CollapsedDimsArray, x, dims, ni, nj)
    collapsed_array_pullback(Ȳ) = (NoTangent(), pullback(Ȳ)[2])
    return y, collapsed_array_pullback
end

ChainRulesCore.rrule(config::RuleConfig, ::typeof(unwrap_collapse), x) = rrule(config, identity, x)
ChainRulesCore.rrule(config::RuleConfig, ::typeof(unwrap_collapse), x::CollapsedDimsArray) = rrule(config, parent, x)

function ChainRulesCore.rrule(::typeof(parent), x::CollapsedDimsArray)
    proj = ProjectTo(x)
    function collapsed_parent_pullback(Ybar)
        Ȳ = unthunk(Ybar)
        ∂x = @thunk proj(Ȳ)
        return (NoTangent(), ∂x)
    end
    collapsed_parent_pullback(::ZeroTangent) = (NoTangent(), ZeroTangent())
    return parent(x), collapsed_parent_pullback
end

function ChainRulesCore.rrule(::typeof(matmul), A::AbstractVecOrMat, B::AbstractVecOrMat, s)
    Y = matmul(A, B, s)
    function matmul_pullback(Ybar)
        Ȳ = unthunk(Ybar)
        if Ȳ isa ChainRulesCore.AbstractZero
            Athunk = Bthunk = sthunk = NoTangent()
        else
            Athunk = @thunk matmul(Ȳ, B', s)
            Bthunk = @thunk matmul(A', Ȳ, s)
            sthunk = @thunk sum(reshape(Ȳ, :) .* reshape(Y, :)) * inv(s)
        end
        return (NoTangent(), Athunk, Bthunk, sthunk)
    end
    matmul_pullback(::ZeroTangent) = (NoTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent())
    return Y, matmul_pullback
end

function ProjectTo(ca::CollapsedDimsArray)
    dims = size(parent(ca))
    return ProjectTo{CollapsedDimsArray}(; parent_size = dims, dims = size(ca), ni = ca.ni, nj = ca.nj)
end

function (project::ProjectTo{CollapsedDimsArray})(dx::AbstractArray)
    dx = unwrap_collapse(dx)
    return CollapsedDimsArray(reshape(dx, project.parent_size), project.dims, project.ni, project.nj)
end

function (project::ProjectTo{AbstractArray})(dx::CollapsedDimsArray)
    dx = reshape(unwrap_collapse(dx), project.axes)
    return project(dx)
end

function ChainRulesCore.rrule(::typeof(matmul), A::AbstractArray, B::AbstractArray, s)
    ProjA = ProjectTo(A)
    ProjB = ProjectTo(B)

    transA, pA = trans(A)
    transB, pB = trans(B)
    a = CollapsedDimsArray(pA)
    b = CollapsedDimsArray(pB)
    Y = matmul_wrapper(transA, transB, s, a, b)

    function matmul_pullback(Ybar)
        Ȳ = unthunk(Ybar)
        Â = trans(transA, a)
        B̂ = trans(transB, b)

        Athunk = @thunk begin
            tmp = matmul(Ȳ, batched_adjoint(B̂), s)
            ProjA(size(Â, 3) == 1 ? _sumbatch(tmp) : tmp)
        end
        Bthunk = @thunk begin
            tmp = matmul(batched_adjoint(Â), Ȳ, s)
            ProjB(size(B̂, 3) == 1 ? _sumbatch(tmp) : tmp)
        end
        sthunk = @thunk sum(reshape(Ȳ, :) .* reshape(unwrap_collapse(Y), :)) * inv(s)
        return (NoTangent(), Athunk, Bthunk, sthunk)
    end
    matmul_pullback(::ZeroTangent) = (NoTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent())
    return Y, matmul_pullback
end

ChainRulesCore.@non_differentiable noncollapsed_size(args...)
ChainRulesCore.@non_differentiable collapsed_size(args...)

_drop3(x) = Base.tail(Base.tail(Base.tail(x)))

function ChainRulesCore.rrule(
    config::RuleConfig, func::Union{
        typeof(collapseddims), typeof(collapseddims_nonbatch),
        typeof(collapseddims_fdim1), typeof(collapseddims_nonbatch_fdim1)
    },
    f, ca::CollapsedDimsArray, args...; kwargs...
)
    y, back = rrule(config, _collapseddims, _collapseddims_f_config(func)..., f, ca, args...; kwargs...)
    pullback(Ȳ) = (NoTangent(), _drop3(back(Ȳ))...)
    return y, pullback
end

@inline function ChainRulesCore.rrule(config::RuleConfig, ::typeof(_collapseddims), nonbatch, fdim1,
                                      f, ca::CollapsedDimsArray, args...; kwargs...)
    proj = ProjectTo(ca)
    input_size = size(ca)
    x = as_bool(nonbatch) ? collapseddims_nonbatch(ca) : collapseddims(ca)
    f_tape = rrule(config, f, x, args...; kwargs...)
    isnothing(f_tape) && (f_tape = rrule_via_ad(config, f, x, args...; kwargs...))
    _y, back = f_tape
    output_size = size(_y)
    @inline function collapseddims_pullback(Ybar)
        Ȳ = reshape(unthunk(Ybar), output_size)
        ∂f, ∂x, ∂args... = back(Ȳ)
        ∂ca = proj(∂x)
        return (NoTangent(), NoTangent(), NoTangent(), ∂f, ∂ca, ∂args...)
    end

    if !as_bool(fdim1)
        @assert output_size[1] == input_size[1] "func cannot change the size of feature dimension; use func with \"_fdim1\" suffix"
        return proj(_y), collapseddims_pullback
    else
        tail_size = last_n(size(parent(ca)), ca.ni + ca.nj)
        y = CollapsedDimsArray(reshape(_y, (:, tail_size...)), ca.ni, ca.nj)
        return y, collapseddims_pullback
    end
end
