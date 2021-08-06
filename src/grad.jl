using FiniteDifferences

function FiniteDifferences.to_vec(X::CollapsedDimArray)
    x_vec, back = to_vec(collapseddim(X))
    s = size(parent(X))
    si = X.si
    sj = X.sj
    function CollapsedDimArray_from_vec(x_vec)
        return CollapsedDimArray(reshape(back(x_vec), s), si, sj)
    end
    return x_vec, CollapsedDimArray_from_vec
end

@inline function _sumbatch(ca::CollapsedDimArray)
     return CollapsedDimArray(sum(parent(ca), dims=ntuple(i->i-1+ca.sj, ndims(parent(ca))+1-ca.sj)), ca.si, ca.sj)
end

using ChainRulesCore
using ChainRulesCore: NoTangent
import ChainRulesCore: ProjectTo
function ChainRulesCore.rrule(::Type{CollapsedDimArray}, x, dims, si, sj)
    s = size(x)
    function CollapsedDimArray_pullback(Ȳ)
        Ȳ = unwrap_collapse(unthunk(Ȳ))
        ∂x = size(Ȳ) == s ? Ȳ : reshape(Ȳ, s)
        return (NoTangent(), ∂x, NoTangent(), NoTangent(), NoTangent())
    end
    return CollapsedDimArray(x, dims, si, sj), CollapsedDimArray_pullback
end

function ChainRulesCore.rrule(::typeof(parent), x::CollapsedDimArray)
    s = size(x)
    si = x.si
    sj = x.sj
    function collapsed_parent_pullback(Ȳ)
        Ȳ = unthunk(Ȳ)
        ∂x = size(Ȳ) == s ? Ȳ : reshape(Ȳ, s)
        return (NoTangent(), CollapsedDimArray(∂x, si, sj))
    end
    return parent(x), collapsed_parent_pullback
end

function ChainRulesCore.rrule(::typeof(matmul), A::AbstractVecOrMat, B::AbstractVecOrMat, s)
    Y = matmul(A, B, s)
    function matmul_pullback(Ȳ)
        Ȳ = unthunk(Ȳ)
        Athunk = ChainRulesCore.@thunk matmul(Ȳ, B', s)
        Bthunk = ChainRulesCore.@thunk matmul(A', Ȳ, s)
        sthunk = ChainRulesCore.@thunk sum(reshape(Ȳ, :) .* reshape(Y, :)) * inv(s)
        return (NoTangent(), Athunk, Bthunk, sthunk)
    end
    return Y, matmul_pullback
end

function ProjectTo(ca::CollapsedDimArray)
    dims = size(parent(ca))
    return ProjectTo{CollapsedDimArray}(; dims=dims, si = ca.si, sj = ca.sj)
end

function (project::ProjectTo{CollapsedDimArray})(dx::AbstractArray)
    dx = unwrap_collapse(dx)
    return CollapsedDimArray(reshape(dx, project.dims), project.si, project.sj)
end

function (project::ProjectTo{AbstractArray})(dx::CollapsedDimArray)
    dx = reshape(unwrap_collapse(dx), project.axes)
    return project(dx)
end

function ChainRulesCore.rrule(::typeof(matmul), A::AbstractArray, B::AbstractArray, s)
    ProjA = ProjectTo(A)
    ProjB = ProjectTo(B)
    Y = matmul(A, B, s)
    function matmul_pullback(Ȳ)
        Ȳ = unthunk(Ȳ)
        ta, pa = trans(A)
        Â = trans(ta, CollapsedDimArray(pa))
        tb, pb = trans(B)
        B̂ = trans(tb, CollapsedDimArray(pb))

        Athunk = ChainRulesCore.@thunk begin
            tmp = matmul(Ȳ, batched_adjoint(B̂), s)
            ProjA(size(Â, 3) == 1 ? _sumbatch(tmp) : tmp)
        end
        Bthunk = ChainRulesCore.@thunk begin
            tmp = matmul(batched_adjoint(Â), Ȳ, s)
            ProjB(size(B̂, 3) == 1 ? _sumbatch(tmp) : tmp)
        end
        sthunk = ChainRulesCore.@thunk sum(reshape(Ȳ, :) .* reshape(unwrap_collapse(Y), :)) * inv(s)
        return (NoTangent(), Athunk, Bthunk, sthunk)
    end
    return Y, matmul_pullback
end
