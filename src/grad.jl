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
using ChainRulesCore: NoTangent, @thunk
import ChainRulesCore: ProjectTo
function ChainRulesCore.rrule(::Type{CollapsedDimArray}, x, dims, si, sj)
    s = size(x)
    function CollapsedDimArray_pullback(Ȳ)
        ∂x = @thunk unwrap_collapse(unthunk(Ȳ))
        return (NoTangent(), ∂x, NoTangent(), NoTangent(), NoTangent())
    end
    return CollapsedDimArray(x, dims, si, sj), CollapsedDimArray_pullback
end

function ChainRulesCore.rrule(::typeof(parent), x::CollapsedDimArray)
    si, sj = x.si, x.sj
    function collapsed_parent_pullback(Ȳ)
        ∂x = @thunk CollapsedDimArray(unthunk(Ȳ), si, sj)
        return (NoTangent(), ∂x)
    end
    return parent(x), collapsed_parent_pullback
end

function ChainRulesCore.rrule(::typeof(matmul), A::AbstractVecOrMat, B::AbstractVecOrMat, s)
    Y = matmul(A, B, s)
    function matmul_pullback(Ȳ)
        Ȳ = unthunk(Ȳ)
        Athunk = @thunk matmul(Ȳ, B', s)
        Bthunk = @thunk matmul(A', Ȳ, s)
        sthunk = @thunk sum(reshape(Ȳ, :) .* reshape(Y, :)) * inv(s)
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

    transA, pA = trans(A)
    transB, pB = trans(B)
    a = CollapsedDimArray(pA)
    b = CollapsedDimArray(pB)
    Y = matmul_wrapper(transA, transB, s, a, b)

    function matmul_pullback(Ȳ)
        Ȳ = unthunk(Ȳ)
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
    return Y, matmul_pullback
end
