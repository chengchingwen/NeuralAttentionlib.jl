@init @require FiniteDifferences="26cc04aa-876d-5657-8c51-4c34ba976000" begin
    using .FiniteDifferences

    function FiniteDifferences.to_vec(X::CollapsedDimArray)
        x_vec, back = to_vec(collapseddim(X))
        s = size(parent(X))
        ni = X.ni
        nj = X.nj
        function CollapsedDimArray_from_vec(x_vec)
            return CollapsedDimArray(reshape(back(x_vec), s), ni, nj)
        end
        return x_vec, CollapsedDimArray_from_vec
    end
end

@inline function _sumbatch(ca::CollapsedDimArray)
    offset = static(ndims(parent(ca))) - ca.nj
    y = sum(parent(ca); dims = ntuple(Base.Fix1(+, offset), ca.nj))
    return CollapsedDimArray(y, ca.ni, ca.nj)
end

using ChainRulesCore
using ChainRulesCore: NoTangent, @thunk
import ChainRulesCore: ProjectTo
function ChainRulesCore.rrule(::Type{CollapsedDimArray}, x, dims, ni, nj)
    s = size(x)
    function CollapsedDimArray_pullback(Ȳ)
        ∂x = @thunk begin
            tmp = unwrap_collapse(unthunk(Ȳ))
            size(tmp) == s ? tmp : reshape(tmp, s)
        end
        return (NoTangent(), ∂x, NoTangent(), NoTangent(), NoTangent())
    end
    return CollapsedDimArray(x, dims, ni, nj), CollapsedDimArray_pullback
end

function ChainRulesCore.rrule(::typeof(parent), x::CollapsedDimArray)
    ni, nj = x.ni, x.nj
    function collapsed_parent_pullback(Ȳ)
        ∂x = @thunk CollapsedDimArray(unthunk(Ȳ), ni, nj)
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
    return ProjectTo{CollapsedDimArray}(; dims=dims, ni = ca.ni, nj = ca.nj)
end

function (project::ProjectTo{CollapsedDimArray})(dx::AbstractArray)
    dx = unwrap_collapse(dx)
    return CollapsedDimArray(reshape(dx, project.dims), project.ni, project.nj)
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
