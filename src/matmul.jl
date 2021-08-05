function collapseddim(x::AbstractArray, xi, xj)
    return reshape(x, collapsed_size(x, xi, xj))
end

struct CollapsedDimArray{T, A<:AbstractArray{T}, S1, S2} <: AbstractArray{T, 3}
    parent::A
    dims::NTuple{3, Int}
    si::S1
    sj::S2
end

Base.unsafe_convert(::Type{Ptr{T}}, ca::CollapsedDimArray{T}) where {T} = Base.unsafe_convert(Ptr{T}, parent(ca))
Base.parent(ca::CollapsedDimArray) = ca.parent
Base.similar(ca::CollapsedDimArray, eltype::Type, dims::Dims) = similar(parent(ca), eltype, dims)
Base.eltype(::CollapsedDimArray{T}) where T = T
Base.length(ca::CollapsedDimArray) = length(parent(ca))
Base.size(ca::CollapsedDimArray) = ca.dims

Base.strides(ca::CollapsedDimArray) = strides(Base.ReshapedArray(parent(ca), ca.dims, ()))

CollapsedDimArray(ca::CollapsedDimArray) = ca
CollapsedDimArray(parent) = CollapsedDimArray(parent, static(2), static(3))
function CollapsedDimArray(parent, si, sj)
    s1 = Bool(is_static(si)) ? si : static(si)
    s2 = Bool(is_static(sj)) ? sj : static(sj)
    dims = collapsed_size(parent, s1, s2)
    return CollapsedDimArray(parent, dims, s1, s2)
end

function Base.getindex(ca::CollapsedDimArray, i...)
    return Base.getindex(Base.ReshapedArray(parent(ca), ca.dims, ()), i...)
end

const CollapsedAdjOrTrans{T} = NNlib.BatchedAdjOrTrans{T, <:CollapsedDimArray{T}}
const Collapsed{T} = Union{CollapsedAdjOrTrans{T}, <:CollapsedDimArray{T}}

collapseddim(x::AbstractArray{T, 3}) where T = x
collapseddim(ca::CollapsedDimArray) = reshape(parent(ca), ca.dims)
collapseddim(ca::CollapsedAdjOrTrans) = ca isa NNlib.BatchedTranspose ? batched_transpose(collapseddim(parent(ca))) :
    batched_adjoint(collapseddim(parent(ca)))

unwrap_collapse(x) = x
unwrap_collapse(ca::CollapsedDimArray) = parent(ca)

trans(b::NNlib.BatchedAdjOrTrans) = (b isa NNlib.BatchedTranspose ? 'T' : 'C'), parent(b)
trans(x) = 'N', x
trans(c, x) = c == 'T' ? batched_transpose(x) :
    c == 'C' ? batched_adjoint(x) : x

matmul(a, b) = matmul(a, b, true)
matmul(a::AbstractVecOrMat, b::AbstractVecOrMat, s::Number) = (a * b) .* s
function matmul(a::AbstractArray, b::AbstractArray, s::Number)
    transA, pA = trans(a)
    transB, pB = trans(b)
    A = CollapsedDimArray(pA)
    B = CollapsedDimArray(pB)
    return matmul_wrapper(transA, transB, s, A, B)
end

gemm_strided_batched_wrapper(transA::Char, transB::Char, alpha::Number, A::AbstractArray, B::AbstractArray) =
    gemm_strided_batched_wrapper(transA, transB, alpha, CollapsedDimArray(A), CollapsedDimArray(B))

gemm_strided_batched_wrapper(transA::Char, transB::Char, alpha::Number, A::AbstractArray, B::CollapsedDimArray) =
    gemm_strided_batched_wrapper(transA, transB, alpha, CollapsedDimArray(A), B)

gemm_strided_batched_wrapper(transA::Char, transB::Char, alpha::Number, A::CollapsedDimArray, B::AbstractArray) =
    gemm_strided_batched_wrapper(transA, transB, alpha, A, CollapsedDimArray(B))

function gemm_strided_batched_wrapper(transA::Char, transB::Char, alpha::Number, A::CollapsedDimArray, B::CollapsedDimArray)
    m = noncollapsed_size(A.parent, A.si, A.sj, transA == 'N' ? static(1) : static(2))
    n = noncollapsed_size(B.parent, B.si, B.sj, transB == 'N' ? static(2) : static(1))
    sc3 = size(A, 3) > size(B, 3) ?
        noncollapsed_size(A.parent, A.si, A.sj, static(3)) :
        noncollapsed_size(B.parent, B.si, B.sj, static(3))

    T = promote_type(eltype(A), eltype(B))
    if eltype(A) == T
        pA = parent(A)
    else
        pA = convert(AbstractArray{T}, parent(A))
    end
    if eltype(B) == T
        pB = parent(B)
    else
        pB = convert(AbstractArray{T}, parent(B))
    end

    Ci = static(length(m) + 1)
    Cj = static(Ci + length(n))
    C = similar(pB, T, (m..., n..., sc3...))
    if ndims(pA) == ndims(pB) == ndims(C) == 3
        gemm_strided_batched!(transA, transB, convert(T, alpha), pA, pB, zero(T), C)
    else
        gemm_strided_batched!(transA, transB, convert(T, alpha), pA, pB, zero(T), C, A.si, A.sj, B.si, B.sj, Ci, Cj)
    end

    return CollapsedDimArray(C, Ci, Cj)
end

function generic_matmul(transA::Char, transB::Char, alpha::Number, A, B)
    T = promote_type(eltype(A), eltype(B))
    scale = convert(T, alpha)
    if A isa CollapsedDimArray
        m = noncollapsed_size(A.parent, A.si, A.sj, transA == 'N' ? static(1) : static(2))
        pA = collapseddim(A)
        sa3 = noncollapsed_size(A.parent, A.si, A.sj, static(3))
    else
        m = size(A, transA == 'N' ? static(1) : static(2))
        pA = A
        sa3 = size(A, 3)
    end

    if B isa CollapsedDimArray
        n = noncollapsed_size(B.parent, B.si, B.sj, transB == 'N' ? static(2) : static(1))
        pB = collapseddim(B)
        sb3 = noncollapsed_size(B.parent, B.si, B.sj, static(3))
    else
        n = size(B, transB == 'N' ? static(2) : static(1))
        pB = B
        sb3 = size(B, 3)
    end
    sc3 = size(A, 3) > size(B, 3) ? sa3 : sb3
    Ci = static(length(m) + 1)
    Cj = static(Ci + length(n))
    outsize = (m..., n..., sc3...)
    y = scale .* batched_mul(trans(transA, pA), trans(transB, pB))
    return CollapsedDimArray(reshape(y, outsize), Ci, Cj)
end

NNlib.is_strided(ca::CollapsedDimArray) = NNlib.is_strided(parent(ca))

function matmul_wrapper(transA::Char, transB::Char, alpha::Number, A::AbstractArray{TA, 3}, B::AbstractArray{TB, 3}) where {TA, TB}
    mA = size(A, transA == 'N' ? 1 : 2)
    kA = size(A, transA == 'N' ? 2 : 1)
    bA = size(A, 3)
    kB = size(B, transB == 'N' ? 1 : 2)
    nB = size(B, transB == 'N' ? 2 : 1)
    bB = size(B, 3)

    if kA != kB || (bA != bB && bA != 1 && bB != 1)
        throw(DimensionMismatch("A has dimensions ($mA,$kA,$bA) but B has dimensions ($kB,$nB,$bB)"))
    end

    if TA <: BLAS.BlasFloat && TB <: BLAS.BlasFloat && NNlib.is_strided(A) && NNlib.is_strided(B)
        return gemm_strided_batched_wrapper(transA, transB, alpha, A, B)
    else
        return generic_matmul(transA, transB, alpha, A, B)
    end
end

function NNlib.softmax(ca::CollapsedDimArray, args...; kwargs...)
    real_size = size(parent(ca))
    y = softmax(collapseddim(ca), args...; kwargs...)
    return CollapsedDimArray(reshape(y, real_size), ca.dims, ca.si, ca.sj)
end

@inline function _sumbatch(ca::CollapsedDimArray)
     return CollapsedDimArray(sum(parent(ca), dims=ntuple(i->i-1+ca.sj, ndims(parent(ca))+1-ca.sj)), ca.si, ca.sj)
end

using ChainRulesCore
using ChainRulesCore: NoTangent
# function ChainRulesCore.rrule(::typeof(matmul_wrapper), transA, transB, s, A, B)
#     Y = matmul_wrapper(transA, transB, s, A, B)
#     function matmul_wrapper_pullback(Ȳ)
#         Athunk = ChainRulesCore.@thunk begin
#             tmp = matmul(Ȳ, batched_adjoint(trans(transB, B)), s)
#             size(A, 3) == 1 ? _sumbatch(tmp) : tmp
#         end
#         Bthunk = ChainRulesCore.@thunk begin
#             tmp = matmul(batched_adjoint(trans(transA,A)), Ȳ, s)
#             size(B, 3) == 1 ? _sumbatch(tmp) : tmp
#         end
#         sthunk = ChainRulesCore.@thunk sum(reshape(Ȳ, :) .* reshape(unwrap_collapse(Y), :)) * inv(s)
#         return (NoTangent(), NoTangent(), NoTangent(),
#                 sthunk, Athunk, Bthunk)
#     end
#     return Y, matmul_wrapper_pullback
# end

function ChainRulesCore.rrule(::Type{CollapsedDimArray}, x, dims, si, sj)
    s = size(x)
    function CollapsedDimArray_pullback(Ȳ)
        Ȳ = unwrap_collapse(Ȳ)
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
        ∂x = size(Ȳ) == s ? Ȳ : reshape(Ȳ, s)
        return (NoTangent(), CollapsedDimArray(∂x, si, sj))
    end
    return parent(x), collapsed_parent_pullback
end

# function ChainRulesCore.rrule(::typeof(trans), x)
#     t, Y = trans(x)
#     function trans_pullback(Ȳ)
#         ∂x = Ȳ[2]
#         return (NoTangent(), ∂x)
#     end
#     return (t, Y), trans_pullback
# end

function ChainRulesCore.rrule(::typeof(matmul), A::AbstractVecOrMat, B::AbstractVecOrMat, s)
    Y = matmul(A, B, s)
    function matmul_pullback(Ȳ)
        Athunk = ChainRulesCore.@thunk matmul(Ȳ, B', s)
        Bthunk = ChainRulesCore.@thunk matmul(A', Ȳ, s)
        sthunk = ChainRulesCore.@thunk sum(reshape(Ȳ, :) .* reshape(Y, :)) * inv(s)
        return (NoTangent(), Athunk, Bthunk, sthunk)
    end
    return Y, matmul_pullback
end

function ChainRulesCore.rrule(::typeof(matmul), A::AbstractArray, B::AbstractArray, s)
    Y = matmul(A, B, s)
    function matmul_pullback(Ȳ)
        ta, pa = trans(A)
        Â = trans(ta, CollapsedDimArray(pa))
        tb, pb = trans(B)
        B̂ = trans(tb, CollapsedDimArray(pb))

        Athunk = ChainRulesCore.@thunk begin
            tmp = matmul(Ȳ, batched_adjoint(B̂), s)
            size(Â, 3) == 1 ? _sumbatch(tmp) : tmp
        end
        Bthunk = ChainRulesCore.@thunk begin
            tmp = matmul(batched_adjoint(A), Ȳ, s)
            size(B̂, 3) == 1 ? _sumbatch(tmp) : tmp
        end
        sthunk = ChainRulesCore.@thunk sum(reshape(Ȳ, :) .* reshape(unwrap_collapse(Y), :)) * inv(s)
        return (NoTangent(), Athunk, Bthunk, sthunk)
    end
    return Y, matmul_pullback
end
