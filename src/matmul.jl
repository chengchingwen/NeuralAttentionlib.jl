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
Base.length(ca::CollapsedDimArray) = length(ca.parent)
Base.size(ca::CollapsedDimArray) = ca.dims

Base.strides(ca::CollapsedDimArray) = strides(Base.ReshapedArray(ca.parent, ca.dims, ()))

CollapsedDimArray(parent) = CollapsedDimArray(parent, static(2), static(3))
function CollapsedDimArray(parent, si, sj)
    s1 = Bool(is_static(si)) ? si : static(si)
    s2 = Bool(is_static(sj)) ? sj : static(sj)
    dims = collapsed_size(parent, s1, s2)
    return CollapsedDimArray(parent, dims, s1, s2)
end

function Base.getindex(ca::CollapsedDimArray, i...)
    return Base.getindex(Base.ReshapedArray(ca.parent, ca.dims, ()), i...)
end

const CollapsedAdjOrTrans{T} = NNlib.BatchedAdjOrTrans{T, <:CollapsedDimArray{T}}
const Collapsed{T} = Union{CollapsedAdjOrTrans{T}, CollapsedDimArray{T}}

collapseddim(ca::CollapsedDimArray) = reshape(ca.parent, ca.dims)
collapseddim(ca::CollapsedAdjOrTrans) = ca isa BatchedTranspose ? batched_transpose(collapseddim(ca.parent)) :
    batched_adjoint(collapseddim(ca.parent))

unwrap_collapse(x) = x
unwrap_collapse(ca::CollapsedDimArray) = parent(ca)

matmul(a::AbstractVecOrMat, b::AbstractVecOrMat, s::Number) = (a * b) .* s

matmul(a::AbstractArray{TA, 3}, b::AbstractArray{TB, 3}, s::Number) where {TA, TB} = batched_mul(a, b) .* s

matmul(a::AbstractArray, b::Collapsed, s::Number) = matmul(a, collapseddim(b), s)
matmul(a::Collapsed, b::AbstractArray, s::Number) = matmul(collapseddim(a), b, s)

matmul(a, b) = matmul(a, b, one(promote_type(eltype(a), eltype(b))))
function matmul(a::CollapsedDimArray{T}, b::CollapsedDimArray{T}, s::Number) where {T <: BLAS.BlasFloat}
    return gemm_strided_batched_wrapper(a, b, static(false), static(false), convert(T, s), a.parent, b.parent, a.si, a.sj, b.si, b.sj)
end

function matmul(a::CollapsedAdjOrTrans{T}, b::CollapsedDimArray{T}, s::Number) where {T <: BLAS.BlasFloat}
    transA = static(true)#a isa NNlib.BatchedTranspose ? 'T' : 'C'
    a = a.parent
    return gemm_strided_batched_wrapper(a, b, transA, static(false), convert(T, s), a.parent, b.parent, a.si, a.sj, b.si, b.sj)
end

function matmul(a::CollapsedDimArray{T}, b::CollapsedAdjOrTrans{T}, s::Number) where {T <: BLAS.BlasFloat}
    transB = static(true)#b isa NNlib.BatchedTranspose ? 'T' : 'C'
    b = b.parent
    return gemm_strided_batched_wrapper(a, b, static(false), transB, convert(T, s), a.parent, b.parent, a.si, a.sj, b.si, b.sj)
end

function matmul(a::CollapsedAdjOrTrans{T}, b::CollapsedAdjOrTrans{T}, s::Number) where {T <: BLAS.BlasFloat}
    transA = static(true)#a isa NNlib.BatchedTranspose ? 'T' : 'C'
    transB = static(true)#b isa NNlib.BatchedTranspose ? 'T' : 'C'
    a = a.parent
    b = b.parent
    return gemm_strided_batched_wrapper(a, b, transA, transB, convert(T, s), a.parent, b.parent, a.si, a.sj, b.si, b.sj)
end

@inline function gemm_strided_batched_wrapper(a::CollapsedDimArray, b::CollapsedDimArray, transA, transB, alpha, A, B, Ai, Aj, Bi, Bj)

    m = noncollapsed_size(A, Ai, Aj, !as_bool(transA) ? static(1) : static(2))
    n = noncollapsed_size(B, Bi, Bj, !as_bool(transB) ? static(2) : static(1))
    sc3 = @inbounds(a.dims[3] > b.dims[3]) ?
        noncollapsed_size(A, Ai, Aj, static(3)) :
        noncollapsed_size(B, Bi, Bj, static(3))

    transA = as_bool(transA) ? (alpha isa Complex ? 'C' : 'T') : 'N'
    transB = as_bool(transB) ? (alpha isa Complex ? 'C' : 'T') : 'N'
    
    Ci = static(length(m) + 1)
    Cj = static(Ci + length(n))
    C = similar(B, (m..., n..., sc3...))
    gemm_strided_batched!(transA, transB, alpha, A, B, zero(alpha), C, Ai, Aj, Bi, Bj, Ci, Cj)

    return CollapsedDimArray(C, Ci, Cj)
end

NNlib.softmax(ca::CollapsedDimArray, args...; kwargs...) = CollapsedDimArray(softmax(parent(ca)), ca.dims, ca.si, ca.sj)
