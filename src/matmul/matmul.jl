@inline trans(b::NNlib.BatchedAdjOrTrans) = (b isa NNlib.BatchedTranspose ? static(Int('T')) : static(Int('C'))), parent(b)
@inline trans(a::LinearAlgebra.AdjOrTrans) = (a isa LinearAlgebra.Transpose ? static(Int('T')) : static(Int('C'))), parent(a)
@inline trans(x) = static(Int('N')), x
@inline trans(c, x) = Int(c) == static(Int('T')) ? batched_transpose(x) :
    Int(c) == static(Int('C')) ? batched_adjoint(x) : x

matmul(a, b) = matmul(a, b, true)
matmul(a::AbstractVector, b::AbstractVecOrMat, s::Number) = matmul(reshape(a, length(a), 1), b, s)
function matmul(a::AbstractMatrix, b::AbstractVecOrMat, s::Number)
    TS = promote_type(eltype(a), eltype(b))
    A = convert(AbstractArray{TS}, a)
    B = convert(AbstractArray{TS}, b)
    m = size(a, 1)
    n = size(b, 2)
    C = similar(b, TS, (m, n))
    return LinearAlgebra.mul!(C, A, B, s, false)
end

function matmul(a::AbstractArray, b::AbstractArray, s::Number)
    transA, pA = trans(a)
    transB, pB = trans(b)
    A = CollapsedDimsArray(pA)
    B = CollapsedDimsArray(pB)
    return matmul_wrapper(transA, transB, s, A, B)
end

@inline gemm_strided_batched_wrapper(transA::Union{AbstractChar, StaticInt}, transB::Union{AbstractChar, StaticInt}, alpha::Number, A::AbstractArray, B::AbstractArray) =
    gemm_strided_batched_wrapper(transA, transB, alpha, CollapsedDimsArray(A), CollapsedDimsArray(B))

@inline gemm_strided_batched_wrapper(transA::Union{AbstractChar, StaticInt}, transB::Union{AbstractChar, StaticInt}, alpha::Number, A::AbstractArray, B::CollapsedDimsArray) =
    gemm_strided_batched_wrapper(transA, transB, alpha, CollapsedDimsArray(A), B)

@inline gemm_strided_batched_wrapper(transA::Union{AbstractChar, StaticInt}, transB::Union{AbstractChar, StaticInt}, alpha::Number, A::CollapsedDimsArray, B::AbstractArray) =
    gemm_strided_batched_wrapper(transA, transB, alpha, A, CollapsedDimsArray(B))

@inline function gemm_strided_batched_wrapper(transA::Union{AbstractChar, StaticInt}, transB::Union{AbstractChar, StaticInt}, alpha::Number, A::CollapsedDimsArray, B::CollapsedDimsArray)
    m = noncollapsed_size(A.parent, A.ni, A.nj, Int(transA) == static(Int('N')) ? static(1) : static(2))
    n = noncollapsed_size(B.parent, B.ni, B.nj, Int(transB) == static(Int('N')) ? static(2) : static(1))
    # batch size differ is allow only when ones batch size is one
    sc3 = B.nj == 0 ?
        noncollapsed_size(A.parent, A.ni, A.nj, static(3)) :
        noncollapsed_size(B.parent, B.ni, B.nj, static(3))

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

    Ci = static(length(n))
    Cj = static(length(sc3))
    C = similar(pB, T, (m..., n..., sc3...))
    if ndims(pA) == ndims(pB) == ndims(C) == 3
        gemm_strided_batched!(as_char(transA), as_char(transB), convert(T, alpha), pA, pB, zero(T), C)
    else
        gemm_strided_batched!(as_char(transA), as_char(transB), convert(T, alpha), pA, pB, zero(T), C, A.ni, A.nj, B.ni, B.nj, Ci, Cj)
    end

    return CollapsedDimsArray(C, Ci, Cj)
end

function generic_matmul(transA::Union{AbstractChar, StaticInt}, transB::Union{AbstractChar, StaticInt}, alpha::Number, A, B)
    T = promote_type(eltype(A), eltype(B))
    scale = convert(T, alpha)
    if A isa CollapsedDimsArray
        m = noncollapsed_size(A.parent, A.ni, A.nj, Int(transA) == static(Int('N')) ? static(1) : static(2))
        pA = collapseddims(A)
        sa3 = noncollapsed_size(A.parent, A.ni, A.nj, static(3))
    else
        m = size(A, Int(transA) == static(Int('N')) ? static(1) : static(2))
        pA = A
        sa3 = size(A, 3)
    end

    if B isa CollapsedDimsArray
        n = noncollapsed_size(B.parent, B.ni, B.nj, Int(transB) == static(Int('N')) ? static(2) : static(1))
        pB = collapseddims(B)
        sb3 = noncollapsed_size(B.parent, B.ni, B.nj, static(3))
    else
        n = size(B, Int(transB) == static(Int('N')) ? static(2) : static(1))
        pB = B
        sb3 = size(B, 3)
    end
    sc3 = B.nj == 0 ? sa3 : sb3
    Ci = static(length(n))
    Cj = static(length(sc3))
    outsize = (m..., n..., sc3...)
    y = scale .* batched_mul(trans(transA, pA), trans(transB, pB))
    return CollapsedDimsArray(reshape(y, outsize), Ci, Cj)
end

NNlib.is_strided(ca::CollapsedDimsArray) = NNlib.is_strided(parent(ca))

@inline function matmul_wrapper(transA::Union{AbstractChar, StaticInt}, transB::Union{AbstractChar, StaticInt}, alpha::Number, A::AbstractArray{TA, 3}, B::AbstractArray{TB, 3}) where {TA, TB}
    mA = size(A, Int(transA) == static(Int('N')) ? 1 : 2)
    kA = size(A, Int(transA) == static(Int('N')) ? 2 : 1)
    bA = size(A, 3)
    kB = size(B, Int(transB) == static(Int('N')) ? 1 : 2)
    nB = size(B, Int(transB) == static(Int('N')) ? 2 : 1)
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
