using SIMD
using CUDA
using CUDA.WMMA
using CUDA: i32
using KernelAbstractions.Extras: @unroll

function fragment_idx1(lane, lda)
    lane0 = lane - 1i32
    cgrp, rgrp = _fast_fldmod(lane0, Val(4))
    offset = cgrp * lda + rgrp << 1i32
    idx0 = Vec{4, Int32}((0i32, 1i32, 8i32, 9i32))
    idx1 = idx0 + (lda << 3i32)
    idx = offset + Vec{8, Int32}(tuplejoin(idx0.data, idx1.data))
    return idx
end
function fragment_idx2(lane, lda)
    lane0 = lane - 1i32
    rgrp, cgrp = _fast_fldmod(lane0, Val(4))
    offset = (cgrp << 1i32) * lda + rgrp
    idx0 = Vec{4, Int32}((0i32, 1i32, 8i32, 9i32)) * lda
    idx1 = idx0 + 8i32
    idx = offset + Vec{8, Int32}(tuplejoin(idx0.data, idx1.data))
    return idx
end
function fragment_idx3(lane, lda)
    idx = fragment_idx1(lane, lda)
    idx = shufflevector(idx, Val((0,1,4,5,2,3,6,7)))
    return idx
end
function fragment_idx4(lane, lda)
    idx = fragment_idx2(lane, lda)
    idx = shufflevector(idx, Val((0,1,4,5,2,3,6,7)))
    return idx
end

fragment_a_idx(::Type{ColMajor}, lane, lda) = fragment_idx4(lane, lda)
fragment_a_idx(::Type{RowMajor}, lane, lda) = fragment_idx3(lane, lda)
fragment_b_idx(::Type{ColMajor}, lane, lda) = fragment_idx1(lane, lda)
fragment_b_idx(::Type{RowMajor}, lane, lda) = fragment_idx2(lane, lda)
fragment_c_idx(::Type{ColMajor}, lane, lda) = fragment_idx4(lane, lda)
fragment_c_idx(::Type{RowMajor}, lane, lda) = fragment_idx3(lane, lda)
fragment_idx(::Type{MatrixA}, ::Type{L}, lane, lda) where L <: FragmentLayout = fragment_a_idx(L, lane, lda)
fragment_idx(::Type{MatrixB}, ::Type{L}, lane, lda) where L <: FragmentLayout = fragment_b_idx(L, lane, lda)
fragment_idx(::Type{Accumulator}, ::Type{L}, lane, lda) where L <: FragmentLayout = fragment_c_idx(L, lane, lda)


struct MMAConfig{M, N, K, m_type, d_type} end

Base.@propagate_inbounds function fragment_load(
    config::Union{Type{<:MMAConfig{16, 16, 16}}, Type{<:Config{16, 16, 16}}}, use::Type{U}, layout::Type{L}, lane,
    mem, r, c, b...
) where {U <: WMMA.FragmentUse, L <: FragmentLayout}
    base = LinearIndices(size(mem))[r, c, b...]
    lda = stride(mem, 2) * i32
    indices = base + fragment_idx(U, L, lane, lda)
    vs = Base.Cartesian.@ntuple 8 i -> mem[indices[i]]
    if !(U <: Accumulator)
        vs = tuplejoin(vs, vs)
    end
    return vs
end

Base.@propagate_inbounds function fragment_load_a(
    config::Type{<:MMAConfig{16, 16, 16, T}}, layout::Type{L}, lane,
    data, r, c, b...
) where {T, L <: FragmentLayout}
    return Fragment{16, 16, 16, 16, T, L, MatrixA}(fragment_load(config, MatrixA, layout, lane, data, r, c, b...))
end
Base.@propagate_inbounds function fragment_load_a(
    config::Type{<:Config{16, 16, 16, T}}, layout::Type{L}, lane,
    data, r, c, b...
) where {T, L <: FragmentLayout}
    return fragment_load_a(MMAConfig{16, 16, 16, eltype(data), T}, layout, lane, data, r, c, b...)
end
Base.@propagate_inbounds function fragment_load_b(
    config::Type{<:MMAConfig{16, 16, 16, T}}, layout::Type{L}, lane,
    data, r, c, b...
) where {T, L <: FragmentLayout}
    return Fragment{16, 16, 16, 16, T, L, MatrixB}(fragment_load(config, MatrixB, layout, lane, data, r, c, b...))
end
Base.@propagate_inbounds function fragment_load_b(
    config::Type{<:Config{16, 16, 16, T}}, layout::Type{L}, lane,
    data, r, c, b...
) where {T, L <: FragmentLayout}
    return fragment_load_b(MMAConfig{16, 16, 16, eltype(data), T}, layout, lane, data, r, c, b...)
end
Base.@propagate_inbounds function fragment_load_c(
    config::Union{Type{<:MMAConfig{16, 16, 16, MT, T}}, Type{<:Config{16, 16, 16, T}}}, layout::Type{L}, lane,
    data, r, c, b...
) where {MT, T, L <: FragmentLayout}
    return Fragment{16, 16, 16, 8, T, Unspecified, Accumulator}(fragment_load(config, Accumulator, L, lane, data, r, c, b...))
end

Base.@propagate_inbounds function fragment_store_d(
    config::Union{Type{<:MMAConfig{16, 16, 16}}, Type{<:Config{16, 16, 16}}}, layout::Type{L}, lane,
    mem::AbstractArray{T}, frag::Fragment{16, 16, 16, 8, T}, r, c, b...
) where {T, L <: FragmentLayout}
    base = LinearIndices(size(mem))[r, c, b...]
    lda = stride(mem, 2) * i32
    indices = base + fragment_c_idx(L, lane, lda)
    Base.Cartesian.@nexprs 8 i -> mem[indices[i]] = frag[i]
    return nothing
end

Base.@propagate_inbounds function fragment_reduce_store_d(
    config::Union{Type{<:MMAConfig{16, 16, 16}}, Type{<:Config{16, 16, 16}}}, layout::Type{L}, lane,
    op, mem::AbstractArray{T}, frag::Fragment{16, 16, 16, 8, T}, r, c, b...
) where {T, L <: FragmentLayout}
    base = LinearIndices(size(mem))[r, c, b...]
    lda = stride(mem, 2) * i32
    idx = base + fragment_c_idx(L, lane, lda)
    vs = frag.x
    Base.Cartesian.@nexprs 8 i -> CUDA.atomic_arrayset(mem, idx[i], op, vs[i])
    return nothing
end

function shfl_dot(
    config::Union{Type{<:MMAConfig{16, 16, 16, MT, T}}, Type{<:Config{16, 16, 16, T}}},
    frag_a::Fragment{16, 16, 16, 16, MT},
    frag_b::Fragment{16, 16, 16, 16, MT},
    odd_even
) where {T, MT}
    @inbounds begin
        shflmask = typemax(UInt32)
        odd = odd_even[1]
        even = odd_even[2]
        a = WMMA.unflatten(NTuple{16, VecElement{MT}}, frag_a.x)
        b = WMMA.unflatten(NTuple{16, VecElement{MT}}, frag_b.x)
        a12, a34, a56, a78 = tuplesplitp(first(tuplesplitp(a, Val(2))), Val(4))
        a1256, a3478 = tuplejoin(a12, a56), tuplejoin(a34, a78)
        b1234, b5678 = tuplesplitp(first(tuplesplitp(b, Val(2))), Val(2))
        sb1234 = _shfl_sync(shflmask, b1234, odd)
        sb5678 = _shfl_sync(shflmask, b5678, odd)
        lb1234 = _shfl_sync(shflmask, b1234, even)
        lb5678 = _shfl_sync(shflmask, b5678, even)
        c12 = (VecElement{T}(_dot(T, a1256, sb1234)), VecElement{T}(_dot(T, a1256, lb1234)))
        c34 = (VecElement{T}(_dot(T, a3478, sb1234)), VecElement{T}(_dot(T, a3478, lb1234)))
        c56 = (VecElement{T}(_dot(T, a1256, sb5678)), VecElement{T}(_dot(T, a1256, lb5678)))
        c78 = (VecElement{T}(_dot(T, a3478, sb5678)), VecElement{T}(_dot(T, a3478, lb5678)))
        return Fragment{16, 16, 16, 8, T, Unspecified, Accumulator}(WMMA.flatten(tuplejoin(c12, c34, c56, c78)))
    end
end

function fragment_mma(
    config::Union{Type{<:MMAConfig{16, 16, 16, MT, T}}, Type{<:Config{16, 16, 16, T}}}, lane,
    frag_a::Fragment{16, 16, 16, 16, MT, L1, MatrixA},
    frag_b::Fragment{16, 16, 16, 16, MT, L2, MatrixB},
    frag_c::Fragment{16, 16, 16, 8, T, Unspecified, Accumulator},
) where {MT, T, L1, L2}
    shflmask = typemax(UInt32)
    ws = Val{32}()
    lane0 = lane - 1i32
    agrp, bgrp = _fast_fldmod(lane0, Val(4))
    base = lane - bgrp # 1i32 + _fast_mul(agrp, Val(4))
    odd_even = (_fast_mul(bgrp, Val(8)) + bgrp) + Vec{2, Int32}((1i32, 5i32))
    frag_c = frag_c .+ shfl_dot(config, frag_a, frag_b, odd_even)
    grp = bgrp
    odd_even = _fast_mod(odd_even + 7i32, ws) + 1i32
    grp = _fast_mod(grp + 3i32, Val(4))
    frag_c = frag_c .+ _shfl_sync(shflmask, shfl_dot(config, frag_a, frag_b, odd_even), grp + base)
    odd_even = _fast_mod(odd_even + 7i32, ws) + 1i32
    grp = _fast_mod(grp + 3i32, Val(4))
    frag_c = frag_c .+ _shfl_sync(shflmask, shfl_dot(config, frag_a, frag_b, odd_even), grp + base)
    odd_even = _fast_mod(odd_even + 7i32, ws) + 1i32
    grp = _fast_mod(grp + 3i32, Val(4))
    frag_c = frag_c .+ _shfl_sync(shflmask, shfl_dot(config, frag_a, frag_b, odd_even), grp + base)
    return frag_c
end
