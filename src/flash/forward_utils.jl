using CUDA
using CUDA: i32
using KernelAbstractions.Extras: @unroll

function block_shm_fill!(shm, v)
    D = size(shm, 1) % Int32
    works = length(shm)
    workload = cld(works, blockDim().x) % Int32
    top = threadIdx().x * workload
    base = top - workload + 1i32
    for i = base:min(top, works)
        c, r = fldmod1(i, D)
        @inbounds shm[r, c] = v
    end
    return nothing
end

function block_glb2shm!(config, shm, glb, range, b = 1)
    # size(shm) = (D, Bx)
    # size(glb) = (d, N, B)
    # assume D >= d, Bx >= N
    D = size(shm, 1) % Int32
    d = size(glb, 1) % Int32
    N = length(range) % Int32
    cbase = first(range) - 1i32
    works = length(shm) % Int32
    workload = cld(works, blockDim().x) % Int32
    top = threadIdx().x * workload
    base = top - workload + 1i32
    for i = base:min(top, works)
        c, r = fldmod1(i, D)
        @inbounds if r > d || c > N
            shm[r, c] = zero(eltype(shm))
        else
            shm[r, c] = convert(eltype(shm), glb[r, c + cbase, b])
        end
    end
    return nothing
end

function block_shm2glb!(config, glb, shm, range, b = 1)
    # size(shm) = (D, Bx)
    # size(glb) = (d, N, B)
    # assume D >= d, Bx >= N
    D = size(shm, 1) % Int32
    d = size(glb, 1) % Int32
    N = length(range) % Int32
    cbase = first(range) - 1i32
    works = (d * N) % Int32
    workload = cld(works, blockDim().x) % Int32
    top = threadIdx().x * workload
    base = top - workload + 1i32
    for i = base:min(top, works)
        c, r = fldmod1(i, d)
        @inbounds glb[r, c + cbase, b] = shm[r, c]
    end
    return nothing
end

function chunkrange(B, N, i)
    stop = i * B
    start0 = stop - B
    start = start0 + 1i32
    if stop > N
        stop = N
        len = N - start0
        full = false
    else
        len = B
        full = true
    end
    return (full, start:stop, len)
end

_elseif(a, p, b) = ifelse(p, a, b)

@inline function _warp_gen_pmask(
    ::Union{Type{<:MMAConfig{16, 16, 16, MT, T}}, Type{<:Config{16, 16, 16, T}}}, lane,
    klen, qlen, sj, si) where {MT, T}
    r, c = _fast_fldmod(lane - 1i32, Val(4))
    vs = ((sj + r + Vec{8, Int32}((0i32, 0i32, 8i32, 8i32, 0i32, 0i32, 8i32, 8i32))) <= klen) &
        ((si + c + Vec{8, Int32}((0i32, 1i32, 0i32, 1i32, 8i32, 9i32, 8i32, 9i32))) <= qlen)
    return Fragment{16, 16, 16, 8, Bool, Unspecified, Accumulator}(WMMA.flatten(vs.data))
end

@inline function warp_gen_pmask(config, lane, klen, qlen, sj, si)
    (; Wm, Wn, Wk) = config
    (; computeT, reduceT) = config
    return _warp_gen_pmask(MMAConfig{Wn, Wm, Wk, computeT, reduceT}, lane, klen, qlen, sj, si)
end

@inline function warp_shm_reduce_nxm_atomic!(config, lane, op, mem, frag, r, c)
    (; Wm, Wn, Wk) = config
    (; computeT, reduceT) = config
    @inbounds fragment_reduce_store_d(MMAConfig{Wn, Wm, Wk, computeT, reduceT}, ColMajor, lane, op, mem, frag, r, c)
    return nothing
end

@inline function _warp_reduce_1xm!(
    ::Union{Type{<:MMAConfig{16, 16, 16, MT, T}}, Type{<:Config{16, 16, 16, T}}}, lane,
    op, mi, m::Fragment{16, 16, 16, 4, T, Unspecified, Accumulator}, si
) where {MT, T}
    shflmask = typemax(UInt32)
    grp = _fast_mod(lane - 1i32, Val(4))
    vs = m.x
    vs = op.(vs, _shfl_xor_sync(shflmask, vs, 4i32))
    vs = op.(vs, _shfl_xor_sync(shflmask, vs, 8i32))
    vs = op.(vs, _shfl_xor_sync(shflmask, vs, 16i32))
    @inbounds if lane <= 4
        # m = [(1, 1) (1, 2) (9, 1) (9, 2) (1, 9) (1, 10) (9, 9) (9, 10)]
        idx = si + _fast_mul(grp, Val(2)) + Vec{4, Int32}((0i32, 1i32, 8i32, 9i32))
        Base.Cartesian.@nexprs 4 i -> CUDA.atomic_arrayset(mi, idx[i], op, vs[i])
    end
    return nothing
end

@inline function warp_shm_reduce_1xm_atomic!(config, lane, op, mi, m, si)
    (; Wm, Wn, Wk) = config
    (; computeT, reduceT) = config
    _warp_reduce_1xm!(MMAConfig{Wn, Wm, Wk, computeT, reduceT}, lane, op, mi, m, si)
    return nothing
end

@inline function _warp_reducerow_nxm(
    ::Union{Type{<:MMAConfig{16, 16, 16, MT, T}}, Type{<:Config{16, 16, 16, T}}},
    op,
    frag::Fragment{16, 16, 16, 8, T, Unspecified, Accumulator},
    acc::Union{Nothing, Fragment{16, 16, 16, 4, T, Unspecified, Accumulator}} = nothing
) where {MT, T}
    vs = frag.x
    if isnothing(acc)
        v1 = @inbounds op(vs[1], vs[3])
        v2 = @inbounds op(vs[2], vs[4])
        v3 = @inbounds op(vs[5], vs[7])
        v4 = @inbounds op(vs[6], vs[8])
    else
        a = acc.x
        v1 = @inbounds op(op(a[1], vs[1]), vs[3])
        v2 = @inbounds op(op(a[2], vs[2]), vs[4])
        v3 = @inbounds op(op(a[3], vs[5]), vs[7])
        v4 = @inbounds op(op(a[4], vs[6]), vs[8])
    end
    return Fragment{16, 16, 16, 4, T, Unspecified, Accumulator}((v1, v2, v3, v4))
end

@inline function warp_reducerow_nxm(config, op, frag, acc = nothing)
    (; Wm, Wn, Wk) = config
    (; computeT, reduceT) = config
    return _warp_reducerow_nxm(MMAConfig{Wn, Wm, Wk, computeT, reduceT}, op, frag, acc)
end

@inline function _warp_load_1xm(
    ::Union{Type{<:MMAConfig{16, 16, 16, MT, T}}, Type{<:Config{16, 16, 16, T}}},
    lane, mi, si
) where {MT, T}
    grp = _fast_mod(lane - 1i32, Val(4))
    if lane <= 4
        indices = si + _fast_mul(grp, Val(2)) + Vec{4, Int32}((0i32, 1i32, 8i32, 9i32))
        vs = @inbounds Base.Cartesian.@ntuple 4 i -> VecElement(mi[indices[i]])
    else
        vs = (VecElement(zero(T)), VecElement(zero(T)), VecElement(zero(T)), VecElement(zero(T)))
    end
    vs = _shfl_sync(typemax(UInt32), vs, grp + 1i32)
    return Fragment{16, 16, 16, 4, T, Unspecified, Accumulator}(WMMA.flatten(vs))
end

@inline function warp_load_1xm(config, lane, mi, si)
    (; Wm, Wn, Wk) = config
    (; computeT, reduceT) = config
    return _warp_load_1xm(MMAConfig{Wn, Wm, Wk, computeT, reduceT}, lane, mi, si)
end

@inline function warp_fill_c(config, value)
    (; Wm, Wn, Wk) = config
    (; reduceT) = config
    return WMMA.fill_c(value, Config{Wn, Wm, Wk, reduceT})
end

@inline function _warp_fill_reduce_c(::Union{Type{<:MMAConfig{16, 16, 16, MT, T}}, Type{<:Config{16, 16, 16, T}}}, value) where {MT, T}
    v = convert(T, value)
    return Fragment{16, 16, 16, 4, T, Unspecified, Accumulator}(ntuple(_->v, Val(4)))
end
@inline function warp_fill_reduce_c(config, value)
    (; Wm, Wn, Wk) = config
    (; computeT, reduceT) = config
    return _warp_fill_reduce_c(MMAConfig{Wn, Wm, Wk, computeT, reduceT}, value)
end

@inline function warp_load_kxm(config, lane, Qi, sk, si)
    (; Wm, Wn, Wk) = config
    (; computeT, reduceT) = config
    return @inbounds fragment_load_b(MMAConfig{Wn, Wm, Wk, computeT, reduceT}, ColMajor, lane, Qi, sk, si)
end
@inline function warp_load_mxk(config, lane, Qi, sk, si)
    (; Wm, Wn, Wk) = config
    (; computeT, reduceT) = config
    return @inbounds fragment_load_b(MMAConfig{Wn, Wm, Wk, computeT, reduceT}, RowMajor, lane, Qi, sk, si)
end
@inline function warp_load_kxn(config, lane, Kj, sk, sj)
    (; Wm, Wn, Wk) = config
    (; computeT, reduceT) = config
    return @inbounds fragment_load_a(MMAConfig{Wn, Wm, Wk, computeT, reduceT}, RowMajor, lane, Kj, sk, sj)
end
@inline function warp_load_nxk(config, lane, Vj, sk, sj)
    (; Wm, Wn, Wk) = config
    (; computeT, reduceT) = config
    return @inbounds fragment_load_a(MMAConfig{Wn, Wm, Wk, computeT, reduceT}, ColMajor, lane, Vj, sk, sj)
end
@inline function warp_load_nxm(config, lane, Oi, sk, si)
    (; Wm, Wn, Wk) = config
    (; computeT, reduceT) = config
    return @inbounds fragment_load_c(MMAConfig{Wn, Wm, Wk, computeT, reduceT}, ColMajor, lane, Oi, sk, si)
end
@inline function warp_mma(config, lane, a, b, c)
    (; Wm, Wn, Wk) = config
    (; computeT, reduceT) = config
    return @inbounds fragment_mma(MMAConfig{Wn, Wm, Wk, computeT, reduceT}, lane, a, b, c)
end
@inline function warp_shm_write_nxm!(config, lane, Oi, o, sj, si)
    (; Wm, Wn, Wk) = config
    (; computeT, reduceT) = config
    return @inbounds fragment_store_d(MMAConfig{Wn, Wm, Wk, computeT, reduceT}, ColMajor, lane, Oi, o, sj, si)
end
@inline function warp_shm_write_1xm!(config, lane, li, l, si)
    grp = _fast_mod(lane - 1i32, Val(4))
    if lane <= 4
        indices = si + _fast_mul(grp, Val(2)) + Vec{4, Int32}((0i32, 1i32, 8i32, 9i32))
        @unroll for i = Base.OneTo{Int32}(length(indices))
            @inbounds li[indices[i]] = l[i]
        end
    end
    return nothing
end

@inline function rbroadcast(
    f,
    frag::Fragment{16, 16, 16, 8, T, Unspecified, Accumulator},
    acc::Fragment{16, 16, 16, 4, T, Unspecified, Accumulator}
) where T
    a, b, c, d = tuplesplitp(frag.x, Val(4))
    x, y = tuplesplitp(acc.x, Val(2))
    vs = tuplejoin(f.(a, x), f.(b, x), f.(c, y), f.(d, y))
    return Fragment{16, 16, 16, 8, T, Unspecified, Accumulator}(vs)
end
@inline function rbroadcast(
    f,
    acc::Fragment{16, 16, 16, 4, T, Unspecified, Accumulator},
    frag::Fragment{16, 16, 16, 8, T, Unspecified, Accumulator},
) where T
    a, b, c, d = tuplesplitp(frag.x, Val(4))
    x, y = tuplesplitp(acc.x, Val(2))
    vs = tuplejoin(f.(x, a), f.(x, b), f.(y, c), f.(y, d))
    return Fragment{16, 16, 16, 8, T, Unspecified, Accumulator}(vs)
end

@inline function fragment_type(::Type{T}, frag::Fragment{M, N, K, E, T0, L, U}) where {T, M, N, K, E, T0, L, U}
    return Fragment{M, N, K, E, T, L, U}(frag.x)
end
