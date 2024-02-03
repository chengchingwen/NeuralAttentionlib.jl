using CUDA
using CUDA: i32
using KernelAbstractions.Extras: @unroll

@inline function create_forward_dynamic_shm(config)
    (; Br, Bc) = config
    (; Wk, Wn) = config
    (; Dk, Dv) = config
    (; computeT, reduceT) = config
    D = max(Dk, Dv)
    offset = 0i32
    S = @inbounds CuDynamicSharedArray(reduceT, (Bc, Br), offset)
    offset += sizeof(S) % Int32
    P = @inbounds CuDynamicSharedArray(computeT, (Bc, Br), offset)
    offset += sizeof(P) % Int32
    pi = @inbounds CuDynamicSharedArray(reduceT, (1, Br), offset)
    offset += sizeof(pi) % Int32
    li = @inbounds CuDynamicSharedArray(reduceT, (1, Br), offset)
    offset += sizeof(li) % Int32
    li2 = @inbounds CuDynamicSharedArray(reduceT, (1, Br), offset)
    offset += sizeof(li2) % Int32
    mi = @inbounds CuDynamicSharedArray(reduceT, (1, Br), offset)
    offset += sizeof(mi) % Int32
    mi2 = @inbounds CuDynamicSharedArray(reduceT, (1, Br), offset)
    offset += sizeof(mi2) % Int32
    Qi = @inbounds CuDynamicSharedArray(computeT, (Dk, Br), offset)
    offset += sizeof(Qi) % Int32
    KVj = @inbounds CuDynamicSharedArray(computeT, (D, Bc), offset)
    offset += sizeof(KVj) % Int32
    Oi = @inbounds CuDynamicSharedArray(reduceT, (Dv, Br), offset)
    return (; Qi, KVj, Oi, S, P, pi, li, li2, mi, mi2)
end

@inline function compute_KᵀQ(config, grps, shms, sizes)
    (; Br, Bc, Wm, Wn, Wk, Dk) = config
    (; computeT, reduceT) = config
    (; ss) = config
    (; Qi, KVj, S, mi, mi2) = shms
    (; W, warp, lane) = grps
    (; klen, qlen, tr, tc, tdk) = sizes
    rstop = min(Br, qlen)
    cstop = min(Bc, klen)
    np = tr * tc
    @unroll for ti_tj = warp:W:np
        ti, tj = fldmod1(ti_tj, tc)
        si = (ti - 1i32) * Wm + 1i32
        sj = (tj - 1i32) * Wn + 1i32
        (si > rstop || sj > cstop) && continue
        s = warp_fill_c(config, zero(reduceT))
        @unroll for tk = 1i32:tdk
            sk = (tk - 1i32) * Wk + 1i32
            sk > Dk && break
            q = warp_load_kxm(config, lane, Qi, sk, si)
            k = warp_load_kxn(config, lane, KVj, sk, sj)
            s = warp_mma(config, lane, k, q, s) # s += k' * q
        end
        s = s .* ss
        warp_shm_write_nxm!(config, lane, S, s, sj, si)
        m = warp_reducerow_nxm(config, max, s) # 8 -> 4
        warp_shm_reduce_1xm_atomic!(config, lane, max, mi2, m, si)
    end
    return nothing
end

@inline function compute_exp_S(config, grps, shms, sizes)
    (; Br, Bc, Wm, Wn) = config
    (; computeT, reduceT) = config
    (; minval) = config
    (; S, P, pi, mi2) = shms
    (; W, warp, lane) = grps
    (; sfull, klen, qlen, tr, tc, tdk) = sizes
    rstop = min(Br, qlen)
    cstop = min(Bc, klen)
    np = tr * tc
    @unroll for ti_tj = warp:W:np
        ti, tj = fldmod1(ti_tj, tc)
        si = (ti - 1i32) * Wm + 1i32
        sj = (tj - 1i32) * Wn + 1i32
        (si > rstop || sj > cstop) && continue
        m = warp_load_1xm(config, lane, mi2, si)
        p = warp_load_nxm(config, lane, S, sj, si)
        p = rbroadcast(-, p, m)
        if !sfull
            pmask = warp_gen_pmask(config, lane, klen, qlen, sj, si)
            p = _elseif.(p, pmask, minval) # @. ifelse(pmask, p, minval)
        end
        p = exp.(p) # @. exp(p - m)
        ps = warp_reducerow_nxm(config, +, p) # 8 -> 4
        p0 = fragment_type(computeT, p)
        warp_shm_write_nxm!(config, lane, P, p0, sj, si)
        warp_shm_reduce_1xm_atomic!(config, lane, +, pi, ps, si)
    end
    return nothing
end

@inline function compute_exp_m_O_VP(config, grps, shms, sizes)
    (; Br, Bc, Wm, Wn, Wk, Dv) = config
    (; computeT, reduceT) = config
    (; mi, mi2, pi, li, li2, P, KVj, Oi) = shms
    (; W, warp, lane) = grps
    (; klen, qlen, tr, tc, tdv) = sizes
    (; j) = grps
    (; Tc) = sizes
    rstop = min(Br, qlen)
    cstop = min(Bc, klen)
    np = tr * tdv
    is_last = j == Tc
    @unroll for ti_tk = warp:W:np
        ti, tk = fldmod1(ti_tk, tdv)
        si = (ti - 1i32) * Wm + 1i32
        sk = (tk - 1i32) * Wn + 1i32
        (si > rstop || sk > Dv) && continue
        mp = warp_load_1xm(config, lane, mi, si)
        m  = warp_load_1xm(config, lane, mi2, si)
        ps = warp_load_1xm(config, lane, pi, si)
        l = warp_load_1xm(config, lane, li, si)
        mdiff = mp .- m
        em = exp.(mdiff) #@. exp(mp - m)
        l = em .* l
        l = l .+ ps # @. em * l + ps
        o = warp_load_nxm(config, lane, Oi, sk, si)
        o = rbroadcast(*, em, o) # @. em * o
        if is_last
            m0 = CUDA.log.(l)
            m = m .+ m0
            warp_shm_write_1xm!(config, lane, li2, m, si)
            l = inv.(l)
            @unroll for tj = 1i32:tc
                sj = (tj - 1i32) * Wk + 1i32
                sj > cstop && break
                p = warp_load_kxm(config, lane, P, sj, si)
                v = warp_load_nxk(config, lane, KVj, sk, sj)
                o = warp_mma(config, lane, v, p, o) # o += v * p
            end
            o = rbroadcast(*, l, o) # @. l * o
        else
            warp_shm_write_1xm!(config, lane, li2, l, si)
            @unroll for tj = 1i32:tc
                sj = (tj - 1i32) * Wk + 1i32
                sj > cstop && break
                p = warp_load_kxm(config, lane, P, sj, si)
                v = warp_load_nxk(config, lane, KVj, sk, sj)
                o = warp_mma(config, lane, v, p, o) # o += v * p
            end
        end
        warp_shm_write_nxm!(config, lane, Oi, o, sk, si)
    end
    return nothing
end

function flash_attention_forward_kernel!(config, O, L, Q, K, V)
    (; Br, Bc, Dk, Dv) = config # share memory size
    Wm = config.Wm % Int32 # WMMA size
    Wn = config.Wn % Int32
    Wk = config.Wk % Int32
    (; computeT, reduceT) = config
    (; minval) = config
    # warp groups
    threads = blockDim().x
    ws = warpsize()
    W = fld(threads, ws)
    index = threadIdx().x
    warp, lane = fldmod1(index, ws)
    grps = (; W, index, warp, lane)
    # chunks
    B = size(O, 3) % Int32
    Nq = size(Q, 2) % Int32
    Nk = size(K, 2) % Int32
    dk = size(Q, 1) % Int32
    dv = size(V, 1) % Int32
    Tr = cld(Nq, Br) % Int32
    Tc = cld(Nk, Bc) % Int32
    tr = cld(Br, Wm) % Int32
    tc = cld(Bc, Wn) % Int32
    tdk = cld(Dk, Wk) % Int32
    tdv = cld(Dv, Wn) % Int32
    sizes = (; Nq, Nk, Tr, Tc, dk, dv, tr, tc, tdk, tdv)
    # allocs shms
    shms = create_forward_dynamic_shm(config)
    # batch loop
    stride = gridDim().x
    bidx = blockIdx().x
    NP = B * Tr
    for b_i = bidx:stride:NP
        b, i = fldmod1(b_i, Tr)
        qfull, qrange, qlen = chunkrange(Br, Nq, i)
        sizes = merge(sizes, (; qfull, qrange, qlen,))
        block_glb2shm!(config, shms.Qi, Q, qrange, b)
        block_shm_fill!(shms.mi, minval)
        block_shm_fill!(shms.mi2, minval)
        block_shm_fill!(shms.li, zero(reduceT))
        block_shm_fill!(shms.li2, zero(reduceT))
        block_shm_fill!(shms.Oi, zero(reduceT))
        for j = 1i32:Tc
            grps = merge(grps, (; j))
            kfull, krange, klen = chunkrange(Bc, Nk, j)
            sfull = qfull & kfull
            sizes = merge(sizes, (; kfull, krange, klen, sfull))
            block_glb2shm!(config, shms.KVj, K, krange, b)
            block_shm_fill!(shms.pi, zero(reduceT))
            sync_threads() # Q, K, S
            # S = K^T * Q * dk^-1/2
            compute_KᵀQ(config, grps, shms, sizes)
            sync_threads() # S, m
            # P = exp(S - m)
            block_glb2shm!(config, shms.KVj, V, krange, b)
            compute_exp_S(config, grps, shms, sizes)
            sync_threads() # P, pi, V
            # O = exp(mp - m) * O + V * P
            # O *= l
            compute_exp_m_O_VP(config, grps, shms, sizes)
            shms = merge(shms, (; mi = shms.mi2, mi2 = shms.mi, li = shms.li2, li2 = shms.li))
            sync_threads()
        end # Tc loop
        block_shm2glb!(config, O, shms.Oi, qrange, b)
        block_shm2glb!(config, L, shms.li, qrange, b)
    end
    return nothing
end
