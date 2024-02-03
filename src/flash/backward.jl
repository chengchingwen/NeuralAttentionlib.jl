using CUDA
using CUDA: i32
using KernelAbstractions.Extras: @unroll

function create_backward_dynamic_shm(config)
    (; Br, Bc) = config
    (; Wk, Wn) = config
    (; Dk, Dv) = config
    (; computeT, reduceT) = config
    D = max(Dk, Dv)
    offset = 0i32
    Li = @inbounds CuDynamicSharedArray(reduceT, (1, Br), offset)
    offset += sizeof(Li) % Int32
    Di = @inbounds CuDynamicSharedArray(reduceT, (1, Br), offset)
    offset += sizeof(Di) % Int32
    P = @inbounds CuDynamicSharedArray(computeT, (Bc, Br), offset)
    offset += sizeof(P) % Int32
    dS = @inbounds CuDynamicSharedArray(computeT, (Bc, Br), offset)
    offset += sizeof(dS) % Int32
    Qi = @inbounds CuDynamicSharedArray(computeT, (Dk, Br), offset)
    offset += sizeof(Qi) % Int32
    dQi = @inbounds CuDynamicSharedArray(reduceT, (Dk, Br), offset)
    offset += sizeof(dQi) % Int32
    Kj = @inbounds CuDynamicSharedArray(computeT, (Dk, Bc), offset)
    offset += sizeof(Kj) % Int32
    dKj = @inbounds CuDynamicSharedArray(reduceT, (Dk, Bc), offset)
    offset += sizeof(dKj) % Int32
    Vj = @inbounds CuDynamicSharedArray(computeT, (Dv, Bc), offset)
    offset += sizeof(Vj) % Int32
    dVj = @inbounds CuDynamicSharedArray(reduceT, (Dv, Bc), offset)
    offset += sizeof(dVj) % Int32
    dOi = @inbounds CuDynamicSharedArray(computeT, (Dv, Br), offset)
    offset += sizeof(dOi) % Int32
    return (; P, dS, Kj, dKj, Vj, dVj, dOi, Qi, dQi, Li, Di)
end

@inline function compute_P_and_dS(config, grps, shms, sizes)
    (; Br, Bc, Wm, Wn, Wk, Dk, Dv) = config
    (; computeT, reduceT) = config
    (; ss, minval) = config
    (; Qi, Kj, Vj, Li, dOi, P, dS, Di) = shms
    (; W, warp, lane) = grps
    (; sfull, klen, qlen, tr, tc, tdk, tdv) = sizes
    rstop = min(Br, qlen)
    cstop = min(Bc, klen)
    np = tr * tc
    @unroll for ti_tj = warp:W:np
        ti, tj = fldmod1(ti_tj, tc)
        sj = (tj - 1i32) * Wn + 1i32
        si = (ti - 1i32) * Wm + 1i32
        (si > rstop || sj > cstop) && continue
        d = warp_load_1xm(config, lane, Di, si)
        dp = warp_fill_c(config, zero(reduceT))
        @unroll for tk = 1i32:tdv
            sk = (tk - 1i32) * Wk + 1i32
            sk > Dv && break
            doi = warp_load_kxm(config, lane, dOi, sk, si)
            v = warp_load_kxn(config, lane, Vj, sk, sj)
            dp = warp_mma(config, lane, v, doi, dp) # dp += v' * doi
        end
        dpd = rbroadcast(-, dp, d)
        l = warp_load_1xm(config, lane, Li, si)
        s = warp_fill_c(config, zero(reduceT))
        @unroll for tk = 1i32:tdk
            sk = (tk - 1i32) * Wk + 1i32
            sk > Dk && break
            q = warp_load_kxm(config, lane, Qi, sk, si)
            k = warp_load_kxn(config, lane, Kj, sk, sj)
            s = warp_mma(config, lane, k, q, s) # s += k' * q
        end
        s = s .* ss
        p = rbroadcast(-, s, l)
        if !sfull
            pmask = warp_gen_pmask(config, lane, klen, qlen, sj, si)
            p = _elseif.(p, pmask, minval) # @. ifelse(pmask, p, minval)
        end
        p = exp.(p)
        warp_shm_write_nxm!(config, lane, P, fragment_type(computeT, p), sj, si)
        ds = p .* dpd
        ds = ds .* ss
        warp_shm_write_nxm!(config, lane, dS, fragment_type(computeT, ds), sj, si)
    end
    return nothing
end

@inline function compute_dQ_KdS(config, grps, shms, sizes)
    (; Br, Bc, Wm, Wn, Wk, Dk) = config
    (; computeT, reduceT) = config
    (; ss) = config
    (; dQi, Kj, dS) = shms
    (; W, warp, lane) = grps
    (; klen, qlen) = sizes
    tr = cld(Br, Wk) % Int32
    tc = cld(Bc, Wm) % Int32
    tdk = cld(Dk, Wn) % Int32
    rstop = min(Br, qlen)
    cstop = min(Bc, klen)
    np = tr * tdk
    @unroll for ti_tk = warp:W:np
        ti, tk = fldmod1(ti_tk, tdk)
        si = (ti - 1i32) * Wm + 1i32
        sk = (tk - 1i32) * Wn + 1i32
        (si > rstop || sk > Dk) && continue
        dq = warp_fill_c(config, zero(reduceT))
        @unroll for tj = 1i32:tc
            sj = (tj - 1i32) * Wk + 1i32
            sj > cstop && break
            ds = warp_load_kxm(config, lane, dS, sj, si)
            k = warp_load_nxk(config, lane, Kj, sk, sj)
            dq = warp_mma(config, lane, k, ds, dq) # dQi += K * dSi
        end
        warp_shm_write_nxm!(config, lane, dQi, dq, sk, si)
    end
    return nothing
end

@inline function compute_dV_dOPᵀ(config, grps, shms, sizes)
    (; Br, Bc, Wm, Wn, Wk, Dv) = config
    (; computeT, reduceT) = config
    (; P, dVj, dOi) = shms
    (; W, warp, lane) = grps
    (; klen, qlen) = sizes
    (; Tc) = sizes
    tc = cld(Bc, Wm) % Int32
    tdv = cld(Dv, Wn) % Int32
    tr = cld(Br, Wk) % Int32
    rstop = min(Br, qlen)
    cstop = min(Bc, klen)
    np = tc * tdv
    @unroll for tj_tk = warp:W:np
        tj, tk = fldmod1(tj_tk, tdv)
        sj = (tj - 1i32) * Wm + 1i32
        sk = (tk - 1i32) * Wn + 1i32
        (sj > cstop || sk > Dv) && continue
        dv = warp_load_nxm(config, lane, dVj, sk, sj)
        @unroll for ti = 1i32:tr
            si = (ti - 1i32) * Wk + 1i32
            si > rstop && break
            p = warp_load_mxk(config, lane, P, sj, si)
            doi = warp_load_nxk(config, lane, dOi, sk, si)
            dv = warp_mma(config, lane, doi, p, dv) # dvj += doi * p'
        end
        warp_shm_write_nxm!(config, lane, dVj, dv, sk, sj)
    end
    return nothing
end

@inline function compute_dK_QdSᵀ(config, grps, shms, sizes)
    (; Br, Bc, Wm, Wn, Wk, Dk) = config
    (; computeT, reduceT) = config
    (; ss) = config
    (; Qi, dKj, dS) = shms
    (; W, warp, lane) = grps
    (; klen, qlen) = sizes
    tr = cld(Br, Wk) % Int32
    tc = cld(Bc, Wm) % Int32
    tdk = cld(Dk, Wn) % Int32
    rstop = min(Br, qlen)
    cstop = min(Bc, klen)
    np = tc * tdk
    @unroll for tj_tk = warp:W:np
        tj, tk = fldmod1(tj_tk, tdk)
        sj = (tj - 1i32) * Wm + 1i32
        sk = (tk - 1i32) * Wn + 1i32
        (sj > cstop || sk > Dk) && continue
        dk = warp_load_nxm(config, lane, dKj, sk, sj)
        @unroll for ti = 1i32:tr
            si = (ti - 1i32) * Wk + 1i32
            si > rstop && break
            ds = warp_load_mxk(config, lane, dS, sj, si)
            q = warp_load_nxk(config, lane, Qi, sk, si)
            dk = warp_mma(config, lane, q, ds, dk) # dKj += Q * dSi'
        end
        warp_shm_write_nxm!(config, lane, dKj, dk, sk, sj)
    end
    return nothing
end

function flash_attention_backward_kernel!(config, dQ, dK, dV, dO, O, L, Q, K, V)
    (; Br, Bc, Dk, Dv) = config
    Wm = config.Wm % Int32 # WMMA size
    Wn = config.Wn % Int32
    Wk = config.Wk % Int32
    (; computeT, reduceT) = config
    (; minval, ss) = config
    threads = blockDim().x
    ws = warpsize()
    W = fld(threads, ws)
    index = threadIdx().x
    warp, lane = fldmod1(index, ws)
    grps = (; W, index, warp, lane)
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
    shms = create_backward_dynamic_shm(config)
    stride = gridDim().x
    bidx = blockIdx().x
    NP = B * Tc
    for b_j = bidx:stride:NP
        b, j = fldmod1(b_j, Tc)
        kfull, krange, klen = chunkrange(Bc, Nk, j)
        sizes = merge(sizes, (; kfull, krange, klen))
        block_glb2shm!(config, shms.Kj, K, krange, b)
        block_glb2shm!(config, shms.Vj, V, krange, b)
        block_shm_fill!(shms.dKj, zero(reduceT))
        block_shm_fill!(shms.dVj, zero(reduceT))
        block_shm_fill!(shms.Di, zero(reduceT))
        sync_threads()
        for i = 1i32:Tr
            qfull, qrange, qlen = chunkrange(Br, Nq, i)
            sfull = qfull & kfull
            sizes = merge(sizes, (; sfull, qfull, qrange, qlen,))
            block_glb2shm!(config, shms.dOi, dO, qrange, b)
            block_glb2shm!(config, shms.Li, L, qrange, b)
            block_glb2shm_rowreduce_atomic!(config, *, shms.Di, shms.dOi, O, qrange, b)
            block_glb2shm!(config, shms.Qi, Q, qrange, b)
            sync_threads()
            compute_P_and_dS(config, grps, shms, sizes)
            sync_threads()
            block_shm_fill!(shms.Di, zero(reduceT))
            compute_dV_dOPᵀ(config, grps, shms, sizes)
            compute_dK_QdSᵀ(config, grps, shms, sizes)
            compute_dQ_KdS(config, grps, shms, sizes)
            sync_threads()
            block_shm2glb_atomic!(config, +, dQ, shms.dQi, qrange, b)
            sync_threads()
        end
        block_shm2glb!(config, dK, shms.dKj, krange, b)
        block_shm2glb!(config, dV, shms.dVj, krange, b)
        sync_threads()
    end
    return nothing
end
