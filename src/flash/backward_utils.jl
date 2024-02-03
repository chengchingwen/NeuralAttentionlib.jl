@inline function block_glb2shm_reduce!(config, op, shm, dOi, glb, range, b)
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
            shm[r, c] = convert(eltype(shm), op(dOi[r, c], glb[r, c + cbase, b]))
        end
    end
    return nothing
end

@inline function block_glb2shm_rowreduce_atomic!(config, op, shm, dOi, glb, range, b)
    D = size(dOi, 1) % Int32
    d = size(glb, 1) % Int32
    N = length(range) % Int32
    cbase = first(range) - 1i32
    works = length(dOi) % Int32
    workload = cld(works, blockDim().x) % Int32
    top = threadIdx().x * workload
    base = top - workload + 1i32
    stop = min(top, works)
    acc = zero(eltype(shm))
    for i = base:stop
        c, r = fldmod1(i, D)
        @inbounds if r <= d && c <= N
            v = oftype(acc, op(dOi[r, c], glb[r, c + cbase, b]))
        else
            v = zero(acc)
        end
        acc += v
        if r == D || i == stop
            @inbounds CUDA.atomic_arrayset(shm, c, +, acc)
            acc = zero(eltype(shm))
        end
    end
    return nothing
end

function block_shm2glb_atomic!(config, op, glb, shm, range, b = 1)
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
        @inbounds CUDA.atomic_arrayset(glb, (r, c + cbase, b), op, shm[r, c])
    end
    return nothing
end
