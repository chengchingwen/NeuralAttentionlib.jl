const allow_optin_shmem = Ref{Union{Bool, Nothing}}(nothing)
optin_shmem() = allow_optin_shmem[] = true
optout_shmem() = allow_optin_shmem[] = false
const max_shmem_sizes = Dict{CuDevice, NTuple{2, Int32}}()
get_max_shmem_default(dev) = attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
get_max_shmem_possible(dev) = attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
function get_max_shmem(dev)
    max_shmems = get(max_shmem_sizes, dev, nothing)
    _optin = allow_optin_shmem[]
    optin_set = !isnothing(_optin)
    optin = optin_set ? _optin : true
    if !isnothing(max_shmems)
        shmem_max, shmem_max_possible = max_shmems
        if optin_set && !optin
            shmem_max_possible = shmem_max
        end
        return shmem_max, shmem_max_possible
    else
        support_optin = capability(dev) >= v"7"
        shmem_max = get_max_shmem_default(dev)
        if optin
            shmem_max_possible = if support_optin
                try
                    get_max_shmem_possible(dev)
                catch
                    @error "error occurs when querying the max size of opt-in shared memory, default max size is used."
                    shmem_max
                end
            else
                optin_set && @warn "The current device does not support opt-in shared memory, default max size is used."
                shmem_max
            end
        else
            shmem_max_possible = shmem_max
        end
        max_shmems = (shmem_max, shmem_max_possible)
        max_shmem_sizes[dev] = max_shmems
        return max_shmems
    end
end

struct FlashAttenConfig{MMAConfig, NT<:NamedTuple, Options}
    fields::NT
end
@inline function Base.getproperty(
    config::FlashAttenConfig{MMAConfig{Wm, Wn, Wk, computeT, reduceT}},
    sym::Symbol
) where {Wm, Wn, Wk, computeT, reduceT}
    if sym == :Wm
        return Wm
    elseif sym == :Wn
        return Wn
    elseif sym == :Wk
        return Wk
    elseif sym == :computeT
        return computeT
    elseif sym == :reduceT
        return reduceT
    else
        return getfield(getfield(config, :fields), sym)
    end
end
@inline function Base.hasproperty(
    config::FlashAttenConfig{MMAConfig{Wm, Wn, Wk, computeT, reduceT}},
    sym::Symbol
) where {Wm, Wn, Wk, computeT, reduceT}
    if sym == :Wm || sym == :Wn || sym == :Wk || sym == :computeT || sym == :reduceT
        return true
    else
        return hasproperty(getfield(config, :fields), sym)
    end
end

FlashAttenConfig{M}(kws::NamedTuple) where M = FlashAttenConfig{M, Union{}}(kws)
FlashAttenConfig{M, U}(kws::NamedTuple) where {M, U} = FlashAttenConfig{M, typeof(kws), U}(kws)

struct FlashAttenKernelConfig{static_config, DC <: FlashAttenConfig}
    dynamic_config::DC
end
@inline function FlashAttenKernelConfig(static_config, dynamic_config)
    return FlashAttenKernelConfig{static_config, typeof(dynamic_config)}(dynamic_config)
end
@inline function Base.getproperty(
    config::FlashAttenKernelConfig{static_config},
    sym::Symbol
) where {static_config}
    if hasproperty(static_config, sym)
        return getproperty(static_config, sym)
    else
        return getproperty(getfield(config, :dynamic_config), sym)
    end
end
@inline function Base.hasproperty(config::FlashAttenKernelConfig{sconfig}, sym::Symbol) where sconfig
    return hasproperty(sconfig, sym) || hasproperty(getfield(config, dynamic_config), sym)
end

function find_d_dims(Wm, Wn, Wk, dk, dv)
    Dk = cld(dk, Wk) * Wk
    Dv = cld(dv, Wn) * Wn
    D = max(Dk, Dv)
    return (; Dk, Dv, D)
end
find_d_dims(::Type{<:FlashAttenConfig{<:MMAConfig{Wm, Wn, Wk}}}, dk, dv) where {Wm, Wn, Wk} = find_d_dims(Wm, Wn, Wk, dk, dv)

use_fastmath() = CUDA.default_math_mode[] == CUDA.FAST_MATH
function get_compute_precision(reduceT)
    cuda_math_prec_sym = CUDA.default_math_precision[]
    if isnothing(cuda_math_prec_sym)
        computeT = reduceT
    elseif cuda_math_prec_sym == :TensorFloat32
        computeT = Float32
    elseif cuda_math_prec_sym == :BFloat16
        computeT = BFloat16
    elseif cuda_math_prec_sym == :Float16
        computeT = Float16
    else
        @debug "Unknown precision symbol: $cuda_math_prec_sym\nUse eltype(Q) = $reduceT"
        computeT = reduceT
    end
    return computeT
end

function total_fw_shm_size(config)
    @assert(hasproperty(config, :Br) && hasproperty(config, :Bc) && hasproperty(config, :Dk) && hasproperty(config, :Dv),
            "Cannot compute shmem size without knowing the size of inputs")
    (; Br, Bc, Dk, Dv) = config
    (; computeT, reduceT) = config
    return sizeof(computeT) * (Dk * Br + max(Dk, Dv) * Bc + Br * Bc) + sizeof(reduceT) * (Br * (Bc + Dv + 5))
end

function find_fw_max_size(Wm, Wn, Wk, computeT, reduceT, Dk, Dv, shmem_max, bcrange = nothing, brrange = nothing)
    isnothing(bcrange) && (bcrange = 2:32)
    isnothing(brrange) && (brrange = 2:32)
    D = max(Dk, Dv)
    szC = sizeof(computeT)
    szR = sizeof(reduceT)
    a = szC + szR
    b = szC * Dk + szR * Dv + szR * 5
    c = szC * D
    Brmax = Wm
    Bcmax = Wn
    shmem_possible = -1
    d = typemax(Float64)
    for bc = bcrange, br = brrange
        x = br * Wm
        y = bc * Wn
        shmem = a * x * y + b * x + c * y
        if shmem_possible >> 1 < shmem <= shmem_max
            d2 = abs(x - y) / 16 + abs(shmem_max - shmem) / 2048
            if shmem_possible == -1 || d2 < d
                d = d2
                shmem_possible = shmem
                Brmax = x
                Bcmax = y
            end
        end
    end
    shmem_possible = a * Brmax * Bcmax + b * Brmax + c * Bcmax
    return Brmax, Bcmax, shmem_possible
end
function find_fw_max_size(
    ::Type{<:FlashAttenConfig{MMAConfig{Wm, Wn, Wk, computeT, reduceT}}}, Dk, Dv, shmem_max, bcrange = nothing, brrange = nothing
) where {Wm, Wn, Wk, computeT, reduceT}
    return find_fw_max_size(Wm, Wn, Wk, computeT, reduceT, Dk, Dv, shmem_max, bcrange, brrange)
end

function build_fw_flash_attention_kernel(
    configT::Type{<:FlashAttenConfig{MMAConfig{Wm, Wn, Wk, computeT, reduceT}}},
    O, L, Q, K, V
) where {Wm, Wn, Wk, computeT, reduceT}
    dk, Nq, Bq = size(Q)
    dv, Nk, Bk = size(V)
    (; Dk, Dv, D) = find_d_dims(configT, dk, dv)
    dev = device()
    shmem_max, shmem_max_possible = get_max_shmem(dev)
    Br, Bc, shmem = find_fw_max_size(configT, Dk, Dv, shmem_max, Nk <= 32 ? 2 : nothing, Nq <= 32 ? 2 : nothing)
    if Br < Nq && Bc < Nk && shmem_max_possible > shmem_max
        Br, Bc, shmem = find_fw_max_size(configT, Dk, Dv, shmem_max_possible)
    end
    @debug Br, Bc, shmem
    mma_config = MMAConfig{Wm, Wn, Wk, computeT, reduceT}
    sconfig = FlashAttenConfig{mma_config}(
        @NamedTuple{
            Br::Int32, Bc::Int32,
            Dk::Int32, Dv::Int32}((
                Br, Bc, Dk, Dv)))
    dconfig = FlashAttenConfig{mma_config}(
        @NamedTuple{
            minval::reduceT, ss::reduceT}((
                -1e9, sqrt(inv(dk)))))
    config = FlashAttenKernelConfig(sconfig, dconfig)
    return build_fw_flash_attention_kernel(config, O, L, Q, K, V)
end

function build_fw_flash_attention_kernel(
    config::FlashAttenConfig{mma_config},
    O, L, Q, K, V
) where {mma_config}
    (; Br, Bc, Dk, Dv) = config
    sconfig = @NamedTuple{Br::Int32, Bc::Int32, Dk::Int32, Dv::Int32}((Br, Bc, Dk, Dv))
    dconfig = Base.structdiff(getfields(config, :fields), sconfig)
    kconfig = FlashAttenKernelConfig(FlashAttenConfig{mma_config}(sconfig), FlashAttenConfig{mma_config}(dconfig))
    return build_fw_flash_attention_kernel(kconfig, O, L, Q, K, V)
end

function build_fw_flash_attention_kernel(
    config::FlashAttenKernelConfig,
    O, L, Q, K, V
)
    Br = config.Br
    Nq = size(Q, 2)
    Bk = size(K, 3)
    dev = device()
    ws = warpsize(dev)
    shmem_max, shmem_max_possible = get_max_shmem(dev)
    shmem = total_fw_shm_size(config)
    @assert shmem <= shmem_max_possible
    fastmath = use_fastmath()
    kernel = @cuda(always_inline=true, fastmath=fastmath, launch=false,
                   flash_attention_forward_kernel!(config, O, L, Q, K, V))
    if shmem > shmem_max
        CUDA.cuFuncSetAttribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem)
    end
    compute_threads(threads) = max(fld(threads, ws), 1) * ws
    compute_shmem(threads) = shmem
    kernel_config = launch_configuration(kernel.fun; shmem = compute_shmem ∘ compute_threads)
    threads = compute_threads(kernel_config.threads)
    blocks = min(kernel_config.blocks, Bk * cld(Nq, Br))
    @debug kernel_config
    return config, kernel, (; threads, blocks, shmem)
end

function flash_attention_forward(Q, K, V)
    O = similar(Q, size(V, 1), Base.tail(size(Q))...)
    L = similar(O, 1, Base.tail(size(Q))...)
    reduceT = eltype(Q)
    computeT = get_compute_precision(reduceT)
    Wm = Wn = Wk = 16
    configT = FlashAttenConfig{MMAConfig{Wm, Wn, Wk, computeT, reduceT}}
    config, kernel, kernel_config = build_fw_flash_attention_kernel(configT, O, L, Q, K, V)
    kernel(config, O, L, Q, K, V; kernel_config...)
    return O, L
end

function flash_attention_forward(config, Q, K, V)
    O = similar(Q, size(V, 1), Base.tail(size(Q))...)
    L = similar(O, 1, Base.tail(size(Q))...)
    config, kernel, kernel_config = build_fw_flash_attention_kernel(config, O, L, Q, K, V)
    kernel(config, O, L, Q, K, V; kernel_config...)
    return O, L
end

function total_bw_shm_size(config)
    @assert(hasproperty(config, :Br) && hasproperty(config, :Bc) && hasproperty(config, :Dk) && hasproperty(config, :Dv),
            "Cannot compute shmem size without knowing the size of inputs")
    (; Br, Bc, Dk, Dv) = config
    (; computeT, reduceT) = config
    # dkdv_size = sizeof(computeT) * (2 * Br * Bc + (Dk + Dv) * Bc + (max(Dk, Dv) + Dv) * Br) + sizeof(reduceT) * (2 * Br + (Dk + Dv) * Bc)
    # dq_size = sizeof(computeT) * (Br * Bc + (Dk + Dv) * Br + (Dk + Dv) * Bc) + sizeof(reduceT) * (2 * Br + (Dk + Dv) * Br)
    # return max(dkdv_size, dq_size)
    return sizeof(computeT) * (2 * Br * Bc + (Dk + Dv) * Bc + (Dk + Dv) * Br) + sizeof(reduceT) * (2 * Br + Dk * Br + (Dk + Dv) * Bc)
end

function find_bw_max_size(Wm, Wn, Wk, computeT, reduceT, Dk, Dv, shmem_max, bcrange = nothing, brrange = nothing)
    isnothing(bcrange) && (bcrange = 1:32)
    isnothing(brrange) && (brrange = 1:32)
    D = max(Dk, Dv)
    szC = sizeof(computeT)
    szR = sizeof(reduceT)
    a = 2 * szC
    # b = szC * (D + Dv) + szR * 2
    b = szC * (Dk + Dv) + szR * (2 + Dk)
    c = (szC + szR) * (Dk + Dv)
    Brmax = Wm
    Bcmax = Wn
    shmem_possible = -1
    d = typemax(Float64)
    for bc = bcrange, br = brrange
        x = br * Wm
        y = bc * Wn
        shmem = a * x * y + b * x + c * y
        if shmem_possible >> 1 < shmem <= shmem_max
            d2 = abs(x - y) / 16 + abs(shmem_max - shmem) / 2048
            if shmem_possible == -1 || d2 < d
                d = d2
                shmem_possible = shmem
                Brmax = x
                Bcmax = y
            end
        end
    end
    shmem_possible = a * Brmax * Bcmax + b * Brmax + c * Bcmax
    return Brmax, Bcmax, shmem_possible
end
function find_bw_max_size(
    ::Type{<:FlashAttenConfig{MMAConfig{Wm, Wn, Wk, computeT, reduceT}}}, Dk, Dv, shmem_max, bcrange = nothing, brrange = nothing
) where {Wm, Wn, Wk, computeT, reduceT}
    return find_bw_max_size(Wm, Wn, Wk, computeT, reduceT, Dk, Dv, shmem_max, bcrange, brrange)
end

function build_bw_flash_attention_kernel(
    configT::Type{<:FlashAttenConfig{MMAConfig{Wm, Wn, Wk, computeT, reduceT}}},
    dQ, dK, dV, dO, O, L, Q, K, V
) where {Wm, Wn, Wk, computeT, reduceT}
    dk, Nq, Bq = size(Q)
    dv, Nk, Bk = size(V)
    (; Dk, Dv, D) = find_d_dims(configT, dk, dv)
    dev = device()
    shmem_max, shmem_max_possible = get_max_shmem(dev)
    Br, Bc, shmem = find_bw_max_size(configT, Dk, Dv, shmem_max, Nk <= 32 ? 2 : nothing, Nq <= 32 ? 2 : nothing)
    if Br < Nq && Bc < Nk && shmem_max_possible > shmem_max
        Br, Bc, shmem = find_bw_max_size(configT, Dk, Dv, shmem_max_possible)
    end
    @debug Br, Bc, shmem
    mma_config = MMAConfig{Wm, Wn, Wk, computeT, reduceT}
    sconfig = FlashAttenConfig{mma_config}(
        @NamedTuple{
            Br::Int32, Bc::Int32,
            Dk::Int32, Dv::Int32}((
                Br, Bc, Dk, Dv)))
    dconfig = FlashAttenConfig{mma_config}(
        @NamedTuple{
            minval::reduceT, ss::reduceT}((
                -1e9, sqrt(inv(dk)))))
    config = FlashAttenKernelConfig(sconfig, dconfig)
    return build_bw_flash_attention_kernel(config, dQ, dK, dV, dO, O, L, Q, K, V)
end

function build_bw_flash_attention_kernel(
    config::FlashAttenConfig{mma_config},
    dQ, dK, dV, dO, O, L, Q, K, V
) where {mma_config}
    (; Br, Bc, Dk, Dv) = config
    sconfig = @NamedTuple{Br::Int32, Bc::Int32, Dk::Int32, Dv::Int32}((Br, Bc, Dk, Dv))
    dconfig = Base.structdiff(getfields(config, :fields), sconfig)
    kconfig = FlashAttenKernelConfig(FlashAttenConfig{mma_config}(sconfig), FlashAttenConfig{mma_config}(dconfig))
    return build_bw_flash_attention_kernel(kconfig, dQ, dK, dV, dO, O, L, Q, K, V)
end

function build_bw_flash_attention_kernel(
    config::FlashAttenKernelConfig,
    dQ, dK, dV, dO, O, L, Q, K, V
)
    Br = config.Br
    Bc = config.Bc
    Nq = size(Q, 2)
    Nk = size(K, 2)
    Bk = size(K, 3)
    dev = device()
    ws = warpsize(dev)
    shmem_max, shmem_max_possible = get_max_shmem(dev)
    shmem = total_bw_shm_size(config)
    @assert shmem <= shmem_max_possible
    fastmath = use_fastmath()
    kernel = @cuda(always_inline=true, fastmath=fastmath, launch=false,
                   flash_attention_backward_kernel!(config, dQ, dK, dV, dO, O, L, Q, K, V))
    if shmem > shmem_max
        CUDA.cuFuncSetAttribute(kernel.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem)
    end
    compute_threads(threads) = max(fld(threads, ws), 1) * ws
    compute_shmem(threads) = shmem
    kernel_config = launch_configuration(kernel.fun; shmem = compute_shmem ∘ compute_threads)
    threads = compute_threads(kernel_config.threads)
    blocks = min(kernel_config.blocks, Bk * cld(Nk, Bc))
    @debug kernel_config
    return config, kernel, (; threads, blocks, shmem)
end

function flash_attention_backward(dO, O, L, Q, K, V)
    dQ = zero(Q)
    dK = similar(K)
    dV = similar(V)
    reduceT = eltype(Q)
    computeT = get_compute_precision(reduceT)
    Wm = Wn = Wk = 16
    configT = FlashAttenConfig{MMAConfig{Wm, Wn, Wk, computeT, reduceT}}
    config, kernel, kernel_config = build_bw_flash_attention_kernel(configT, dQ, dK, dV, dO, O, L, Q, K, V)
    kernel(config, dQ, dK, dV, dO, O, L, Q, K, V; kernel_config...)
    return dQ, dK, dV
end

function flash_attention_backward(config, dO, O, L, Q, K, V)
    dQ = zero(Q)
    dK = similar(K)
    dV = similar(V)
    config, kernel, kernel_config = build_bw_flash_attention_kernel(config, dQ, dK, dV, dO, O, L, Q, K, V)
    kernel(config, dQ, dK, dV, dO, O, L, Q, K, V; kernel_config...)
    return dQ, dK, dV
end
