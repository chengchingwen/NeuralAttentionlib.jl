# A simple, fast, reproducible, but reasonably weak RNG for generating random number per array elements.
#  The basic idea is to create a unique RNG for each array positions, so the RNG state is only
#  shared across multiple calls at the same location. This eliminate the need for accessing RNG
#  state from different locations. The initail seeds are created by a another RNG (like `Random.default_rng()`).
#
# We use a Combined LCG (modulo 2^32) and a hashed position number to create the unique RNG for each position.
#  The additive constants are multipled with the hashed position number to create different random
#  streams for each position.
abstract type PosRNGAlg end
randstate(alg::PosRNGAlg) = randstate(Random.default_rng(), alg)
struct CPLCGm32{T<:Tuple{Vararg{UInt32}}} <: PosRNGAlg
    a::T
    c::T
end
CPLCGm32() = CPLCGm32(UInt32.((22695477, 69069, 1664525,)), UInt32.((1, 12345, 1013904223,)))
setpos(lcg::CPLCGm32, pos::UInt32) = CPLCGm32(lcg.a, lcg.c .* pos)
i2fp(::CPLCGm32, x::UInt32) = 2.3283064f-10 * Float32(x)
randstate(rng::AbstractRNG, lcg::CPLCGm32) = ntuple(i->rand(rng, UInt32), Val(length(lcg.a)))
adapt_structure(to::IndexerAdaptor, alg::CPLCGm32) = PosRNG(alg, randstate(to.rng, alg))

struct PosRNG{A <: PosRNGAlg, S}
    alg::A
    state::S
end
PosRNG() = PosRNG(CPLCGm32)
PosRNG(rng::AbstractRNG) = PosRNG(rng, CPLCGm32())
PosRNG(alg::PosRNGAlg) = PosRNG(Random.defalut_rng(), alg)
PosRNG(rng::AbstractRNG, alg::PosRNGAlg) = PosRNG(alg, randstate(rng, alg))
setpos(rng::PosRNG, pos) = PosRNG(setpos(rng.alg, pos), rng.state)
i2fp(rng::PosRNG, x) = i2fp(rng.alg, x)
adapt_structure(to::IndexerAdaptor, rng::PosRNG) = isnothing(to.rng) ? rng : PosRNG(rng.alg, randstate(to.rng, rng.alg))

@inline function rngstep(rng::CPLCGm32{T}, seed::T, pos::UInt32) where T
    state = fma.(rng.a, seed, rng.c)
    val = xor(state...)
    return val, state
end
@inline function rngstep(rng::PosRNG, pos)
    val, state = rngstep(rng.alg, rng.state, pos)
    rng2 = PosRNG(rng.alg, state)
    return val, rng2
end

@inline prand(rng::PosRNG, pos) = prand(Float32, rng, pos)
@inline prand(::Type{UInt32}, rng::PosRNG, pos) = rngstep(rng, pos)
@inline function prand(::Type{Float32}, rng::PosRNG, pos)
    val, rng2 = prand(UInt32, rng, pos)
    val = i2fp(rng, val)
    return val, rng2
end
