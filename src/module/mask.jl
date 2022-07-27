module Masks

using ..NeuralAttentionlib: @imexport

@imexport import ..NeuralAttentionlib:
    apply_mask, NaiveMaskOp, GenericMaskOp,
    CausalMask, LocalMask, RandomMask, BandPartMask,
    GenericAttenMask, SymLengthMask, BiLengthMask,
    BatchedMask, RepeatMask, getmask,
    GenericSequenceMask, LengthMask

import ..NeuralAttentionlib: AbstractMask, AbstractSequenceMask, AbstractAttenMask,
    AbstractDatalessMask, AbstractArrayMask, AttenMask

"""
    AbstractDatalessMask <: AbstractAttenMask

Abstract type for mask without array data.
"""
AbstractDatalessMask

"""
    AbstractArrayMask <: AbstractAttenMask

Abstract type for mask with array data
"""
AbstractArrayMask

"""
    CausalMask() <: AbstractDatalessMask

Attention mask that block the future values.

Similar to applying `LinearAlgebra.triu!` on the score matrix
"""
CausalMask

"""
    LocalMask(width::Int) <: AbstractDatalessMask

Attention mask that only allow local (diagonal like) values to pass.

`width` should be ≥ 0 and `A .* LocalMask(1)` is similar to `Diagonal(A)`
"""
LocalMask

"""
    RandomMask(p::Float64) <: AbstractDatalessMask

Attention mask that block value randomly.

`p` specify the percentage of value to block. e.g. `A .* RandomMask(0)` is equivalent to `identity(A)` and
 `A .* RandomMask(1)` is equivalent to `zero(A)`.
"""
RandomMask

"""
    BandPartMask(l::Int, u::Int) <: AbstractDatalessMask

Attention mask that only allow [band_part](https://www.tensorflow.org/api_docs/python/tf/linalg/band_part)
 values to pass.
"""
BandPartMask

"""
    GenericAttenMask <: AbstractArrayMask

Generic attention mask. Just a wrapper over `AbstractArray{Bool}` for dispatch.
"""
GenericAttenMask

"""
    SymLengthMask(len::AbstractArray{Int, N}) <: AbstractArrayMask

Attention mask specified by an array of integer that indicate the length dimension size.
 assuming *Query* length and *Key* length are the same.

# Example

```julia
julia> m = SymLengthMask([2,3])
SymLengthMask{1, Vector{Int32}}(Int32[2, 3])

julia> trues(3,3, 2) .* m
3×3×2 BitArray{3}:
[:, :, 1] =
 1  1  0
 1  1  0
 0  0  0

[:, :, 2] =
 1  1  1
 1  1  1
 1  1  1

```

See also: [`BiLengthMask`](@ref), [`BatchedMask`](@ref), [`RepeatMask`](@ref)
"""
SymLengthMask

"""
    BiLengthMask(q_len::A, k_len::A) where {A <: AbstractArray{Int, N}} <: AbstractArrayMask

Attention mask specified by two arrays of integer that indicate the length dimension size.

# Example

```julia
julia> bm = BiLengthMask([2,3], [3, 5])
BiLengthMask{1, Vector{Int32}}(Int32[2, 3], Int32[3, 5])

julia> trues(5,5, 2) .* bm
5×5×2 BitArray{3}:
[:, :, 1] =
 1  1  0  0  0
 1  1  0  0  0
 1  1  0  0  0
 0  0  0  0  0
 0  0  0  0  0

[:, :, 2] =
 1  1  1  0  0
 1  1  1  0  0
 1  1  1  0  0
 1  1  1  0  0
 1  1  1  0  0

```

See also: [`SymLengthMask`](@ref), [`BatchedMask`](@ref), [`RepeatMask`](@ref)
"""
BiLengthMask

"""
    !m::AbstractMask

Boolean not of an attention mask
"""
Base.:!(m::AbstractMask)

"""
    m1::AbstractMask | m2::AbstractMask

logical or of two attention mask
"""
Base.:|(m1::AbstractMask, m2::AbstractMask)


"""
    m1::AbstractMask & m2::AbstractMask

logical and of two attention mask
"""
Base.:&(m1::AbstractMask, m2::AbstractMask)

"""
    BatchedMask(mask::AbstractMask) <: AbstractWrapperMask

Attention mask wrapper over array mask for applying the same mask within the same batch.

# Example

```julia
julia> m = SymLengthMask([2,3])
SymLengthMask{1, Vector{Int32}}(Int32[2, 3])

julia> trues(3,3, 2) .* m
3×3×2 BitArray{3}:
[:, :, 1] =
 1  1  0
 1  1  0
 0  0  0

[:, :, 2] =
 1  1  1
 1  1  1
 1  1  1

julia> trues(3,3, 2, 2) .* m
ERROR: DimensionMismatch("arrays could not be broadcast to a common size; mask require ndims(A) == 3")
Stacktrace:
[...]

julia> trues(3,3, 2, 2) .* BatchedMask(m) # 4-th dim become batch dim
3×3×2×2 BitArray{4}:
[:, :, 1, 1] =
 1  1  0
 1  1  0
 0  0  0

[:, :, 2, 1] =
 1  1  0
 1  1  0
 0  0  0

[:, :, 1, 2] =
 1  1  1
 1  1  1
 1  1  1

[:, :, 2, 2] =
 1  1  1
 1  1  1
 1  1  1

```
"""
BatchedMask

"""
    RepeatMask(mask::AbstractMask, num::Int) <: AbstractWrapperMask

Attention mask wrapper over array mask for doing inner repeat on the last dimension.

# Example

```julia
julia> m = SymLengthMask([2,3])
SymLengthMask{1, Vector{Int32}}(Int32[2, 3])

julia> trues(3,3, 2) .* m
3×3×2 BitArray{3}:
[:, :, 1] =
 1  1  0
 1  1  0
 0  0  0

[:, :, 2] =
 1  1  1
 1  1  1
 1  1  1

julia> trues(3,3, 4) .* m
ERROR: DimensionMismatch("arrays could not be broadcast to a common size; mask require 3-th dimension to be 2, but get 4")
Stacktrace:
[...]

julia> trues(3,3, 4) .* RepeatMask(m, 2)
3×3×4 BitArray{3}:
[:, :, 1] =
 1  1  0
 1  1  0
 0  0  0

[:, :, 2] =
 1  1  0
 1  1  0
 0  0  0

[:, :, 3] =
 1  1  1
 1  1  1
 1  1  1

[:, :, 4] =
 1  1  1
 1  1  1
 1  1  1

```
"""
RepeatMask

"""
    getmask(m::AbstractMask, score, scale = 1)

Convert `m` into mask array of `AbstractArray` for `score` with `scale`.

# Example
```julia
julia> getmask(CausalMask(), randn(7,7), 2)
7×7 Matrix{Float64}:
 2.0  2.0  2.0  2.0  2.0  2.0  2.0
 0.0  2.0  2.0  2.0  2.0  2.0  2.0
 0.0  0.0  2.0  2.0  2.0  2.0  2.0
 0.0  0.0  0.0  2.0  2.0  2.0  2.0
 0.0  0.0  0.0  0.0  2.0  2.0  2.0
 0.0  0.0  0.0  0.0  0.0  2.0  2.0
 0.0  0.0  0.0  0.0  0.0  0.0  2.0

```
"""
getmask

end
