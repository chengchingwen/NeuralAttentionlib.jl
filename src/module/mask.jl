module Masks

using ..NeuralAttentionlib: @imexport

@imexport import ..NeuralAttentionlib:
    apply_mask, NaiveAttenMaskOp, GenericAttenMaskOp,
    CausalMask, LocalMask, RandomMask, BandPartMask,
    GenericMask, SymLengthMask, BiLengthMask,
    BatchedMask, RepeatMask

import ..NeuralAttentionlib: AbstractAttenMask, AbstractDatalessMask, AbstractArrayMask


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
    GenericMask <: AbstractArrayMask

Generic attention mask. Just a wrapper over `AbstractArray{Bool}` for dispatch.
"""
GenericMask

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
    !m::AbstractAttenMask

Boolean not of an attention mask
"""
Base.:!(m::AbstractAttenMask)

"""
    m1::AbstractAttenMask | m2::AbstractAttenMask

logical or of two attention mask
"""
Base.:|(m1::AbstractAttenMask, m2::AbstractAttenMask)


"""
    m1::AbstractAttenMask & m2::AbstractAttenMask

logical and of two attention mask
"""
Base.:&(m1::AbstractAttenMask, m2::AbstractAttenMask)

"""
    BatchedMask(mask::AbstractArrayMask) <: AbstractWrapperMask

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
ERROR: [...]

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
    RepeatMask(mask::AbstractAttenMask, num::Int) <: AbstractWrapperMask

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
ERROR: [...]

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

end
