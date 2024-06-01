module Masks

using ..NeuralAttentionlib: @imexport

@imexport import ..NeuralAttentionlib:
    apply_mask, NaiveMaskOp, GenericMaskOp,
    CausalMask, LocalMask, RandomMask, BandPartMask,
    GenericAttenMask, SymLengthMask, BiLengthMask,
    RevSymLengthMask, RevBiLengthMask, BiSeqMask,
    BatchedMask, RepeatMask, getmask, GetIndexer,
    GenericSeqMask, LengthMask, RevLengthMask

import ..NeuralAttentionlib: AbstractMask, AbstractSeqMask, AbstractAttenMask, AbstractDatalessMask, AbstractArrayMask,
    MASKDATA, MASKTYPE, DATALESS, ARRAYDATA, MIXDATA, ATTENTION, SEQUENCE, MIXTYPE,
    AttenMask, SeqMask, AxesConstraint, Indexer, GetIndexer,
    CombinedMask, FlipMask, lengths, GenericSequenceMask, BiSequenceMask

"""
    Indexer(m::AbstractMask, size::Dims{N}) <: AbstractArray{Bool, N}
    Indexer(m::AbstractMask, size::Dims{N}, scale::T) <: AbstractArray{T, N}

A lazy array-like object that "materialize" the mask `m` with `size` and a optional `scale` without size check.

See also: [`GetIndexer`](@ref)
"""
Indexer

"""
    GetIndexer(m::AbstractMask, destsize::Dims{N})

Return the [`Indexer`](@ref) of `m` and check if the mask `m` can be applied to an array with size `destsize`.
"""
GetIndexer

"""
    AbstractMask

Abstract type for mask data.
"""
AbstractMask

"""
    AbstractSeqMask <: AbstractMask

Abstract type for mask data specifically for sequence.
"""
AbstractSeqMask

"""
    AbstractAttenMask <: AbstractMask

Abstract type for mask data specifically for attention.
"""
AbstractAttenMask

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
    AttenMask(m::AbstractMask)

Convert mask into corresponding attention mask.

    AttenMask(q_mask::AbstractSeqMask, k_mask::AbstractSeqMask)

Create a attention mask from 2 sequence masks specific the sequence mask for "query" and "key".
"""
AttenMask

"""
    CausalMask() <: AbstractAttenMask{DATALESS}

Attention mask that block the future values.

Similar to applying `LinearAlgebra.triu!` on the score matrix

# Example

```julia-repl
julia> trues(10, 10) .* CausalMask()
10×10 BitMatrix:
 1  1  1  1  1  1  1  1  1  1
 0  1  1  1  1  1  1  1  1  1
 0  0  1  1  1  1  1  1  1  1
 0  0  0  1  1  1  1  1  1  1
 0  0  0  0  1  1  1  1  1  1
 0  0  0  0  0  1  1  1  1  1
 0  0  0  0  0  0  1  1  1  1
 0  0  0  0  0  0  0  1  1  1
 0  0  0  0  0  0  0  0  1  1
 0  0  0  0  0  0  0  0  0  1
```
"""
CausalMask

"""
    LocalMask(width::Int) <: AbstractAttenMask{DATALESS}

Attention mask that only allow local (diagonal like) values to pass.

`width` should be ≥ 0 and `A .* LocalMask(1)` is similar to `Diagonal(A)`

# Example

```julia-repl
julia> trues(10, 10) .* LocalMask(3)
10×10 BitMatrix:
 1  1  1  0  0  0  0  0  0  0
 1  1  1  1  0  0  0  0  0  0
 1  1  1  1  1  0  0  0  0  0
 0  1  1  1  1  1  0  0  0  0
 0  0  1  1  1  1  1  0  0  0
 0  0  0  1  1  1  1  1  0  0
 0  0  0  0  1  1  1  1  1  0
 0  0  0  0  0  1  1  1  1  1
 0  0  0  0  0  0  1  1  1  1
 0  0  0  0  0  0  0  1  1  1
```
"""
LocalMask

"""
    RandomMask(p::Float32) <: AbstractAttenMask{DATALESS}

Attention mask that block value randomly.

`p` specify the percentage of value to block. e.g. `A .* RandomMask(0)` is equivalent to `identity(A)` and
 `A .* RandomMask(1)` is equivalent to `zero(A)`.

# Example

```julia-repl
julia> trues(10, 10) .* RandomMask(0.5)
10×10 BitMatrix:
 1  1  1  1  1  1  0  1  1  1
 0  0  1  0  1  0  0  0  1  0
 0  0  1  1  0  0  0  0  1  1
 1  0  1  1  1  0  0  1  0  1
 1  1  0  1  0  0  1  0  1  1
 0  1  1  1  1  0  1  0  1  1
 1  1  0  0  0  0  1  0  0  0
 0  0  1  0  1  1  0  1  1  0
 1  1  1  1  1  1  0  0  1  1
 0  0  1  0  1  1  0  0  1  0

julia> trues(10, 10) .* RandomMask(0.5)
10×10 BitMatrix:
 1  0  1  1  0  0  1  1  0  1
 0  1  0  1  1  1  0  0  1  1
 0  0  1  0  0  0  1  1  0  0
 0  0  0  0  1  0  0  1  1  1
 0  1  1  1  1  0  1  0  0  1
 1  0  0  1  1  0  0  0  1  1
 1  1  1  0  1  1  1  0  0  0
 0  0  1  1  0  0  1  1  1  0
 0  1  1  1  1  0  1  0  1  0
 0  0  1  0  0  0  0  1  1  1
```
"""
RandomMask

"""
    BandPartMask(l::Int, u::Int) <: AbstractAttenMask{DATALESS}

Attention mask that only allow [band_part](https://www.tensorflow.org/api_docs/python/tf/linalg/band_part)
 values to pass.

# Example

```julia-repl
julia> trues(10, 10) .* BandPartMask(3, 5)
10×10 BitMatrix:
 1  1  1  1  1  1  0  0  0  0
 1  1  1  1  1  1  1  0  0  0
 1  1  1  1  1  1  1  1  0  0
 1  1  1  1  1  1  1  1  1  0
 0  1  1  1  1  1  1  1  1  1
 0  0  1  1  1  1  1  1  1  1
 0  0  0  1  1  1  1  1  1  1
 0  0  0  0  1  1  1  1  1  1
 0  0  0  0  0  1  1  1  1  1
 0  0  0  0  0  0  1  1  1  1
```
"""
BandPartMask

"""
    GenericAttenMask <: AbstractAttenMask{ARRAYDATA}

Generic attention mask. Just a wrapper over `AbstractArray{Bool}` for dispatch.

# Example

```julia-repl
julia> bitmask = rand(Bool, 10, 10)
10×10 Matrix{Bool}:
 1  0  1  1  0  0  1  0  1  1
 0  0  1  1  0  0  0  1  1  1
 0  1  0  1  0  1  0  0  1  0
 0  1  1  0  1  1  0  0  0  1
 1  0  1  1  1  0  0  0  0  1
 1  0  1  0  1  1  1  1  0  1
 0  0  0  1  1  1  0  1  1  1
 1  0  1  0  1  1  1  0  0  1
 0  1  0  1  0  0  1  1  0  1
 0  0  0  1  0  1  0  0  0  1

julia> trues(10, 10) .* GenericAttenMask(bitmask)
10×10 BitMatrix:
 1  0  1  1  0  0  1  0  1  1
 0  0  1  1  0  0  0  1  1  1
 0  1  0  1  0  1  0  0  1  0
 0  1  1  0  1  1  0  0  0  1
 1  0  1  1  1  0  0  0  0  1
 1  0  1  0  1  1  1  1  0  1
 0  0  0  1  1  1  0  1  1  1
 1  0  1  0  1  1  1  0  0  1
 0  1  0  1  0  0  1  1  0  1
 0  0  0  1  0  1  0  0  0  1
```
"""
GenericAttenMask

"""
    SymLengthMask(len::AbstractArray{Int, N}) <: AbstractAttenMask{ARRAYDATA}

Attention mask specified by an array of integer that indicate the length dimension size.
 assuming *Query* length and *Key* length are the same.

# Example

```julia-repl
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

See also: [`LengthMask`](@ref), [`BiLengthMask`](@ref), [`BatchedMask`](@ref), [`RepeatMask`](@ref)
"""
SymLengthMask

"""
    BiLengthMask(q_len::A, k_len::A) where {A <: AbstractArray{Int, N}} <: AbstractAttenMask{ARRAYDATA}

Attention mask specified by two arrays of integer that indicate the length dimension size.

# Example

```julia-repl
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

See also: [`SymLengthMask`](@ref), [`BiSeqMask`](@ref), [`BatchedMask`](@ref), [`RepeatMask`](@ref)
"""
BiLengthMask

"""
    LengthMask(len::AbstractArray{Int, N}) <: AbstractSeqMask{ARRAYDATA}

A Sequence Mask specified by an array of integer that indicate the length dimension size.
 Can be convert to attention mask ([`SymLengthMask`](@ref), [`BiLengthMask`](@ref)) with [`AttenMask`](@ref).

# Example

```julia-repl
julia> ones(7, 7, 2) .* LengthMask([3, 5])
7×7×2 Array{Float64, 3}:
[:, :, 1] =
 1.0  1.0  1.0  0.0  0.0  0.0  0.0
 1.0  1.0  1.0  0.0  0.0  0.0  0.0
 1.0  1.0  1.0  0.0  0.0  0.0  0.0
 1.0  1.0  1.0  0.0  0.0  0.0  0.0
 1.0  1.0  1.0  0.0  0.0  0.0  0.0
 1.0  1.0  1.0  0.0  0.0  0.0  0.0
 1.0  1.0  1.0  0.0  0.0  0.0  0.0

[:, :, 2] =
 1.0  1.0  1.0  1.0  1.0  0.0  0.0
 1.0  1.0  1.0  1.0  1.0  0.0  0.0
 1.0  1.0  1.0  1.0  1.0  0.0  0.0
 1.0  1.0  1.0  1.0  1.0  0.0  0.0
 1.0  1.0  1.0  1.0  1.0  0.0  0.0
 1.0  1.0  1.0  1.0  1.0  0.0  0.0
 1.0  1.0  1.0  1.0  1.0  0.0  0.0

```
"""
LengthMask

"""
    RevSymLengthMask(len::AbstractArray{Int, N}) <: AbstractAttenMask{ARRAYDATA}

[`SymLengthMask`](@ref) but counts from the end of array, used for left padding.

# Example

```julia-repl
julia> m = RevSymLengthMask([2,3])
RevSymLengthMask{1, Vector{Int32}}(Int32[2, 3])

julia> trues(3,3, 2) .* m
3×3×2 BitArray{3}:
[:, :, 1] =
 0  0  0
 0  1  1
 0  1  1

[:, :, 2] =
 1  1  1
 1  1  1
 1  1  1

```

See also: [`BiLengthMask`](@ref), [`BatchedMask`](@ref), [`RepeatMask`](@ref)
"""
RevSymLengthMask

"""
    RevBiLengthMask(q_len::A, k_len::A) where {A <: AbstractArray{Int, N}} <: AbstractAttenMask{ARRAYDATA}

[`BiLengthMask`](@ref) but counts from the end of array, used for left padding.

# Example

```julia-repl
julia> bm = RevBiLengthMask([2,3], [3, 5])
RevBiLengthMask{1, Vector{Int32}}(Int32[2, 3], Int32[3, 5])

julia> trues(5,5, 2) .* bm
5×5×2 BitArray{3}:
[:, :, 1] =
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  1  1
 0  0  0  1  1
 0  0  0  1  1

[:, :, 2] =
 0  0  1  1  1
 0  0  1  1  1
 0  0  1  1  1
 0  0  1  1  1
 0  0  1  1  1

```

See also: [`RevLengthMask`](@ref), [`RevSymLengthMask`](@ref), [`BiSeqMask`](@ref), [`BatchedMask`](@ref), [`RepeatMask`](@ref)
"""
RevBiLengthMask

"""
    RevLengthMask(len::AbstractArray{Int, N}) <: AbstractSeqMask{ARRAYDATA}

[`LengthMask`](@ref) but counts from the end of array, used for left padding.
 Can be convert to attention mask ([`RevSymLengthMask`](@ref), [`RevBiLengthMask`](@ref)) with [`AttenMask`](@ref).

# Example

```julia-repl
julia> ones(7, 7, 2) .* RevLengthMask([3, 5])
7×7×2 Array{Float64, 3}:
[:, :, 1] =
 0.0  0.0  0.0  0.0  1.0  1.0  1.0
 0.0  0.0  0.0  0.0  1.0  1.0  1.0
 0.0  0.0  0.0  0.0  1.0  1.0  1.0
 0.0  0.0  0.0  0.0  1.0  1.0  1.0
 0.0  0.0  0.0  0.0  1.0  1.0  1.0
 0.0  0.0  0.0  0.0  1.0  1.0  1.0
 0.0  0.0  0.0  0.0  1.0  1.0  1.0

[:, :, 2] =
 0.0  0.0  1.0  1.0  1.0  1.0  1.0
 0.0  0.0  1.0  1.0  1.0  1.0  1.0
 0.0  0.0  1.0  1.0  1.0  1.0  1.0
 0.0  0.0  1.0  1.0  1.0  1.0  1.0
 0.0  0.0  1.0  1.0  1.0  1.0  1.0
 0.0  0.0  1.0  1.0  1.0  1.0  1.0
 0.0  0.0  1.0  1.0  1.0  1.0  1.0

```
"""
RevLengthMask

"""
    GenericSeqMask(mask::AbstractArray{Bool}) <: AbstractSeqMask{ARRAYDATA}

Create a sequence mask from an array of `Bool`.

# Example

```julia-repl
julia> m = GenericSeqMask(rand(Bool, 10, 2))
GenericSeqMask{3, Array{Bool, 3}}([0 1 … 0 0;;; 1 0 … 1 0])

julia> trues(7, 10, 2) .* m
7×10×2 BitArray{3}:
[:, :, 1] =
 0  1  0  0  1  0  0  0  0  0
 0  1  0  0  1  0  0  0  0  0
 0  1  0  0  1  0  0  0  0  0
 0  1  0  0  1  0  0  0  0  0
 0  1  0  0  1  0  0  0  0  0
 0  1  0  0  1  0  0  0  0  0
 0  1  0  0  1  0  0  0  0  0

[:, :, 2] =
 1  0  1  1  0  1  1  1  1  0
 1  0  1  1  0  1  1  1  1  0
 1  0  1  1  0  1  1  1  1  0
 1  0  1  1  0  1  1  1  1  0
 1  0  1  1  0  1  1  1  1  0
 1  0  1  1  0  1  1  1  1  0
 1  0  1  1  0  1  1  1  1  0

julia> m.mask
1×10×2 Array{Bool, 3}:
[:, :, 1] =
 0  1  0  0  1  0  0  0  0  0

[:, :, 2] =
 1  0  1  1  0  1  1  1  1  0

```
"""
GenericSeqMask

"""
    BiSeqMask(qmask::A1, kmask::A2) where {A1 <: AbstractSeqMask, A2 <: AbstractSeqMask} <: AbstractAttenMask

Take two sequence mask and construct an attention mask.

# Example

```julia-repl
julia> trues(7, 7, 2) .* Masks.BiSeqMask(Masks.LengthMask([3, 5]), Masks.RevLengthMask([3, 5]))
7×7×2 BitArray{3}:
[:, :, 1] =
 0  0  0  0  0  0  0
 0  0  0  0  0  0  0
 0  0  0  0  0  0  0
 0  0  0  0  0  0  0
 1  1  1  0  0  0  0
 1  1  1  0  0  0  0
 1  1  1  0  0  0  0

[:, :, 2] =
 0  0  0  0  0  0  0
 0  0  0  0  0  0  0
 1  1  1  1  1  0  0
 1  1  1  1  1  0  0
 1  1  1  1  1  0  0
 1  1  1  1  1  0  0
 1  1  1  1  1  0  0
```

See also: [`BiLengthMask`](@ref), [`RevBiLengthMask`](@ref)
"""
BiSeqMask

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

```julia-repl
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

```julia-repl
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

```julia-repl
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

"""
    lengths(::AbstractSeqMask)

Get the number of `true`s of each batch in the sequence mask.
"""
lengths

end
