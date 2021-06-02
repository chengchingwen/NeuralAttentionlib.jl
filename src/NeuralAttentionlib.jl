module NeuralAttentionlib

using Flux
using Zygote
using Functors
using LazyArrays
using CUDA

import Base

# fix julia #41054
Base.reducedim_init(f, op::Union{typeof(+), typeof(Base.add_sum)}, A::Base.AbstractBroadcasted, region) = Base._reducedim_init(f, op, zero, sum, A, region)

# finx julia #41055
Base.has_fast_linear_indexing(::Base.RefValue{typeof(^)}) = true
Base.has_fast_linear_indexing(::Base.RefValue{Val{2}}) = true

Base.LinearIndices(bc::Base.AbstractBroadcasted) = LinearIndices(map(Base.OneTo, size(bc)))

include("./simple_attention.jl")

end # module
