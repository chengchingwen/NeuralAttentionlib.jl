module NeuralAttentionlib

using Flux
using Zygote
using Functors
using LazyArrays
using PartialFunctions
using NNlib
using CUDA

import Base

export matmul, CollapsedDimArray, collapseddim, unwrap_collapse,
    attention, generic_attention


# fix julia #41054
Base.reducedim_init(f, op::Union{typeof(+), typeof(Base.add_sum)}, A::Base.AbstractBroadcasted, region) = Base._reducedim_init(f, op, zero, sum, A, region)

# finx julia #41055
Base.has_fast_linear_indexing(::Base.Ref) = true
# Base.has_fast_linear_indexing(::Base.RefValue{typeof(^)}) = true
# Base.has_fast_linear_indexing(::Base.RefValue{Val{2}}) = true

Base.LinearIndices(bc::Base.AbstractBroadcasted) = LinearIndices(map(Base.OneTo, size(bc)))


# function attention(x)
#   # x_ = x_transform(x)
#   # q = input_transform_q(x_)
#   # k = input_transform_k(x_)
#   # v = input_transform_v(x_)
#   s = dropout(attention_score(q, k, v))
#   s_ = normalize(s)
#   y = weightsum(v, s)
#   # y_ = output_transform(y)
#   # z = dropout(y_transform(y_))
#   # out = residual(x, z)
#   return out
# end



# include("./mapreduce_utils.jl")
# include("./attention.jl")
# include("./simple_attention.jl")
include("./utils.jl")
include("./collapseddim.jl")
include("./gemm.jl")
include("./matmul.jl")
include("./grad.jl")
include("./mask.jl")

include("./functional/utils.jl")
include("./functional/score.jl")
include("./functional/mixing.jl")
include("./functional/attention.jl")

include("./types.jl")


end # module
