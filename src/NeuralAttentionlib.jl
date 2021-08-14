module NeuralAttentionlib

using Flux
using Zygote
using Functors
using LazyArrays
using PartialFunctions
using NNlib
using CUDA

import Base

export attention, generic_attention


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


include("./utils.jl")

# matrix multiplication
include("./matmul/collapseddim.jl")
include("./matmul/gemm.jl")
include("./matmul/matmul.jl")
include("./matmul/grad.jl")

# attention score masking
include("./mask/indexer.jl")
include("./mask/axes.jl")
include("./mask/mask.jl")
include("./mask/dataless.jl")
include("./mask/array.jl")
include("./mask/wrapper.jl")
include("./mask/grad.jl")

# attention architecture
include("./functional/utils.jl")
include("./functional/score.jl")
include("./functional/mixing.jl")
include("./functional/attention.jl")

include("./types.jl")

include("./module/utils.jl")
include("./module/mask.jl")
include("./module/matmul.jl")

using .Masks
using .Matmul

end # module
