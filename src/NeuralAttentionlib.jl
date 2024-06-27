module NeuralAttentionlib

using Static

using Adapt
import Adapt: adapt_structure, adapt
using ChainRulesCore

using NNlib

export multihead_qkv_attention, Functional, Masks

include("./utils.jl")

# matrix multiplication
include("./matmul/collapseddims.jl")
include("./matmul/gemm.jl")
include("./matmul/matmul.jl")
include("./matmul/grad.jl")
include("./matmul/scaled_matmul.jl")

# attention score masking
include("./mask/mask.jl")
include("./mask/indexer.jl")
include("./mask/constraint.jl")
include("./mask/broadcast.jl")
include("./mask/sequence.jl")
include("./mask/dataless.jl")
include("./mask/array.jl")
include("./mask/wrapper.jl")
include("./mask/grad.jl")

# attention architecture
include("./functional/utils.jl")
include("./functional/score.jl")
include("./functional/mixing.jl")
include("./functional/attention.jl")
include("./functional/grad.jl")

# position embedding
include("./functional/position_embedding/relative.jl")
include("./functional/position_embedding/sincos.jl")
include("./functional/position_embedding/rotary.jl")
include("./functional/position_embedding/alibi.jl")

# extra helper functions
include("./functional/layernorm.jl")
include("./functional/l2norm.jl")

include("./types.jl")

# docs
include("./module/utils.jl")
include("./module/mask.jl")
include("./module/matmul.jl")
include("./module/functional.jl")

include("./functional/optimized.jl")

using .Masks
using .Matmul
using .Functional

end # module
