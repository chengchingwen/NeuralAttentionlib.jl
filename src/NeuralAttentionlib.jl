module NeuralAttentionlib

using PartialFunctions
using NNlib

using Requires
using Adapt
import Adapt: adapt_structure, adapt

import Base

export attention, generic_attention

include("./utils.jl")

# matrix multiplication
include("./matmul/collapseddim.jl")
include("./matmul/gemm.jl")
include("./matmul/matmul.jl")
include("./matmul/grad.jl")
include("./matmul/gpu.jl")

# attention score masking
include("./mask/indexer.jl")
include("./mask/mask.jl")
include("./mask/constraint.jl")
include("./mask/broadcast.jl")
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
include("./module/functional.jl")

using .Masks
using .Matmul
using .Functional

end # module
