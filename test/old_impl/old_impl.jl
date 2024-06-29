module Old_Impl

import NeuralAttentionlib

export MultiheadAttention

const Abstract3DTensor{T} = AbstractArray{T, 3}

include("batched_tril.jl")
include("batchedmul.jl")

# an multi-head attention used in Transformers.jl v0.1.13
include("./mh_atten.jl")
include("./getmask.jl")

end
