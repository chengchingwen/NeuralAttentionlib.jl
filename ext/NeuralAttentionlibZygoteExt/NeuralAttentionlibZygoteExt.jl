module NeuralAttentionlibZygoteExt

using NeuralAttentionlib
using Zygote

Zygote.unbroadcast(x::NeuralAttentionlib.AbstractMask, _) = nothing

end
