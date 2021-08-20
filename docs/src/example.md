# Example

## Comparing to the existing implementation in Transformers.jl

See the code in the [NeuralAttentionlib's test](https://github.com/chengchingwen/NeuralAttentionlib.jl/blob/master/test/mha.jl),
 where we compare output/gradient from `NeuralAttenionlib` v.s. the `MultiheadAttention` layer from [`Transformers.jl`](https://github.com/chengchingwen/Transformers.jl).
 This should provide enough knowledge for implementing a multi-head QKV attention layer with DL framework like Flux.jl.
