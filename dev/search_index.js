var documenterSearchIndex = {"docs":
[{"location":"term/#Terminology","page":"Terminology","title":"Terminology","text":"","category":"section"},{"location":"term/","page":"Terminology","title":"Terminology","text":"Term and Naming explanation.","category":"page"},{"location":"term/#Prerequisite","page":"Terminology","title":"Prerequisite","text":"","category":"section"},{"location":"term/","page":"Terminology","title":"Terminology","text":"Some term for better understanding this docs.","category":"page"},{"location":"term/#.-[PartialFunctions](https://github.com/archermarx/PartialFunctions.jl)","page":"Terminology","title":"1. PartialFunctions","text":"","category":"section"},{"location":"term/","page":"Terminology","title":"Terminology","text":"This actually live outside the scope of this package, but is extremely useful for illustrate the overall design.  We'll use the $ operation to denote partial function application   (i.e. f $ x is equivanlent to (arg...)->f(x, arg...)).","category":"page"},{"location":"term/#.-Feature-/-Length-/-Batch-Dimension","page":"Terminology","title":"2. Feature / Length / Batch Dimension","text":"","category":"section"},{"location":"term/","page":"Terminology","title":"Terminology","text":"Under the context of attention operation in deep learning, the input data can be viewed as a 3-dimensional array.  The feature dimension, the length dimension, and the batch dimension (f-dim, l-dim, b-dim for short).  Following the Julia's multidimensional array implementation  (column-major),  the data is store in a AbstractArray{T, 3} whose size is (f-dim, l-dim, b-dim).","category":"page"},{"location":"term/","page":"Terminology","title":"Terminology","text":"For example, given 3 sentence as a batch, each sentence have 10 word, and we choose to represent a word with  a vector of 32 element. This data will be store in an 3-dim array with size (32, 10, 3).","category":"page"},{"location":"term/","page":"Terminology","title":"Terminology","text":"General speaking, batch stands for how many independent data you are going to run in one function call,  usually just for performance/optimization need. length means how many entry you have for each data sample,  like the #-words in a sentence or #-pixels in an image. feature is the number of value you used to  represent an entry.","category":"page"},{"location":"term/#Attention","page":"Terminology","title":"Attention","text":"","category":"section"},{"location":"term/","page":"Terminology","title":"Terminology","text":"The overall attention operation can be viewed as three mutually inclusive block:","category":"page"},{"location":"term/","page":"Terminology","title":"Terminology","text":"\t     (main input)\n\t        Value           Key             Query  (Extras...)\n\t+---------|--------------|----------------|------|||---- Attention Operation ---+\n\t|         |              |                |      |||                            |\n\t|         |              |                |      |||   multihead, ...           |\n\t|         |              |                |      |||                            |\n\t|   +-----|--------------|----------------|------|||-----------------------+    |\n\t|   |     |              |                |      |||                       |    |\n\t|   |     |          +---|----------------|------|||-------------+         |    |\n\t|   |     |          |   |                |      |||             |         |    |\n\t|   |     |          |   |  scoring func  |      |||             |         |    |\n\t|   |     |          |   +------>+<-------+<=======+             |         |    |\n\t|   |     |          |           |                               |         |    |\n\t|   |     |          |           | masked_score,                 |         |    |\n\t|   |     |          |           | normalized_score,             |         |    |\n\t|   |     |          |           | ...                           |         |    |\n\t|   |     |          |           |                               |         |    |\n\t|   |     |          +-----------|------------ Attention Score --+         |    |\n\t|   |     |                      |                                         |    |\n\t|   |     |     mixing func      |                                         |    |\n\t|   |     +--------->+<----------+                                         |    |\n\t|   |                |                                                     |    |\n\t|   +----------------|------------------------------- Mixing --------------+    |\n\t|                    |                                                          |\n\t+--------------------|----------------------------------------------------------+\n\t              Attentive Value\n\t               (main output)","category":"page"},{"location":"term/","page":"Terminology","title":"Terminology","text":"The attention operation is actually a special way to \"mix\" (or \"pick\" in common lecture) the input information.  In (probably) the first attention paper, the attention is defined as weighted  sum of the input sequence given a word embedding. The idea is furthur generalize to QKV attention in the first  transformer paper. ","category":"page"},{"location":"term/#.-Attention-Score","page":"Terminology","title":"1. Attention Score","text":"","category":"section"},{"location":"term/","page":"Terminology","title":"Terminology","text":"The attention score is used to decide how much the each piece of input information will contribute to the  output value and also how many entry the attention operation will output. The operation that will modify  the attention score matrix should be consider as part of this block. For example: Different attention masks  (local attention, random attention, ...), normalization (softmax, l2-norm, ...), and some special attention  that take other inputs (transformer decoder, relative position encoding, ...).","category":"page"},{"location":"term/#.-Mixing","page":"Terminology","title":"2. Mixing","text":"","category":"section"},{"location":"term/","page":"Terminology","title":"Terminology","text":"We refer to the operation that take the attention score and input value as \"mixing\". Usually it's just a  weighted sum over the input value and use the attention score as the weight.","category":"page"},{"location":"term/#.-Attention-Operation","page":"Terminology","title":"3. Attention Operation","text":"","category":"section"},{"location":"term/","page":"Terminology","title":"Terminology","text":"The whole scoring + mixing and other pre/post processing made up an attention operation. Things like handling  multi-head should happen at this level.","category":"page"},{"location":"term/#Attention-Mask","page":"Terminology","title":"Attention Mask","text":"","category":"section"},{"location":"term/","page":"Terminology","title":"Terminology","text":"Attention masks are a bunch of operation that modified the attention score.","category":"page"},{"location":"term/#.-Dataless-mask","page":"Terminology","title":"1. Dataless mask","text":"","category":"section"},{"location":"term/","page":"Terminology","title":"Terminology","text":"We use \"dataless\" to refer to masks that are independent to the input. For example, CausalMask works the same  on each data regardless of the batch size or the data content.","category":"page"},{"location":"term/#.-Array-mask","page":"Terminology","title":"2. Array mask","text":"","category":"section"},{"location":"term/","page":"Terminology","title":"Terminology","text":"We call the mask that is dependent to the input as \"array mask\". For example, SymLengthMask is used to avoid  the padding token being considered in the attention operation, thus each data batch might have different mask value.","category":"page"},{"location":"api/#API-Reference","page":"API Reference","title":"API Reference","text":"","category":"section"},{"location":"api/","page":"API Reference","title":"API Reference","text":"Order   = [:function, :type]","category":"page"},{"location":"api/#Functional","page":"API Reference","title":"Functional","text":"","category":"section"},{"location":"api/","page":"API Reference","title":"API Reference","text":"Modules = [NeuralAttentionlib, NeuralAttentionlib.Functional, NeuralAttentionlib.Masks]\nPages   = [\"functional.jl\"]","category":"page"},{"location":"api/#NeuralAttentionlib.attention_score","page":"API Reference","title":"NeuralAttentionlib.attention_score","text":"attention_score(f, args...) = f(args...)\n\nAttention score api. Can be overload for doing custom implementation with generic_qkv_attention.  f is the score function.\n\nSee also: generic_qkv_attention, generic_multihead_qkv_attention, mixing\n\n\n\n\n\n","category":"function"},{"location":"api/#NeuralAttentionlib.biased_score","page":"API Reference","title":"NeuralAttentionlib.biased_score","text":"biased_score(bias, score, args...)\n\nAdding a precomputed bias to the attention score. bias should be in shape (key length, query length, ...) and  size(bias, 1) == size(s, 1) == size(bias, 2) == size(s, 2) && ndims(bias) <= ndims(s) where s = score(args...)  must hold.\n\n\n\n\n\n","category":"function"},{"location":"api/#NeuralAttentionlib.dot_product_score","page":"API Reference","title":"NeuralAttentionlib.dot_product_score","text":"dot_product_score(q, k)\n\nDot-product attention score function. Equivalent to scaled_dot_product_score(q, k, 1).\n\nSee also: scaled_dot_product_score\n\n\n\n\n\n","category":"function"},{"location":"api/#NeuralAttentionlib.generic_multihead_qkv_attention","page":"API Reference","title":"NeuralAttentionlib.generic_multihead_qkv_attention","text":"generic_multihead_qkv_attention(mixingf, scoref, head, q, k, v, args...)\n\nGeneric version of multihead_qkv_attention. Need to specify mixing and score function.\n\n\n\n\n\n","category":"function"},{"location":"api/#NeuralAttentionlib.generic_qkv_attention","page":"API Reference","title":"NeuralAttentionlib.generic_qkv_attention","text":"generic_qkv_attention(mixingf, scoref, q, k, v, args...)\n\nGeneric version of naive_qkv_attention. Need to specify mixing and score function.\n\n\n\n\n\n","category":"function"},{"location":"api/#NeuralAttentionlib.get_sincos_position_embeddings","page":"API Reference","title":"NeuralAttentionlib.get_sincos_position_embeddings","text":"get_sincos_position_embeddings(hidden_size::Integer, normalized::Bool, x)\n\nsincos position embeddings. x can be either a integer specifying the length or an array of position indices.\n\n\n\n\n\n","category":"function"},{"location":"api/#NeuralAttentionlib.layer_norm","page":"API Reference","title":"NeuralAttentionlib.layer_norm","text":"layer_norm([epsilon = 1e-5,] alpha, beta, x)\n\nFunction which perform layer normalization on x. alpha and beta can a Vector, Number or Nothing.\n\nlayer_norm(α β x) = αfrac(x - μ)σ + β\n\nIf both alpha and beta is Nothing, this is just a standardize function applied on the first dimension.\n\n\n\n\n\n","category":"function"},{"location":"api/#NeuralAttentionlib.masked_score","page":"API Reference","title":"NeuralAttentionlib.masked_score","text":"masked_score(mask) = masked_score $ mask\nmasked_score(maskop, mask) = masked_score $ maskop $ mask\nmasked_score(maskop::AbstractMaskOp, mask::AbstractMask, score, args...)\n\nMasked attention score api. Applying the mask according to maskop on the attention score  compute from score(args...).\n\nSee also: naive_qkv_attention, SymLengthMask, BiLengthMask\n\n\n\n\n\n","category":"function"},{"location":"api/#NeuralAttentionlib.merge_head","page":"API Reference","title":"NeuralAttentionlib.merge_head","text":"merge_head(x)\n\nmerge the head dimension split by split_head.\n\n\n\n\n\n","category":"function"},{"location":"api/#NeuralAttentionlib.mixing","page":"API Reference","title":"NeuralAttentionlib.mixing","text":"mixing(f, v, g, args...) = f(attention_score(g, args...), v)\n\nMixing function api. Can be overload for doing custom implementation with generic_qkv_attention.  f is the mixing function and g is score function.\n\nSee also: generic_qkv_attention, generic_multihead_qkv_attention, attention_score\n\n\n\n\n\n","category":"function"},{"location":"api/#NeuralAttentionlib.move_head_dim_in","page":"API Reference","title":"NeuralAttentionlib.move_head_dim_in","text":"move_head_dim_in(x::AbstractArray, nobatch=false)\n\nEquivanlent to permutedims(x, move_head_dim_in_perm(x, nobatch)))\n\nSee also: merge_head, move_head_dim_in_perm\n\n\n\n\n\n","category":"function"},{"location":"api/#NeuralAttentionlib.move_head_dim_in_perm","page":"API Reference","title":"NeuralAttentionlib.move_head_dim_in_perm","text":"move_head_dim_in_perm(x::AbstractArray{T, N}, nobatch=false)\nmove_head_dim_in_perm(N::Int, nobatch=false)\n\nDimension order for permutedims to move the head dimension (created by split_head) from batch dimension  to feature dimension (for merge_head). Return a tuple of integer of length n.  nobatch specify where x is a batch of data.\n\nExample\n\njulia> Functional.move_head_dim_in_perm(5, false)\n(1, 4, 2, 3, 5)\n\njulia> Functional.move_head_dim_in_perm(5, true)\n(1, 5, 2, 3, 4)\n\n\nSee also: merge_head, move_head_dim_in\n\n\n\n\n\n","category":"function"},{"location":"api/#NeuralAttentionlib.move_head_dim_out","page":"API Reference","title":"NeuralAttentionlib.move_head_dim_out","text":"move_head_dim_out(x::AbstractArray, nobatch=false)\n\nEquivanlent to permutedims(x, move_head_dim_out_perm(x, nobatch)))\n\nSee also: split_head, move_head_dim_out_perm\n\n\n\n\n\n","category":"function"},{"location":"api/#NeuralAttentionlib.move_head_dim_out_perm","page":"API Reference","title":"NeuralAttentionlib.move_head_dim_out_perm","text":"move_head_dim_out_perm(x::AbstractArray{T, N}, nobatch=false)\nmove_head_dim_out_perm(N::Int, nobatch=false)\n\nDimension order for permutedims to move the head dimension (created by split_head) to batch dimension.  Return a tuple of integer of length n. nobatch specify where x is a batch of data.\n\nExample\n\njulia> Functional.move_head_dim_out_perm(5, false)\n(1, 3, 4, 2, 5)\n\njulia> Functional.move_head_dim_out_perm(5, true)\n(1, 3, 4, 5, 2)\n\n\nSee also: split_head, move_head_dim_out\n\n\n\n\n\n","category":"function"},{"location":"api/#NeuralAttentionlib.multihead_qkv_attention","page":"API Reference","title":"NeuralAttentionlib.multihead_qkv_attention","text":"multihead_qkv_attention(head, q, k, v, mask=nothing)\n\nMultihead version of naive_qkv_attention. The core operation for implement a regular transformer layer.\n\n\n\n\n\n","category":"function"},{"location":"api/#NeuralAttentionlib.naive_qkv_attention","page":"API Reference","title":"NeuralAttentionlib.naive_qkv_attention","text":"naive_qkv_attention(q, k, v, mask=nothing)\n\nThe scaled dot-product attention of a regular transformer layer.\n\nAttention(Q K V) = softmax(fracQK^Tsqrtd_k)V\n\nIt's equivalent to generic_qkv_attention(weighted_sum_mixing, normalized_score(NNlib.softmax) $ masked_score(GenericMaskOp(), mask) $ scaled_dot_product_score, q, k, v).\n\n#Example\n\njulia> fdim, ldim, bdim = 32, 10, 4;\n\njulia> x = randn(fdim, ldim, bdim);\n\njulia> y = naive_qkv_attention(x, x, x); # simple self attention\n\n# no mask here\njulia> z = generic_qkv_attention(weighted_sum_mixing, normalized_score(NNlib.softmax) $ scaled_dot_product_score, x, x, x);\n\njulia> y ≈ z\ntrue\n\n\nSee also: generic_qkv_attention\n\n\n\n\n\n","category":"function"},{"location":"api/#NeuralAttentionlib.normalized_score","page":"API Reference","title":"NeuralAttentionlib.normalized_score","text":"normalized_score(norm) = normalized_score $ norm\nnormalized_score(norm, score, args...)\n\nNormalized attenion score api. norm is the normalize function (like softmax) and score is the function  that compute attention score from args....\n\nSee also: naive_qkv_attention\n\n\n\n\n\n","category":"function"},{"location":"api/#NeuralAttentionlib.rms_layer_norm","page":"API Reference","title":"NeuralAttentionlib.rms_layer_norm","text":"rms_layer_norm([epsilon = 1e-5,] alpha, x)\n\nFunction which perform root-mean-square layer normalization on x. alpha and beta can a Vector, Number  or Nothing.\n\nrms_layer_norm(α x) = αfracxsqrtsum_i=1^N x^2  N\n\nIf both alpha is Nothing, this is just a normalization with root-mean-square function applied on the first  dimension.\n\n\n\n\n\n","category":"function"},{"location":"api/#NeuralAttentionlib.scalar_relative_position_embedding","page":"API Reference","title":"NeuralAttentionlib.scalar_relative_position_embedding","text":"scalar_relative_position_embedding(relative_position_id_func, embedding_table, score, args...)\n\nA relative position embedding that produce a trainable scalar bias for each value in the attention score.  relative_position_id_func is a function that take the attention score and return a relative_position_id  matrix with the same size of the attention score with batches (normally (key length, query length)). This  relative_position_id would be used to index (or gather) the embedding_table. embedding_table is an  array with multiple dimensions, where the first dimension is the number of possible \"id\"s and the remaining  dimensions are for giving different value to each heads. By default we treat the last dimension of attention  score as the batch dimension and the dimension between last dimension and the \"length\" dimension as the head  dimensions.\n\n\n\n\n\n","category":"function"},{"location":"api/#NeuralAttentionlib.scaled_dot_product_score","page":"API Reference","title":"NeuralAttentionlib.scaled_dot_product_score","text":" scaled_dot_product_score(q, k, s = sqrt(inv(size(k, 1))))\n\nThe scaled dot-product attention score function of a regular transformer layer.\n\nScore(Q K) = fracQK^Tsqrtd_k\n\nscaled_dot_product_score(f, q, k)\n\nApply a transform function f on q/k before dot-product.\n\nSee also: naive_qkv_attention\n\n\n\n\n\n","category":"function"},{"location":"api/#NeuralAttentionlib.split_head","page":"API Reference","title":"NeuralAttentionlib.split_head","text":"split_head(head::Int, x)\n\nSplit the first dimension into head piece of small vector. Equivalent to  reshape(x, :, head, tail(size(x))...).\n\n\n\n\n\n","category":"function"},{"location":"api/#NeuralAttentionlib.t5_bucketed_position_id","page":"API Reference","title":"NeuralAttentionlib.t5_bucketed_position_id","text":"t5_bucketed_position_id(n_buckets::Int, max_distance::Int)\n\nA relative_position_id_func used in the T5 Transformer model. The relative distances is assigned to a  logarithmical buecket and the distance beyond max_distance would be assigned to the same bucket.\n\nSee also: scalar_relative_position_embedding, t5_causal_bucketed_position_id\n\n\n\n\n\n","category":"function"},{"location":"api/#NeuralAttentionlib.t5_causal_bucketed_position_id","page":"API Reference","title":"NeuralAttentionlib.t5_causal_bucketed_position_id","text":"t5_causal_bucketed_position_id(n_buckets::Int, max_distance::Int)\n\nSame as t5_bucketed_position_id but only attent to past. Should be used with CausalMask\n\nSee also: scalar_relative_position_embedding, t5_bucketed_position_id\n\n\n\n\n\n","category":"function"},{"location":"api/#NeuralAttentionlib.weighted_sum_mixing","page":"API Reference","title":"NeuralAttentionlib.weighted_sum_mixing","text":"weighted_sum_mixing(s, v)\n\nThe mixing function of a regular transformer layer. s is the attention score and v is the value of QKV attention.\n\n\n\n\n\n","category":"function"},{"location":"api/#NeuralAttentionlib.with_rotary_position_embedding","page":"API Reference","title":"NeuralAttentionlib.with_rotary_position_embedding","text":"with_rotary_position_embedding([size,] x)\n\nApply rotary position embedding to x. Can take an size argument and the rotary position embedding will only apply  to x[1:size, :, ...]. Should be used with scaled_dot_product_score/dot_product_score.\n\n\n\n\n\n","category":"function"},{"location":"api/#Mask","page":"API Reference","title":"Mask","text":"","category":"section"},{"location":"api/","page":"API Reference","title":"API Reference","text":"Modules = [NeuralAttentionlib, NeuralAttentionlib.Masks]\nPages   = [\"mask.jl\"]","category":"page"},{"location":"api/#NeuralAttentionlib.AbstractAttenMask","page":"API Reference","title":"NeuralAttentionlib.AbstractAttenMask","text":"AbstractAttenMask <: AbstractMask\n\nAbstract type for mask data specifically for attention.\n\n\n\n\n\n","category":"type"},{"location":"api/#NeuralAttentionlib.AbstractMask","page":"API Reference","title":"NeuralAttentionlib.AbstractMask","text":"AbstractMask\n\nAbstract type for mask data.\n\n\n\n\n\n","category":"type"},{"location":"api/#NeuralAttentionlib.AbstractMaskOp","page":"API Reference","title":"NeuralAttentionlib.AbstractMaskOp","text":"AbstractMaskOp\n\nTrait-like abstract type for holding operation related argument, defined how the mask should be apply to input array\n\n\n\n\n\n","category":"type"},{"location":"api/#NeuralAttentionlib.AbstractSequenceMask","page":"API Reference","title":"NeuralAttentionlib.AbstractSequenceMask","text":"AbstractSequenceMask <: AbstractMask\n\nAbstract type for mask data specifically for sequence.\n\n\n\n\n\n","category":"type"},{"location":"api/#NeuralAttentionlib.apply_mask-Tuple{NeuralAttentionlib.GenericMaskOp, NeuralAttentionlib.AbstractMask, Any}","page":"API Reference","title":"NeuralAttentionlib.apply_mask","text":"apply_mask(op::GenericMaskOp, mask::AbstractMask, score)\n\nEquivalent to op.apply(score, op.scale .* (op.flip ? .! mask : mask)).\n\nExample\n\njulia> x = randn(10, 10);\n\njulia> m = CausalMask()\nCausalMask()\n\njulia> apply_mask(GenericMaskOp(.+, true, -1e9), m, x) ==  @. x + (!m * -1e9)\ntrue\n\n\n\n\n\n\n","category":"method"},{"location":"api/#NeuralAttentionlib.apply_mask-Tuple{NeuralAttentionlib.NaiveMaskOp, NeuralAttentionlib.AbstractMask, Any}","page":"API Reference","title":"NeuralAttentionlib.apply_mask","text":"apply_mask(op::NaiveMaskOp, mask::AbstractMask, score)\n\nDirectly broadcast multiply mask to attention score, i.e. score .* mask.\n\n\n\n\n\n","category":"method"},{"location":"api/#NeuralAttentionlib.AbstractArrayMask","page":"API Reference","title":"NeuralAttentionlib.AbstractArrayMask","text":"AbstractArrayMask <: AbstractAttenMask\n\nAbstract type for mask with array data\n\n\n\n\n\n","category":"type"},{"location":"api/#NeuralAttentionlib.AbstractDatalessMask","page":"API Reference","title":"NeuralAttentionlib.AbstractDatalessMask","text":"AbstractDatalessMask <: AbstractAttenMask\n\nAbstract type for mask without array data.\n\n\n\n\n\n","category":"type"},{"location":"api/#NeuralAttentionlib.BandPartMask","page":"API Reference","title":"NeuralAttentionlib.BandPartMask","text":"BandPartMask(l::Int, u::Int) <: AbstractDatalessMask\n\nAttention mask that only allow band_part  values to pass.\n\n\n\n\n\n","category":"type"},{"location":"api/#NeuralAttentionlib.BatchedMask","page":"API Reference","title":"NeuralAttentionlib.BatchedMask","text":"BatchedMask(mask::AbstractMask) <: AbstractWrapperMask\n\nAttention mask wrapper over array mask for applying the same mask within the same batch.\n\nExample\n\njulia> m = SymLengthMask([2,3])\nSymLengthMask{1, Vector{Int32}}(Int32[2, 3])\n\njulia> trues(3,3, 2) .* m\n3×3×2 BitArray{3}:\n[:, :, 1] =\n 1  1  0\n 1  1  0\n 0  0  0\n\n[:, :, 2] =\n 1  1  1\n 1  1  1\n 1  1  1\n\njulia> trues(3,3, 2, 2) .* m\nERROR: DimensionMismatch(\"arrays could not be broadcast to a common size; mask require ndims(A) == 3\")\nStacktrace:\n[...]\n\njulia> trues(3,3, 2, 2) .* BatchedMask(m) # 4-th dim become batch dim\n3×3×2×2 BitArray{4}:\n[:, :, 1, 1] =\n 1  1  0\n 1  1  0\n 0  0  0\n\n[:, :, 2, 1] =\n 1  1  0\n 1  1  0\n 0  0  0\n\n[:, :, 1, 2] =\n 1  1  1\n 1  1  1\n 1  1  1\n\n[:, :, 2, 2] =\n 1  1  1\n 1  1  1\n 1  1  1\n\n\n\n\n\n\n","category":"type"},{"location":"api/#NeuralAttentionlib.BiLengthMask","page":"API Reference","title":"NeuralAttentionlib.BiLengthMask","text":"BiLengthMask(q_len::A, k_len::A) where {A <: AbstractArray{Int, N}} <: AbstractArrayMask\n\nAttention mask specified by two arrays of integer that indicate the length dimension size.\n\nExample\n\njulia> bm = BiLengthMask([2,3], [3, 5])\nBiLengthMask{1, Vector{Int32}}(Int32[2, 3], Int32[3, 5])\n\njulia> trues(5,5, 2) .* bm\n5×5×2 BitArray{3}:\n[:, :, 1] =\n 1  1  0  0  0\n 1  1  0  0  0\n 1  1  0  0  0\n 0  0  0  0  0\n 0  0  0  0  0\n\n[:, :, 2] =\n 1  1  1  0  0\n 1  1  1  0  0\n 1  1  1  0  0\n 1  1  1  0  0\n 1  1  1  0  0\n\n\nSee also: SymLengthMask, BatchedMask, RepeatMask\n\n\n\n\n\n","category":"type"},{"location":"api/#NeuralAttentionlib.CausalMask","page":"API Reference","title":"NeuralAttentionlib.CausalMask","text":"CausalMask() <: AbstractDatalessMask\n\nAttention mask that block the future values.\n\nSimilar to applying LinearAlgebra.triu! on the score matrix\n\n\n\n\n\n","category":"type"},{"location":"api/#NeuralAttentionlib.GenericAttenMask","page":"API Reference","title":"NeuralAttentionlib.GenericAttenMask","text":"GenericAttenMask <: AbstractArrayMask\n\nGeneric attention mask. Just a wrapper over AbstractArray{Bool} for dispatch.\n\n\n\n\n\n","category":"type"},{"location":"api/#NeuralAttentionlib.LengthMask","page":"API Reference","title":"NeuralAttentionlib.LengthMask","text":"LengthMask(len::AbstractArray{Int, N}) <: AbstractSequenceMask\n\nA Sequence Mask specified by an array of integer that indicate the length dimension size.  Can be convert to attention mask (SymLengthMask, BiLengthMask) with AttenMask.\n\nExample\n\njulia> ones(7, 7, 2) .* LengthMask([3, 5])\n7×7×2 Array{Float64, 3}:\n[:, :, 1] =\n 1.0  1.0  1.0  0.0  0.0  0.0  0.0\n 1.0  1.0  1.0  0.0  0.0  0.0  0.0\n 1.0  1.0  1.0  0.0  0.0  0.0  0.0\n 1.0  1.0  1.0  0.0  0.0  0.0  0.0\n 1.0  1.0  1.0  0.0  0.0  0.0  0.0\n 1.0  1.0  1.0  0.0  0.0  0.0  0.0\n 1.0  1.0  1.0  0.0  0.0  0.0  0.0\n\n[:, :, 2] =\n 1.0  1.0  1.0  1.0  1.0  0.0  0.0\n 1.0  1.0  1.0  1.0  1.0  0.0  0.0\n 1.0  1.0  1.0  1.0  1.0  0.0  0.0\n 1.0  1.0  1.0  1.0  1.0  0.0  0.0\n 1.0  1.0  1.0  1.0  1.0  0.0  0.0\n 1.0  1.0  1.0  1.0  1.0  0.0  0.0\n 1.0  1.0  1.0  1.0  1.0  0.0  0.0\n\n\n\n\n\n\n","category":"type"},{"location":"api/#NeuralAttentionlib.LocalMask","page":"API Reference","title":"NeuralAttentionlib.LocalMask","text":"LocalMask(width::Int) <: AbstractDatalessMask\n\nAttention mask that only allow local (diagonal like) values to pass.\n\nwidth should be ≥ 0 and A .* LocalMask(1) is similar to Diagonal(A)\n\n\n\n\n\n","category":"type"},{"location":"api/#NeuralAttentionlib.RandomMask","page":"API Reference","title":"NeuralAttentionlib.RandomMask","text":"RandomMask(p::Float64) <: AbstractDatalessMask\n\nAttention mask that block value randomly.\n\np specify the percentage of value to block. e.g. A .* RandomMask(0) is equivalent to identity(A) and  A .* RandomMask(1) is equivalent to zero(A).\n\n\n\n\n\n","category":"type"},{"location":"api/#NeuralAttentionlib.RepeatMask","page":"API Reference","title":"NeuralAttentionlib.RepeatMask","text":"RepeatMask(mask::AbstractMask, num::Int) <: AbstractWrapperMask\n\nAttention mask wrapper over array mask for doing inner repeat on the last dimension.\n\nExample\n\njulia> m = SymLengthMask([2,3])\nSymLengthMask{1, Vector{Int32}}(Int32[2, 3])\n\njulia> trues(3,3, 2) .* m\n3×3×2 BitArray{3}:\n[:, :, 1] =\n 1  1  0\n 1  1  0\n 0  0  0\n\n[:, :, 2] =\n 1  1  1\n 1  1  1\n 1  1  1\n\njulia> trues(3,3, 4) .* m\nERROR: DimensionMismatch(\"arrays could not be broadcast to a common size; mask require 3-th dimension to be 2, but get 4\")\nStacktrace:\n[...]\n\njulia> trues(3,3, 4) .* RepeatMask(m, 2)\n3×3×4 BitArray{3}:\n[:, :, 1] =\n 1  1  0\n 1  1  0\n 0  0  0\n\n[:, :, 2] =\n 1  1  0\n 1  1  0\n 0  0  0\n\n[:, :, 3] =\n 1  1  1\n 1  1  1\n 1  1  1\n\n[:, :, 4] =\n 1  1  1\n 1  1  1\n 1  1  1\n\n\n\n\n\n\n","category":"type"},{"location":"api/#NeuralAttentionlib.RevBiLengthMask","page":"API Reference","title":"NeuralAttentionlib.RevBiLengthMask","text":"RevBiLengthMask(q_len::A, k_len::A) where {A <: AbstractArray{Int, N}} <: AbstractArrayMask\n\nBiLengthMask but counts from the end of array, used for left padding.\n\nExample\n\njulia> bm = RevBiLengthMask([2,3], [3, 5])\nRevBiLengthMask{1, Vector{Int32}}(Int32[2, 3], Int32[3, 5])\n\njulia> trues(5,5, 2) .* bm\n5×5×2 BitArray{3}:\n[:, :, 1] =\n 0  0  0  0  0\n 0  0  0  0  0\n 0  0  0  1  1\n 0  0  0  1  1\n 0  0  0  1  1\n\n[:, :, 2] =\n 0  0  1  1  1\n 0  0  1  1  1\n 0  0  1  1  1\n 0  0  1  1  1\n 0  0  1  1  1\n\n\nSee also: RevLengthMask, RevSymLengthMask, BatchedMask, RepeatMask\n\n\n\n\n\n","category":"type"},{"location":"api/#NeuralAttentionlib.RevLengthMask","page":"API Reference","title":"NeuralAttentionlib.RevLengthMask","text":"RevLengthMask(len::AbstractArray{Int, N}) <: AbstractSequenceMask\n\nLengthMask but counts from the end of array, used for left padding.  Can be convert to attention mask (RevSymLengthMask, RevBiLengthMask) with AttenMask.\n\nExample\n\njulia> ones(7, 7, 2) .* RevLengthMask([3, 5])\n7×7×2 Array{Float64, 3}:\n[:, :, 1] =\n 0.0  0.0  0.0  0.0  1.0  1.0  1.0\n 0.0  0.0  0.0  0.0  1.0  1.0  1.0\n 0.0  0.0  0.0  0.0  1.0  1.0  1.0\n 0.0  0.0  0.0  0.0  1.0  1.0  1.0\n 0.0  0.0  0.0  0.0  1.0  1.0  1.0\n 0.0  0.0  0.0  0.0  1.0  1.0  1.0\n 0.0  0.0  0.0  0.0  1.0  1.0  1.0\n\n[:, :, 2] =\n 0.0  0.0  1.0  1.0  1.0  1.0  1.0\n 0.0  0.0  1.0  1.0  1.0  1.0  1.0\n 0.0  0.0  1.0  1.0  1.0  1.0  1.0\n 0.0  0.0  1.0  1.0  1.0  1.0  1.0\n 0.0  0.0  1.0  1.0  1.0  1.0  1.0\n 0.0  0.0  1.0  1.0  1.0  1.0  1.0\n 0.0  0.0  1.0  1.0  1.0  1.0  1.0\n\n\n\n\n\n\n","category":"type"},{"location":"api/#NeuralAttentionlib.RevSymLengthMask","page":"API Reference","title":"NeuralAttentionlib.RevSymLengthMask","text":"RevSymLengthMask(len::AbstractArray{Int, N}) <: AbstractArrayMask\n\nSymLengthMask but counts from the end of array, used for left padding.\n\nExample\n\njulia> m = RevSymLengthMask([2,3])\nRevSymLengthMask{1, Vector{Int32}}(Int32[2, 3])\n\njulia> trues(3,3, 2) .* m\n3×3×2 BitArray{3}:\n[:, :, 1] =\n 0  0  0\n 0  1  1\n 0  1  1\n\n[:, :, 2] =\n 1  1  1\n 1  1  1\n 1  1  1\n\n\nSee also: BiLengthMask, BatchedMask, RepeatMask\n\n\n\n\n\n","category":"type"},{"location":"api/#NeuralAttentionlib.SymLengthMask","page":"API Reference","title":"NeuralAttentionlib.SymLengthMask","text":"SymLengthMask(len::AbstractArray{Int, N}) <: AbstractArrayMask\n\nAttention mask specified by an array of integer that indicate the length dimension size.  assuming Query length and Key length are the same.\n\nExample\n\njulia> m = SymLengthMask([2,3])\nSymLengthMask{1, Vector{Int32}}(Int32[2, 3])\n\njulia> trues(3,3, 2) .* m\n3×3×2 BitArray{3}:\n[:, :, 1] =\n 1  1  0\n 1  1  0\n 0  0  0\n\n[:, :, 2] =\n 1  1  1\n 1  1  1\n 1  1  1\n\n\nSee also: LengthMask, BiLengthMask, BatchedMask, RepeatMask\n\n\n\n\n\n","category":"type"},{"location":"api/#Base.:!-Tuple{NeuralAttentionlib.AbstractMask}","page":"API Reference","title":"Base.:!","text":"!m::AbstractMask\n\nBoolean not of an attention mask\n\n\n\n\n\n","category":"method"},{"location":"api/#Base.:&-Tuple{NeuralAttentionlib.AbstractMask, NeuralAttentionlib.AbstractMask}","page":"API Reference","title":"Base.:&","text":"m1::AbstractMask & m2::AbstractMask\n\nlogical and of two attention mask\n\n\n\n\n\n","category":"method"},{"location":"api/#Base.:|-Tuple{NeuralAttentionlib.AbstractMask, NeuralAttentionlib.AbstractMask}","page":"API Reference","title":"Base.:|","text":"m1::AbstractMask | m2::AbstractMask\n\nlogical or of two attention mask\n\n\n\n\n\n","category":"method"},{"location":"api/#NeuralAttentionlib.AttenMask","page":"API Reference","title":"NeuralAttentionlib.AttenMask","text":"AttenMask(m::AbstractMask)\n\nconvert mask into corresponding attention mask\n\n\n\n\n\n","category":"function"},{"location":"api/#NeuralAttentionlib.getmask","page":"API Reference","title":"NeuralAttentionlib.getmask","text":"getmask(m::AbstractMask, score, scale = 1)\n\nConvert m into mask array of AbstractArray for score with scale.\n\nExample\n\njulia> getmask(CausalMask(), randn(7,7), 2)\n7×7 Matrix{Float64}:\n 2.0  2.0  2.0  2.0  2.0  2.0  2.0\n 0.0  2.0  2.0  2.0  2.0  2.0  2.0\n 0.0  0.0  2.0  2.0  2.0  2.0  2.0\n 0.0  0.0  0.0  2.0  2.0  2.0  2.0\n 0.0  0.0  0.0  0.0  2.0  2.0  2.0\n 0.0  0.0  0.0  0.0  0.0  2.0  2.0\n 0.0  0.0  0.0  0.0  0.0  0.0  2.0\n\n\n\n\n\n\n","category":"function"},{"location":"api/#Matmul","page":"API Reference","title":"Matmul","text":"","category":"section"},{"location":"api/","page":"API Reference","title":"API Reference","text":"Modules = [NeuralAttentionlib, NeuralAttentionlib.Matmul]\nPages   = [\"collapseddim.jl\", \"matmul.jl\"]","category":"page"},{"location":"api/#NeuralAttentionlib.CollapsedDimsArray","page":"API Reference","title":"NeuralAttentionlib.CollapsedDimsArray","text":"CollapsedDimsArray{T}(array, ni::Integer, nj::Integer) <: AbstractArray{T, 3}\n\nSimilar to lazy reshape array with collapsed_size\n\n\n\n\n\n","category":"type"},{"location":"api/#NeuralAttentionlib.collapsed_size","page":"API Reference","title":"NeuralAttentionlib.collapsed_size","text":"collapsed_size(x, ni, nj [, n])::Dim{3}\n\nCollapse the dimensionality of x into 3 according to ni and nj where ni, nj specify the number of  second and third dimensions it take.\n\n(X1, X2, ..., Xk, Xk+1, Xk+2, ..., Xk+ni, Xk+ni+1, ..., Xn)\n |______dim1___|  |_________ni_________|  |______nj______|\n\nExample\n\njulia> x = randn(7,6,5,4,3,2);\n\njulia> collapsed_size(x, 2, 2, 1)\n42\n\njulia> collapsed_size(x, 2, 2, 2)\n20\n\njulia> collapsed_size(x, 2, 2, 3)\n6\n\njulia> collapsed_size(x, 2, 2)\n(42, 20, 6)\n\n\nSee also: noncollapsed_size\n\n\n\n\n\n","category":"function"},{"location":"api/#NeuralAttentionlib.collapseddims-Tuple{AbstractArray, Any, Any}","page":"API Reference","title":"NeuralAttentionlib.collapseddims","text":"collapseddims(x::AbstractArray, xi, xj)\n\nReshape x into 3 dim array, equivalent to reshape(x, collapsed_size(x, xi, xj))\n\nSee also: collapsed_size\n\n\n\n\n\n","category":"method"},{"location":"api/#NeuralAttentionlib.collapseddims-Tuple{NeuralAttentionlib.CollapsedDimsArray}","page":"API Reference","title":"NeuralAttentionlib.collapseddims","text":"collapseddims(ca::CollapsedDimsArray)\n\nremove the wrapper and really reshape it.\n\nSee also: CollapsedDimsArray, unwrap_collapse\n\n\n\n\n\n","category":"method"},{"location":"api/#NeuralAttentionlib.matmul","page":"API Reference","title":"NeuralAttentionlib.matmul","text":"matmul(a::AbstractArray, b::AbstractArray, s::Number = 1)\n\nEquivalent to s .* (a * b) if a and b are Vector or Matrix. For array with higher dimension,  it will convert a and b to CollapsedDimsArray and perform batched matrix multiplication, and then  return the result as CollapsedDimsArray. This is useful for preserving the dimensionality. If the batch dimension  of a and b have different shape, it pick the shape of b for batch dimension. Work with NNlib.batch_transpose  and NNlib.batch_adjoint.\n\nExample\n\n# b-dim shape: (6,)\njulia> a = CollapsedDimsArray(randn(3,4,2,3,6), 2, 1); size(a)\n(12, 6, 6)\n\n# b-dim shape: (3,1,2)\njulia> b = CollapsedDimsArray(randn(6,2,3,1,2), 1, 3); size(b)\n(6, 2, 6)\n\njulia> c = matmul(a, b); size(c), typeof(c)\n((12, 2, 6), CollapsedDimsArray{Float64, Array{Float64, 6}, Static.StaticInt{1}, Static.StaticInt{3}})\n\n# b-dim shape: (3,1,2)\njulia> d = unwrap_collapse(c); size(d), typeof(d)\n((3, 4, 2, 3, 1, 2), Array{Float64, 6})\n\n# equivanlent to `batched_mul` but preserve shape\njulia> NNlib.batched_mul(collapseddims(a), collapseddims(b)) == collapseddims(matmul(a, b))\ntrue\n\n\nSee also: CollapsedDimsArray, unwrap_collapse, collapseddims\n\n\n\n\n\n","category":"function"},{"location":"api/#NeuralAttentionlib.noncollapsed_size","page":"API Reference","title":"NeuralAttentionlib.noncollapsed_size","text":"noncollapsed_size(x, ni, nj [, n])\n\nCollapse the dimensionality of x into 3 according to ni and nj.\n\n(X1, X2, ..., Xk, Xk+1, Xk+2, ..., Xk+ni, Xk+ni+1, ..., Xn)\n |______dim1___|  |_________ni_________|  |______nj______|\n\nBut take the size before collapse. e.g. noncollapsed_size(x, ni, nj, 2) will be (Xi, Xi+1, ..., Xj-1).\n\nExample\n\njulia> x = randn(7,6,5,4,3,2);\n\njulia> noncollapsed_size(x, 2, 2, 1)\n(7, 6)\n\njulia> noncollapsed_size(x, 2, 2, 2)\n(5, 4)\n\njulia> noncollapsed_size(x, 2, 2, 3)\n(3, 2)\n\njulia> noncollapsed_size(x, 2, 2)\n((7, 6), (5, 4), (3, 2))\n\n\nSee also: collapsed_size\n\n\n\n\n\n","category":"function"},{"location":"api/#NeuralAttentionlib.scaled_matmul","page":"API Reference","title":"NeuralAttentionlib.scaled_matmul","text":"scaled_matmul(a::AbstractArray, b::AbstractArray, s::Number = 1)\n\nBasically equivalent to unwrap_collapse(matmul(a, b, s)), but not differentiable w.r.t. to s.\n\n\n\n\n\n","category":"function"},{"location":"api/#NeuralAttentionlib.unwrap_collapse","page":"API Reference","title":"NeuralAttentionlib.unwrap_collapse","text":"unwrap_collapse(ca::CollapsedDimsArray)\n\nReturn the underlying array of CollapsedDimsArray, otherwise just return the input.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = NeuralAttentionlib","category":"page"},{"location":"#[NeuralAttentionlib](https://github.com/chengchingwen/NeuralAttentionlib.jl)","page":"Home","title":"NeuralAttentionlib","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Reusable functionality for defining custom attention/transformer layers.","category":"page"},{"location":"","page":"Home","title":"Home","text":"NeuralAttentionlib.jl aim to be highly extendable and reusable function for implementing attention variants.  Will be powering Transformers.jl.","category":"page"},{"location":"#Outline","page":"Home","title":"Outline","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Pages = [\n\t\"term.md\",\n\t\"example.md\",\n\t\"api.md\",\n]","category":"page"},{"location":"example/#Example","page":"Example","title":"Example","text":"","category":"section"},{"location":"example/#Comparing-to-the-existing-implementation-in-Transformers.jl","page":"Example","title":"Comparing to the existing implementation in Transformers.jl","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"See the code in the NeuralAttentionlib's test,  where we compare output/gradient from NeuralAttenionlib v.s. the MultiheadAttention layer from Transformers.jl.  This should provide enough knowledge for implementing a multi-head QKV attention layer with DL framework like Flux.jl.","category":"page"}]
}
