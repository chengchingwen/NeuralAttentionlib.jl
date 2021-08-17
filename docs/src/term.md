# Terminology

Term and Naming explanation.
    
## Prerequisite

Some term for better understanding this docs.

### 1. [PartialFunctions](https://github.com/archermarx/PartialFunctions.jl)

This actually live outside the scope of this package, but is extremely useful for illustrate the overall design.
 We'll use the `$` operation to denote partial function application 
 (i.e. `f $ x` is equivanlent to `(arg...)->f(x, arg...)`).
 

### 2. Feature / Length / Batch Dimension

Under the context of attention operation in deep learning, the input data can be viewed as a 3-dimensional array.
 The *feature* dimension, the *length* dimension, and the *batch* dimension (f-dim, l-dim, b-dim for short).
 Following the Julia's multidimensional array implementation
 ([column-major](https://en.wikipedia.org/wiki/Row-_and_column-major_order#Neither_row-major_nor_column-major)),
 the data is store in a `AbstractArray{T, 3}` whose size is `(f-dim, l-dim, b-dim)`.

For example, given 3 sentence as a batch, each sentence have 10 word, and we choose to represent a word with
 a vector of 32 element. This data will be store in an 3-dim array with size `(32, 10, 3)`.
 
General speaking, *batch* stands for how many independent data you are going to run in one function call,
 usually just for performance/optimization need. *length* means how many entry you have for each data sample,
 like the #-words in a sentence or #-pixels in an image. *feature* is the number of value you used to
 represent an entry.


## Attention

The overall attention operation can be viewed as three mutually inclusive block:

```
	     (main input)
	        Value           Key             Query  (Extras...)
	+---------|--------------|----------------|------|||---- Attention Operation ---+
	|         |              |                |      |||                            |
	|         |              |                |      |||   multihead, ...           |
	|         |              |                |      |||                            |
	|   +-----|--------------|----------------|------|||-----------------------+    |
	|   |     |              |                |      |||                       |    |
	|   |     |          +---|----------------|------|||-------------+         |    |
	|   |     |          |   |                |      |||             |         |    |
	|   |     |          |   |  scoring func  |      |||             |         |    |
	|   |     |          |   +------>+<-------+<=======+             |         |    |
	|   |     |          |           |                               |         |    |
	|   |     |          |           | masked_score,                 |         |    |
	|   |     |          |           | normalized_score,             |         |    |
	|   |     |          |           | ...                           |         |    |
	|   |     |          |           |                               |         |    |
	|   |     |          +-----------|------------ Attention Score --+         |    |
	|   |     |                      |                                         |    |
	|   |     |     mixing func      |                                         |    |
	|   |     +--------->+<----------+                                         |    |
	|   |                |                                                     |    |
	|   +----------------|------------------------------- Mixing --------------+    |
	|                    |                                                          |
	+--------------------|----------------------------------------------------------+
	              Attentive Value
	               (main output)
```

The attention operation is actually a special way to "mix" (or "pick" in common lecture) the input information.
 In (probably) the first [attention paper](https://arxiv.org/abs/1409.0473), the attention is defined as weighted
 sum of the input sequence given a word embedding. The idea is furthur generalize to *QKV attention* in the first
 [transformer paper](https://arxiv.org/abs/1706.03762). 

### 1. Attention Score

The attention score is used to decide how much the each piece of input information will contribute to the
 output value and also how many entry the attention operation will output. The operation that will modify
 the attention score matrix should be consider as part of this block. For example: Different attention masks
 (local attention, random attention, ...), normalization (softmax, l2-norm, ...), and some special attention
 that take other inputs (transformer decoder, relative position encoding, ...).

### 2. Mixing

We refer to the operation that take the attention score and input value as "mixing". Usually it's just a
 weighted sum over the input value and use the attention score as the weight.

### 3. Attention Operation

The whole scoring + mixing and other pre/post processing made up an attention operation. Things like handling
 multi-head should happen at this level.


## Attention Mask

Attention masks are a bunch of operation that modified the attention score.

### 1. Dataless mask

We use "dataless" to refer to masks that are independent to the input. For example, `CausalMask` works the same
 on each data regardless of the batch size or the data content.

### 2. Array mask

We call the mask that is dependent to the input as "array mask". For example, `SymLengthMask` is used to avoid
 the padding token being considered in the attention operation, thus each data batch might have different mask value.
