# API Reference

```@index
Order   = [:function, :type]
```

## Functional

```@autodocs
Modules = [NeuralAttentionlib, NeuralAttentionlib.Functional, NeuralAttentionlib.Masks]
Pages   = ["functional.jl"]
```

```@autodocs
Modules = [NeuralAttentionlib]
Pages   = ["types.jl", "utils.jl"]
Filter  = t -> !(t isa typeof(NeuralAttentionlib.var"@imexport"))
```

## Mask

```@autodocs
Modules = [NeuralAttentionlib, NeuralAttentionlib.Masks]
Pages   = ["mask.jl"]
```

## Matmul

```@autodocs
Modules = [NeuralAttentionlib, NeuralAttentionlib.Matmul]
Pages   = ["collapseddim.jl", "matmul.jl"]
```
