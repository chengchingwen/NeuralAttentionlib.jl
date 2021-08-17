using NeuralAttentionlib
using Documenter

DocMeta.setdocmeta!(NeuralAttentionlib, :DocTestSetup, :(using NeuralAttentionlib); recursive=true)

makedocs(;
    modules=[NeuralAttentionlib],
    authors="chengchingwen <adgjl5645@hotmail.com> and contributors",
    repo="https://github.com/chengchingwen/NeuralAttentionlib.jl/blob/{commit}{path}#{line}",
    sitename="NeuralAttentionlib.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://chengchingwen.github.io/NeuralAttentionlib.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Terminology" => "term.md",
        "Example" => "example.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/chengchingwen/NeuralAttentionlib.jl",
)
