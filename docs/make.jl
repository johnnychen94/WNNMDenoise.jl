using WNNMDenoise
using Documenter

makedocs(;
    modules=[WNNMDenoise],
    authors="Johnny Chen <johnnychen94@hotmail.com>",
    repo="https://github.com/johnnychen94/WNNMDenoise.jl/blob/{commit}{path}#L{line}",
    sitename="WNNMDenoise.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://johnnychen94.github.io/WNNMDenoise.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/johnnychen94/WNNMDenoise.jl",
)
