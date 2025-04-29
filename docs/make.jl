
push!(LOAD_PATH, "../src/")

using Documenter, MuSink

makedocs(
  sitename="MuSink.jl Documentation",
  pages = [
    "Overview" => "index.md",
    "Usage" => "usage.md",
    "API Reference" => "reference.md",
  ]
)

deploydocs(
  repo = "github.com/tscode/MuSink.jl.git",
)
