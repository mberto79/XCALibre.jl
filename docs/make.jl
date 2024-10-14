# Use same base setup used for local builds

include("makeLocal.jl")

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/mberto79/XCALibre.jl.git",
    devurl = "dev",
    versions = ["stable" => "v^", "dev" => "dev"],
    devbranch = "main"
)
