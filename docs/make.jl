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

foreach(rm, filter(endswith(".vtk"), readdir("docs", join=true)))
foreach(rm, filter(endswith(".vtu"), readdir("docs", join=true)))
