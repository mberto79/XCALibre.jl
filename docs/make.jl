using Documenter
using XCALibre

push!(LOAD_PATH,"../src/")

makedocs(
    sitename = "XCALibre",
    format = Documenter.HTML(),
    doctest = false, # only set to false when sorting out docs structure
    modules = [XCALibre]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
