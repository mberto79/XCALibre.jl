using Documenter
using FVM_1D

push!(LOAD_PATH,"../src/")

makedocs(
    sitename = "FVM_1D",
    format = Documenter.HTML(),
    doctest = false, # only set to false when sorting out docs structure
    modules = [FVM_1D]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
