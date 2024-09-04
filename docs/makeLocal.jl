using Documenter
using XCALibre

# push!(LOAD_PATH,"../src/") # for local build only

makedocs(
    sitename = "XCALibre.jl",
    format = Documenter.HTML(),
    doctest = false, # only set to false when sorting out docs structure
    modules = [XCALibre],
    pages = [
        "Home" => "index.md",
        "quick_start.md",
        "Verification & validation" => Any[
            "VV/2d-isothermal-backward-facing-step.md",
            "VV/2d-constant-temperature-flat-plate.md"
        ],
        "User Guide" => Any[
            "user_guide/workflow.md",
            "user_guide/mesh.md",
            "user_guide/physics.md",
            "user_guide/boundary_conditions.md",
            "user_guide/discretisation_schemes.md",
            "user_guide/linear_solvers.md",
            "user_guide/runtime_configuration.md",
            "user_guide/flow_solvers.md"
        ],
        "Theory Guide" => "theory_guide/introduction.md",
        "contributor_guide.md",
        "reference.md",
        "release_notes.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
# deploydocs(
#     repo = "github.com/mberto79/XCALibre.jl.git",
#     devbranch = "dev-0.3-main"
# )
