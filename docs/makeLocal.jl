using Documenter
using XCALibre

# push!(LOAD_PATH,"../src/") # for local build only

USER_GUIDE_PAGES = Any[
    "0_introduction_and_workflow.md",
    "1_preprocessing.md",
    "2_physics_and_models.md",
    "3_numerical_setup.md",
    "4_runtime_and_solvers.md",
    "5_postprocessing.md"
]

makedocs(
    sitename = "XCALibre.jl",
    format = Documenter.HTML(),
    # doctest = false, # only set to false when sorting out docs structure
    modules = [XCALibre],
    pages = [
        "Home" => "index.md",
        "quick_start.md",
        "Verification & validation" => Any[
            "VV/2d-isothermal-backward-facing-step.md",
            "VV/2d-constant-temperature-flat-plate.md"
        ],
        "User Guide" => "user_guide/" .* USER_GUIDE_PAGES,
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
