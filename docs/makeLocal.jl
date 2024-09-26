using Documenter
using InteractiveUtils
using AbstractTrees
using XCALibre

# push!(LOAD_PATH,"../src/") # for local build only
ENV["GKSwstype"] = "100"

USER_GUIDE_PAGES = Any[
    "0_introduction_and_workflow.md",
    "1_preprocessing.md",
    "2_physics_and_models.md",
    "3_numerical_setup.md",
    "4_runtime_and_solvers.md",
    "5_postprocessing.md"
]

EXAMPLES_PAGES = Any[
    "01_2d-isothermal-backward-facing-step.md",
    "02_2d-incompressible-transient-cylinder.md",
    "03_2d-constant-temperature-flat-plate.md",
    "04_2d-aerofoil-inflow-optimisation.md",
    "05_2d-inflow-using-Flux.md"
]

makedocs(
    sitename = "XCALibre.jl",
    format = Documenter.HTML(),
    doctest = false, # only set to false when sorting out docs structure
    modules = [XCALibre],
    pages = [
        "Home" => "index.md",
        "quick_start.md",
        "Examples" => "examples/" .* EXAMPLES_PAGES,
        "User Guide" => "user_guide/" .* USER_GUIDE_PAGES,
        hide("Theory Guide" => "theory_guide/introduction.md"),
        "contributor_guide.md",
        "reference.md",
        "release_notes.md"
    ]
)

# foreach(rm, filter(endswith(".vtk"), readdir("docs", join=true)))
# foreach(rm, filter(endswith(".vtu"), readdir("docs", join=true)))