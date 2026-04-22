# Repository Guidelines

## Project Structure & Module Organization
`src/` contains the package entrypoint `XCALibre.jl` plus domain modules such as `Mesh/`, `Discretise/`, `Solve/`, `Solvers/`, `ModelPhysics/`, and `Postprocess/`. Keep new functionality inside the closest existing submodule and follow the current split-file pattern such as `Module_0_types.jl`, `Module_1_functions.jl`, or feature-specific files under `boundary_conditions/`.

`test/` mixes unit tests (`unit_test_*.jl`, `test_*.jl`) with larger functional cases in `test/0_TEST_CASES/`. `docs/` holds the Documenter setup and published pages under `docs/src/`. `examples/` contains runnable case scripts, while `ext/` contains optional GPU backend extensions.

## Build, Test, and Development Commands
Use Julia 1.10 or 1.11 for parity with CI.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. test/runtests.jl
julia --project=docs/ -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs/ docs/make.jl
```

`Pkg.test()` runs the full suite. Running `test/runtests.jl` directly is useful when iterating locally. Build docs before changing user-facing APIs, examples, or release notes.

## Coding Style & Naming Conventions
Follow the Julia style guide and the established repository conventions: 4-space indentation, CamelCase for types, and descriptive lowercase names with underscores for functions, for example `calculate_flux`. User-facing API names should stay ASCII where practical; limited Unicode is already used internally for mathematical variables.

Prefer small, focused files that are included from the parent module. Match existing naming patterns when adding tests, solvers, or boundary conditions.

## Testing Guidelines
Add or update tests with every behavioral change. Place focused checks in `test/unit_test_*.jl` or `test/test_*.jl`; put end-to-end solver cases in `test/0_TEST_CASES/`. Keep test filenames descriptive, mirroring the feature or physics model they cover, such as `2d_laplace_steady.jl`.

## Commit & Pull Request Guidelines
Recent commits use short, imperative summaries such as `change solver name to GODUNOV and add tests and update changelog`. Keep commit titles brief, specific, and action-oriented. PRs should explain the numerical or API impact, list added tests, and note any documentation updates. Link related issues, and include figures only when output or docs visuals changed.
