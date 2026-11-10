# Changelog entries

Changes waiting for the next release live in `changelog/unreleased/`, one file per pull request, instead of in `CHANGELOG.md`. Separate files mean pull requests never conflict over the changelog.

## Adding an entry

Create `changelog/unreleased/<branch-name>.md` with one line per entry, each starting with its category:

```
* added: `Richardson()` solver, running repeated AMG V-cycles.
* breaking: `AMG()` is now a preconditioner, `SolverSetup(solver=Cg(), preconditioner=AMG())`.
* fixed: `Bicgstab()` no longer returns NaN on breakdown. [#238](@ref)
```

* Categories: `added`, `changed`, `fixed`, `breaking`, `deprecated`, `removed`.
* One entry per line; blank lines are ignored.
* The `[#N](@ref)` link is optional: entries without one get the number of the pull request that added the file when the release is assembled. Write links yourself to reference several pull requests.
* Small changes users would not notice need no entry.

## Releasing

```bash
julia changelog/assemble.jl          # preview the next release section
julia changelog/assemble.jl v0.7.0   # write it into CHANGELOG.md and delete the entries
```

Review the `CHANGELOG.md` diff and commit. The documentation shows unreleased entries in an "Unreleased" section.
